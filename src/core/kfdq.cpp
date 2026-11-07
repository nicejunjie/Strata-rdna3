// [claude] 2026-11-07 kfdq: park ROCr's idle utility queue while the engine is idle (Linux, AMD KFD).
//
// gfx1100 gives a process 8 compute queue slots a card. With GPU_MAX_HW_QUEUES=8 (the 2-GPU setting) HIP opens 8 queues
// and ROCr adds its internal utility queue (64 packets; GpuAgent::QueueCreate creates it before the first user queue,
// used for blit-kernel copies and cache invalidates): 9 > 8, so MES time-slices forever and both cards sit at 100% busy,
// ~95 W, while idle. Fewer HIP queues fit but decode 2-5% slower. This keeps the 9 queues while the engine submits work
// and deactivates the utility queue (AMDKFD_IOC_UPDATE_QUEUE, percentage 0 - the call ROCr itself uses - so MES unmaps it)
// after STRATA_KFDQ_IDLE_MS without any write-pointer change on any queue; the next submission reactivates it within
// STRATA_KFDQ_POLL_US. Only a card holding more than STRATA_KFDQ_SLOTS (8) queues is touched.
// Measured (2x RX 7900 XTX, IQ3/IQ4/Q4, 1K-128K): idle 98 + 91 W -> 5 + 13 W, prefill and ms/window equal.
//
// Mechanism: the strata executable defines and exports ioctl() (CMake: --export-dynamic-symbol=ioctl), so ROCr's
// KFD calls come here first; we record each queue's ring and read/write pointers and forward to libc.
// Env: STRATA_KFDQ=0 off; STRATA_KFDQ_SLOTS (8); STRATA_KFDQ_IDLE_MS (2000); STRATA_KFDQ_POLL_US (2000);
//      STRATA_KFDQ_LOG=1 logs queue creation, park and unpark to stderr.
#if defined(__linux__) && __has_include(<linux/kfd_ioctl.h>)
#include <dlfcn.h>
#include <linux/kfd_ioctl.h>
#include <pthread.h>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <unistd.h>

namespace {

constexpr int kMaxQueues = 128;
struct Queue {
    int used, fd, parked, util;
    kfd_ioctl_create_queue_args a;   // as created (queue_id filled by KFD)
    uint64_t last_w;
};

Queue qs[kMaxQueues];
pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
using IoctlFn = int (*)(int, unsigned long, ...);
IoctlFn real_ioctl;
int enabled = -1, logon;
long idle_ms = 2000, poll_us = 2000, slots = 8;
int th_started;
uint32_t util_gpu_seen[16];
int n_util_gpu;

long now_ms() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000L + t.tv_nsec / 1000000L;
}

long env_long(const char* name, long def) {
    const char* e = getenv(name);
    return e && *e ? atol(e) : def;
}

int set_pct(Queue& q, uint32_t pct) {
    kfd_ioctl_update_queue_args u{};
    u.ring_base_address = q.a.ring_base_address;
    u.queue_id = q.a.queue_id;
    u.ring_size = q.a.ring_size;
    u.queue_percentage = pct;
    u.queue_priority = q.a.queue_priority;
    return real_ioctl(q.fd, AMDKFD_IOC_UPDATE_QUEUE, &u);
}

uint32_t run_pct(const Queue& q) { return q.a.queue_percentage ? q.a.queue_percentage : 100; }

int gpu_queues(uint32_t gpu) {
    int n = 0;
    for (auto& q : qs) n += q.used && q.a.gpu_id == gpu;
    return n;
}

void* watcher(void*) {
    long last_act = now_ms();
    bool parked = false;
    for (;;) {
        usleep(parked ? poll_us : 20000);
        bool act = false;
        pthread_mutex_lock(&mu);
        for (auto& q : qs) {
            if (!q.used) continue;
            uint64_t w = *reinterpret_cast<volatile uint64_t*>(static_cast<uintptr_t>(q.a.write_pointer_address));
            uint64_t r = *reinterpret_cast<volatile uint64_t*>(static_cast<uintptr_t>(q.a.read_pointer_address));
            if (w != q.last_w || r != w) act = true;
            q.last_w = w;
        }
        long t = now_ms();
        if (act) last_act = t;
        if (parked && act) {
            for (auto& q : qs)
                if (q.used && q.parked) {
                    int rc = set_pct(q, run_pct(q));
                    q.parked = 0;
                    if (logon) fprintf(stderr, "[kfdq %ld] unpark gpu %u q %u rc %d\n", (long) time(nullptr), q.a.gpu_id, q.a.queue_id, rc);
                }
            parked = false;
        } else if (!parked && t - last_act >= idle_ms) {
            for (auto& q : qs)
                if (q.used && q.util && gpu_queues(q.a.gpu_id) > slots) {
                    int rc = set_pct(q, 0);
                    if (rc == 0) { q.parked = 1; parked = true; }
                    if (logon) fprintf(stderr, "[kfdq %ld] park gpu %u q %u rc %d\n", (long) time(nullptr), q.a.gpu_id, q.a.queue_id, rc);
                }
            if (!parked) last_act = t;   // nothing over the slots: look again after another idle period
        }
        pthread_mutex_unlock(&mu);
    }
    return nullptr;
}

void init() {
    real_ioctl = reinterpret_cast<IoctlFn>(dlsym(RTLD_NEXT, "ioctl"));
    const char* e = getenv("STRATA_KFDQ");
    enabled = !(e && e[0] == '0');
    idle_ms = env_long("STRATA_KFDQ_IDLE_MS", 2000);
    poll_us = env_long("STRATA_KFDQ_POLL_US", 2000);
    slots = env_long("STRATA_KFDQ_SLOTS", 8);
    logon = (e = getenv("STRATA_KFDQ_LOG")) && e[0] == '1';
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int ioctl(int fd, unsigned long req, ...) {
    va_list ap;
    va_start(ap, req);
    void* arg = va_arg(ap, void*);
    va_end(ap);
    if (enabled < 0) init();
    if (!enabled) return real_ioctl(fd, req, arg);
    if (req == AMDKFD_IOC_CREATE_QUEUE) {
        pthread_mutex_lock(&mu);
        int rc = real_ioctl(fd, req, arg);
        auto* a = static_cast<kfd_ioctl_create_queue_args*>(arg);
        if (rc == 0 && a->queue_type == KFD_IOC_QUEUE_TYPE_COMPUTE_AQL) {
            for (auto& q : qs)
                if (!q.used) {
                    q = Queue{};
                    q.used = 1; q.fd = fd; q.a = *a;
                    // The first AQL queue of each GPU is ROCr's utility queue (created right before the first user
                    // queue, at ROCr's minimum AQL size: 4 KB; HIP queues are 1 MB).
                    bool seen = false;
                    for (int g = 0; g < n_util_gpu; g++) seen |= util_gpu_seen[g] == a->gpu_id;
                    if (!seen && a->ring_size <= 65536 && n_util_gpu < 16) { util_gpu_seen[n_util_gpu++] = a->gpu_id; q.util = 1; }
                    if (logon) fprintf(stderr, "[kfdq %ld] create gpu %u q %u ring %u util %d\n", (long) time(nullptr), a->gpu_id, a->queue_id, a->ring_size, q.util);
                    break;
                }
            if (!th_started) {
                th_started = 1;
                pthread_t th;
                if (pthread_create(&th, nullptr, watcher, nullptr) == 0) pthread_detach(th);
            }
        }
        pthread_mutex_unlock(&mu);
        return rc;
    }
    if (req == AMDKFD_IOC_DESTROY_QUEUE) {
        pthread_mutex_lock(&mu);
        auto* d = static_cast<kfd_ioctl_destroy_queue_args*>(arg);
        for (auto& q : qs)
            if (q.used && q.fd == fd && q.a.queue_id == d->queue_id) {
                if (q.parked) set_pct(q, run_pct(q));
                q.used = 0;
            }
        int rc = real_ioctl(fd, req, arg);
        pthread_mutex_unlock(&mu);
        return rc;
    }
    if (req == AMDKFD_IOC_UPDATE_QUEUE) {
        pthread_mutex_lock(&mu);
        auto* u = static_cast<kfd_ioctl_update_queue_args*>(arg);
        for (auto& q : qs)
            if (q.used && q.fd == fd && q.a.queue_id == u->queue_id) {
                if (u->queue_percentage) q.a.queue_percentage = u->queue_percentage;
                q.a.queue_priority = u->queue_priority;
                q.parked = 0;   // the runtime set it itself: active again
            }
        int rc = real_ioctl(fd, req, arg);
        pthread_mutex_unlock(&mu);
        return rc;
    }
    return real_ioctl(fd, req, arg);
}
#endif
