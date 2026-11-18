// Host-only stand-in for the HIP runtime, for conversation_copy_bounce_test.cpp only.  A stream is a FIFO of deferred
// copies that run when an event recorded after them, or the stream, is synchronized - so a bounce buffer reused
// before its copy ran shows up as wrong bytes.  Every pointer the "runtime" is handed is recorded.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <set>
#include <vector>

typedef int cudaError_t;
typedef struct FakeStream* cudaStream_t;
typedef struct FakeEvent* cudaEvent_t;
enum cudaMemcpyKind { cudaMemcpyDefault = 4 };
constexpr cudaError_t cudaSuccess = 0, cudaErrorInvalidValue = 1;
constexpr unsigned cudaStreamNonBlocking = 1, cudaEventDisableTiming = 2, cudaHostAllocPortable = 1;

namespace fake {
struct Op { size_t seq; std::function<void()> run; };
inline std::deque<Op>& queue() { static std::deque<Op> q; return q; }
inline size_t& seq() { static size_t s = 0; return s; }
inline std::set<const void*>& pinned() { static std::set<const void*> p; return p; }   // the runtime's own host memory
inline std::vector<std::pair<const void*, size_t>>& handed() { static std::vector<std::pair<const void*, size_t>> h; return h; }
inline void run_until(size_t s) {
    while (!queue().empty() && queue().front().seq <= s) { queue().front().run(); queue().pop_front(); }
}
}  // namespace fake
struct FakeEvent { size_t seq = 0; };

inline const char* cudaGetErrorString(cudaError_t) { return "fake"; }
inline cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) { *s = reinterpret_cast<cudaStream_t>(1); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind, cudaStream_t) {
    fake::handed().push_back({dst, n});
    fake::handed().push_back({src, n});
    fake::queue().push_back({++fake::seq(), [=] { std::memcpy(dst, src, n); }});
    return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { fake::run_until(fake::seq()); return cudaSuccess; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { *e = new FakeEvent; return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t) { e->seq = fake::seq(); return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t e) { fake::run_until(e->seq); return cudaSuccess; }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) { return cudaSuccess; }
template<class T> inline cudaError_t cudaHostAlloc(T** p, size_t n, unsigned) {
    *p = reinterpret_cast<T*>(new uint8_t[n]);
    fake::pinned().insert(*p);
    return cudaSuccess;
}
