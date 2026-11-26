// crashfix-20261117 S+N+L, Fable "Ruling on S+N+L 6d" run 3: the shared checkpoint and the unfilled buffers, byte for byte,
// on the REAL runtime with the real conversation_state / snapshot / buffer / file code, production geometry (ModelGeometry
// defaults: 48 layers, 36 GDN -> a 117.7 MB checkpoint, 12 QSA layers, INT8 K/V), park classes 178 / 8512 / 33087 tokens
// (16 MiB segments, last one partial), on every GPU; and a 2-GPU split / merge.  Reference bytes come from a seeded host
// generator, never from the save path.  Exit 0 = every byte matched.
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_file.hpp"
#include "conversation_copy.hpp"
#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace strata::core;
using namespace strata::kernels;
using strata::core::conversation_detail::copy_checkpoint_beating;

namespace {
int checks = 0, fails = 0;
void check(bool ok, const char* what) { ++checks; if (!ok) { ++fails; std::fprintf(stderr, "FAIL: %s\n", what); } }
void ck(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(2); } }
uint8_t gen(uint64_t seed, uint64_t i) { uint64_t x = seed * 0x9E3779B97F4A7C15ull + i * 0xBF58476D1CE4E5B9ull; x ^= x >> 31; x *= 0x94D049BB133111EBull; return uint8_t(x >> 29); }
std::vector<uint8_t> ref(uint64_t seed, size_t n) { std::vector<uint8_t> v(n); for (size_t i = 0; i < n; ++i) v[i] = gen(seed, i); return v; }

struct Dev {   // one device's buffers, freed at the end
    std::vector<void*> mem;
    template<class T> T* alloc(size_t n) { void* p = nullptr; if (n) { ck(cudaMalloc(&p, n)); mem.push_back(p); } return (T*) p; }
    ~Dev() { for (void* p : mem) cudaFree(p); }
};
struct Session {   // a whole-model session of production geometry on the current device, INT8 resident K/V
    ModelGeometry g;
    Dev d;
    std::vector<QsaState> q;
    SessionState ss;
    ConversationStateSizes z;
    QsaState draft;
    std::vector<std::pair<void*, size_t>> parts;   // every device buffer a save reads: running state, then K/V
    explicit Session(int64_t cells) {
        std::string err;
        const int64_t nq = g.n_qsa_layers();
        q.resize((size_t) nq);
        const auto s = qsa_real_shapes();
        auto kv = [&](QsaState& st, bool index) {
            st.kv_mode = 0; st.kv_int8 = true; st.n_pages = (cells + s.page_size - 1) / s.page_size; st.n_slots = st.n_pages;
            st.max_cells = st.n_pages * s.page_size; st.idx_pooled_rows = index ? st.max_cells / s.idx_block + 2 : 0;
            const size_t rows = (size_t) st.max_cells * g.n_head_kv, per = (size_t) g.head_dim, sc = (size_t) (g.head_dim / 64) * 2;
            st.k_q = d.alloc<int8_t>(rows * per); st.v_q = d.alloc<int8_t>(rows * per);
            st.k_scale = d.alloc<uint16_t>(rows * sc); st.v_scale = d.alloc<uint16_t>(rows * sc);
            st.page_table = d.alloc<int32_t>((size_t) st.n_pages * 4);
            parts.push_back({st.k_q, rows * per}); parts.push_back({st.v_q, rows * per});
            parts.push_back({st.k_scale, rows * sc}); parts.push_back({st.v_scale, rows * sc});
            if (index) {
                const size_t pooled = (size_t) st.idx_pooled_rows * g.idx_key_dim * 4;
                st.idx_pooled = d.alloc<float>(pooled); parts.push_back({st.idx_pooled, pooled});
            }
        };
        ss.max_cells = cells; ss.qsa_states = q.data(); ss.layer_hi = g.n_layers;
        ss.gdn_alloc = g.n_gdn_layers(); ss.qsa_alloc = nq;
        check(conversation_session_sizes(g, ss, z, err), "session sizes");
        ss.gdn_state = d.alloc<float>(z.gdn); ss.ple_hist = d.alloc<float>(z.ple);
        parts.push_back({ss.gdn_state, z.gdn}); parts.push_back({ss.ple_hist, z.ple});
        for (auto& st : q) {
            st.idx_tail = d.alloc<float>(z.tail); st.idx_dead = d.alloc<float>(z.dead); st.idx_block_pos = d.alloc<int32_t>(z.block_pos);
            parts.push_back({st.idx_tail, z.tail}); parts.push_back({st.idx_dead, z.dead}); parts.push_back({st.idx_block_pos, z.block_pos});
            kv(st, true);
        }
        kv(draft, false);
        ck(cudaDeviceSynchronize());
    }
    void fill(uint64_t seed) {   // every buffer from the seeded generator (host -> device, blocking)
        for (size_t i = 0; i < parts.size(); ++i) { auto r = ref(seed * 1000 + i, parts[i].second); ck(cudaMemcpy(parts[i].first, r.data(), r.size(), cudaMemcpyHostToDevice)); }
        ck(cudaDeviceSynchronize());
    }
    bool device_is(uint64_t seed, size_t first, size_t count, const std::vector<size_t>& upto_bytes) {   // read back, plain blocking copy
        bool ok = true;
        for (size_t i = first; i < first + count && i < parts.size(); ++i) {
            const size_t n = upto_bytes.empty() ? parts[i].second : std::min(parts[i].second, upto_bytes[i]);
            std::vector<uint8_t> got(n, 0); ck(cudaMemcpy(got.data(), parts[i].first, n, cudaMemcpyDeviceToHost));
            auto r = ref(seed * 1000 + i, n); ok = ok && got == r;
        }
        return ok;
    }
};
bool block_is(const CheckpointBlock& b, uint64_t seed, size_t part) { auto r = ref(seed * 1000 + part, b.size()); return b == r; }
// part indices: 0 gdn, 1 ple, then per QSA layer j: 2+3j+0 tail, +1 dead, +2 block_pos (interleaved with K/V parts below)
bool ckpt_is(const Session& S, const ConversationCheckpoint& c, uint64_t seed) {
    if (!block_is(c.gdn, seed, 0) || !block_is(c.ple, seed, 1)) return false;
    // tails / dead / block_pos are concatenated per owned layer in the checkpoint
    size_t idx = 2; bool ok = true; size_t off_t = 0, off_d = 0, off_b = 0;
    for (size_t j = 0; j < S.q.size(); ++j) {
        auto rt = ref(seed * 1000 + idx, S.z.tail), rd = ref(seed * 1000 + idx + 1, S.z.dead), rb = ref(seed * 1000 + idx + 2, S.z.block_pos);
        ok = ok && std::equal(rt.begin(), rt.end(), c.tails.data() + off_t) && std::equal(rd.begin(), rd.end(), c.dead.data() + off_d) &&
             std::equal(rb.begin(), rb.end(), c.block_pos.data() + off_b);
        off_t += S.z.tail; off_d += S.z.dead; off_b += S.z.block_pos;
        idx += 3 + 5;   // tail, dead, block_pos, then the layer's 5 K/V parts
    }
    return ok;
}
bool kv_is(const Session& S, const SavedConversation& img, uint64_t seed, int64_t upto) {
    // K/V parts of layer j: k, v, k_scale, v_scale, pooled at parts index 2 + 8j + 3 ..; the image holds the first `upto` cells
    bool ok = img.kv.size() == S.q.size() + 1;
    const auto s = qsa_real_shapes();
    const int64_t cells = ((upto + s.page_size - 1) / s.page_size) * s.page_size;
    for (size_t j = 0; ok && j <= S.q.size(); ++j) {
        const bool index = j < S.q.size();
        const size_t base = index ? 2 + 8 * j + 3 : 2 + 8 * S.q.size();
        const ConversationKv& kv = img.kv[j];
        const std::array<const ConversationBuffer*, 5> b = {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
        const size_t per[4] = {(size_t) S.g.n_head_kv * S.g.head_dim, (size_t) S.g.n_head_kv * S.g.head_dim,
                               (size_t) S.g.n_head_kv * (S.g.head_dim / 64) * 2, (size_t) S.g.n_head_kv * (S.g.head_dim / 64) * 2};
        for (size_t p = 0; p < (index ? 5u : 4u); ++p) {
            const size_t n = p < 4 ? per[p] * (size_t) cells : (upto > 0 ? (size_t) (upto / s.idx_block + 1) * S.g.idx_key_dim * 4 : 0);
            std::vector<uint8_t> got(b[p]->size()); ok = ok && b[p]->size() == n && b[p]->read(got.data(), 0, n);
            auto r = ref(seed * 1000 + base + p, n); ok = ok && got == r;
        }
    }
    return ok;
}
void one_device(int dev) {
    ck(cudaSetDevice(dev));
    char bus[64] = {0}; (void) hipDeviceGetPCIBusId(bus, sizeof bus, dev);
    std::printf("device %d %s\n", dev, bus);
    for (int64_t T : {178, 8512, 33087}) {
        Session S(T + 64);
        std::vector<int32_t> ids((size_t) T); for (size_t i = 0; i < ids.size(); ++i) ids[i] = int32_t(i % 1000 + 1);
        std::vector<ConversationImageKey> imgs;
        std::string err;
        // save: device = R1, live save (assign_for_overwrite), all five vectors equal R1
        S.fill(11);
        ConversationCheckpoint ckp; ckp.ids.assign(ids.begin(), ids.begin() + T / 2);
        const bool saved1 = conversation_checkpoint_save(ckp, S.ss, S.g, err);
        if (std::getenv("STRATA_TEST_CKPT_HELPER_DELAY_MS")) check(ckp.gdn.pending(), "(k) with the helper delayed, the block is pending after the save (design O path taken)");
        if (std::getenv("STRATA_TEST_CKPT_STAGING_FAIL")) check(!ckp.gdn.pending(), "(n) staging refused: the save took the bounce path");
        if (!std::getenv("STRATA_TEST_CKPT_STAGING_FAIL")) {   // P1: the copy is held until a release point
            check(ckp.gdn.pending(), "(P1) right after the save the copy is held");
            conversation_checkpoint_release_held(CkptRelease::request_end);   // the request ended before any decode
        }
        check(saved1 && ckpt_is(S, ckp, 11), "live save = R1 (L)");
        check(ckp.gdn.capacity() == ckp.gdn.size() && ckp.gdn.unique(), "capacity == size, unique after save");
        std::vector<ConversationCheckpoint> live{ckp};
        const ConversationView view{ids, imgs, live, false};
        SavedConversation img;
        check(conversation_snapshot_save(img, view, S.ss, S.g, &S.draft, err), "park");
        check(img.checkpoints.size() == 1 && img.checkpoints[0].gdn.data() == live[0].gdn.data() && ckpt_is(S, img.checkpoints[0], 11),
              "the parked checkpoint shares the live block and is R1 (S)");
        check(ckpt_is(S, img.live, 11) && kv_is(S, img, 11, T), "the parked live state and every K/V segment are R1 (N, incl. the partial last segment)");
        // share then rewrite: device = R2, live save again into the shared checkpoint: parked still R1, live R2
        S.fill(22);
        check(conversation_checkpoint_save(live[0], S.ss, S.g, err) && ckpt_is(S, live[0], 22), "live save again = R2");
        check(ckpt_is(S, img.checkpoints[0], 11), "the parked block is still R1 after the live rewrite (copy-on-write)");
        // P1: a save with its copy still held, parked (shared) and then restored with the copy pending: the restore's read of the
        // block reaches the latch, releases the held copy and waits for it
        {
            S.fill(12); ConversationCheckpoint pend; pend.ids = ckp.ids;
            check(conversation_checkpoint_save(pend, S.ss, S.g, err), "(P1) save, copy held");
            ConversationCheckpoint parked_pend = copy_checkpoint_beating(pend);
            S.fill(13);
            check(conversation_checkpoint_restore(parked_pend, S.ss, S.g, err) && S.device_is(12, 0, 2, {}),
                  "(P1) restore with the copy pending: the reader releases it, the device gets R12");
        }
        // restore: the parked image back to the device, read back with a plain blocking copy into zero-filled vectors
        check(conversation_snapshot_restore(img, S.ss, S.g, &S.draft, err) == ConversationRestore::restored, "restore");
        ck(cudaDeviceSynchronize());
        check(S.device_is(11, 0, 2, {}), "restored running state = R1");
        // slot copy / borrow (generate.cpp 13844 / 17058), then drop or rewrite either side
        std::vector<ConversationCheckpoint> slot; slot.push_back(img.checkpoints[0]);
        std::vector<ConversationCheckpoint> borrow; borrow.push_back(img.checkpoints[0]);
        check(conversation_checkpoint_restore(borrow[0], S.ss, S.g, err) && S.device_is(11, 0, 2, {}), "borrow restore = R1");
        S.fill(33); check(conversation_checkpoint_save(slot[0], S.ss, S.g, err) && ckpt_is(S, slot[0], 33), "rewrite the slot copy = R3");
        check(ckpt_is(S, borrow[0], 11) && ckpt_is(S, img.checkpoints[0], 11), "the borrow and the parked copy keep R1");
        borrow.clear(); check(ckpt_is(S, img.checkpoints[0], 11), "dropping the borrow keeps the parked copy");
        // superseded drop / eviction of one of two holders
        {
            ConversationCache cache(16ull << 30, 4);
            SavedConversation a = img, b = img;
            check(cache.put(std::move(a)), "park holder 1");
            check(cache.evict_oldest(), "evict holder 1");
            check(ckpt_is(S, b.checkpoints[0], 11) && ckpt_is(S, img.checkpoints[0], 11) && kv_is(S, b, 11, T), "the other holders keep R1");
        }
        // session file write and read back
        {
            namespace fs = std::filesystem;
            const fs::path p = fs::temp_directory_path() / ("strata-share-gpu-" + std::to_string(dev) + "-" + std::to_string(T) + ".bin");
            const SessionFileIdentity id{1, 2}; size_t n = 0, m = 0; SavedConversation back;
            check(session_file_write(p.string(), img, id, n, err) && session_file_read(p.string(), id, back, m, err) && n == m, "session file round trip");
            check(back.checkpoints.size() == 1 && ckpt_is(S, back.checkpoints[0], 11) && ckpt_is(S, back.live, 11) && kv_is(S, back, 11, T),
                  "the file holds R1 bytes");
            std::error_code ec; fs::remove(p, ec);
        }
        // STRATA_CKPT_THREADS-style: two threads save their own checkpoints at once from the same device state
        {
            S.fill(44);
            ConversationCheckpoint c1, c2; c1.ids = c2.ids = ckp.ids;
            bool r1 = false, r2 = false; std::string e1, e2;
            std::thread t1([&] { ck(cudaSetDevice(dev)); r1 = conversation_checkpoint_save(c1, S.ss, S.g, e1); });
            std::thread t2([&] { ck(cudaSetDevice(dev)); r2 = conversation_checkpoint_save(c2, S.ss, S.g, e2); });
            t1.join(); t2.join();
            check(r1 && r2 && ckpt_is(S, c1, 44) && ckpt_is(S, c2, 44), "two threads save R4 at once");
        }
        // (j) the pool's reuse: a shared block saved into gets a new one; a unique same-size block is reused in place
        {
            S.fill(55); ConversationCheckpoint pc; pc.ids = ckp.ids;
            check(conversation_checkpoint_save(pc, S.ss, S.g, err), "(j) save R5");
            ConversationCheckpoint parked = copy_checkpoint_beating(pc);
            ConversationCheckpoint taken = std::move(pc);                       // drop, give, take
            S.fill(66); check(conversation_checkpoint_save(taken, S.ss, S.g, err) && ckpt_is(S, taken, 66) && ckpt_is(S, parked, 55),
                              "(j) save R6 into the taken checkpoint: it is R6, the parked copy stays R5");
            parked = ConversationCheckpoint{};
            const uint8_t* before = taken.gdn.data();
            S.fill(77); check(conversation_checkpoint_save(taken, S.ss, S.g, err) && ckpt_is(S, taken, 77) && taken.gdn.data() == before,
                              "(j) a unique same-size block is saved into in place (the pool's reuse)");
        }
        // (l) three saves in a row on one device (two staging areas: the third waits for one)
        {
            S.fill(88); ConversationCheckpoint t3[3];
            bool ok = true; for (auto& c : t3) { c.ids = ckp.ids; ok = ok && conversation_checkpoint_save(c, S.ss, S.g, err); }
            check(ok && ckpt_is(S, t3[0], 88) && ckpt_is(S, t3[1], 88) && ckpt_is(S, t3[2], 88), "(l) three saves in a row");
        }
        std::printf("  %lld tokens: snapshot %zu bytes, checks so far %d, failures %d\n", (long long) T, img.bytes(), checks, fails);
    }
}
void split_merge(int d0, int d1) {   // 2 GPUs: stage parts on each device, split / merge, bytes and sharing kept
    std::string err;
    ck(cudaSetDevice(d0)); Session A(8576); A.fill(55);
    ck(cudaSetDevice(d1)); Session B(8576); B.fill(66);
    std::vector<ConversationCheckpoint> cks(1); cks[0].ids.assign(100, 7);
    ck(cudaSetDevice(d0)); check(conversation_checkpoint_save(cks[0], A.ss, A.g, err), "stage 0 part");
    ConversationCheckpoint part; part.ids = cks[0].ids;
    ck(cudaSetDevice(d1)); check(conversation_checkpoint_save(part, B.ss, B.g, err), "stage 1 part");
    cks[0].stage_parts.push_back(std::move(part));
    {   // (m) two threads on the split: each stage's part saved on its own device at once
        ConversationCheckpoint p0, p1; p0.ids = p1.ids = cks[0].ids; bool r0 = false, r1 = false; std::string e0, e1;
        std::thread t0([&] { ck(cudaSetDevice(d0)); r0 = conversation_checkpoint_save(p0, A.ss, A.g, e0); });
        std::thread t1([&] { ck(cudaSetDevice(d1)); r1 = conversation_checkpoint_save(p1, B.ss, B.g, e1); });
        t0.join(); t1.join();
        check(r0 && r1 && ckpt_is(A, p0, 55) && ckpt_is(B, p1, 66), "(m) two threads on the split save both parts");
    }
    {   // stage-1 draft K/V on the split (Fable, drafter ruling q3): the drafter's stage parks its image with the draft layer,
        // the park shares, another state overwrites the device, the restore puts it back; blocking read-back of the restored
        // cells of every draft part against the seeded reference
        ck(cudaSetDevice(d1));
        const int64_t T = 8512;
        std::vector<int32_t> ids((size_t) T); for (size_t i = 0; i < ids.size(); ++i) ids[i] = int32_t(i % 1000 + 1);
        std::vector<ConversationImageKey> imgs; std::vector<ConversationCheckpoint> none;
        const ConversationView view{ids, imgs, none, false};
        B.fill(77);
        SavedConversation img;
        check(conversation_snapshot_save(img, view, B.ss, B.g, &B.draft, err), "stage-1 park with the draft layer");
        SavedConversation shared = img;   // a second holder (the park shares)
        B.fill(78);
        check(conversation_snapshot_restore(img, B.ss, B.g, &B.draft, err) == ConversationRestore::restored, "stage-1 restore");
        ck(cudaDeviceSynchronize());
        const auto s = qsa_real_shapes();
        const size_t cells = (size_t) (((T + s.page_size - 1) / s.page_size) * s.page_size);
        const size_t nq = B.q.size(), base = 2 + 8 * nq;
        std::vector<size_t> upto(B.parts.size(), 0);
        upto[base] = upto[base + 1] = cells * B.g.n_head_kv * B.g.head_dim;
        upto[base + 2] = upto[base + 3] = cells * B.g.n_head_kv * (B.g.head_dim / 64) * 2;
        check(B.device_is(77, base, 4, upto), "stage-1 draft K/V restored byte for byte (blocking read-back)");
        check(kv_is(B, shared, 77, T), "the second holder's draft K/V is still the seeded bytes");
    }
    std::vector<ConversationCheckpoint> keep{copy_checkpoint_beating(cks[0])};
    auto split = conversation_checkpoints_split(std::move(cks), 1);
    std::vector<ConversationCheckpoint> back;
    check(conversation_checkpoints_merge(std::move(split), back) && back.size() == 1 && back[0].stage_parts.size() == 1, "merge");
    check(ckpt_is(A, back[0], 55) && ckpt_is(B, back[0].stage_parts[0], 66) && back[0].gdn.data() == keep[0].gdn.data(),
          "2-GPU split / merge keep both devices' bytes and the sharing");
}
}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) return 77;
    for (int d = 0; d < n; ++d) one_device(d);
    if (n >= 2) split_merge(0, 1);
    std::printf("conversation_share_gpu_test: %d checks, %d failures on %d device(s)\n", checks, fails, n);
    return fails ? 1 : 0;
}
