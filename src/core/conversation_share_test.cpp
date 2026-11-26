// crashfix-20261117 variants S, N, L (review-fable-crash.md "Ruling on S+N"), host only against the fake runtime.
// (a) copy-on-write: a parked copy shares the blocks; rewriting the source (save into it, or a write through it) leaves the
//     parked bytes unchanged; (b) dropping / evicting the parked image leaves the live checkpoint intact, and the reverse;
// (c) the checkpoints moved back on a restore carry the original bytes; (e) N: resize() still zero-fills on reused memory,
//     resize_for_overwrite() leaves the bytes to the caller, on the new-segment and on the extend path; (f) bytes() and the
//     put() charge are the same with shared and with deep-copied checkpoints; (g) a slot copy / borrow shares, and rewriting
//     or dropping either side keeps the other's bytes; (i) capacity == size after a park and after assign_for_overwrite, and
//     split / merge (2 GPUs) keep bytes and sharing.  (h) a failed live save clearing its blocks is in conversation_state.cpp
//     (failed()); it needs the device path, so it is covered by review and the GPU loads, not here.
// Build: c++ -std=c++20 -DSTRATA_USE_HIP=1 -I src/core/test_stubs/bounce -I include src/core/conversation_share_test.cpp
#include "conversation_copy.hpp"
#include "strata/core/conversation_cache.hpp"

#include <cstdio>
#include <cstring>

using namespace strata::core;
using strata::core::conversation_detail::copy_checkpoint_beating;
namespace {
int checks = 0, failures = 0;
void check(bool ok, const char* what) { ++checks; if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); } }
ConversationCheckpoint ckpt(std::vector<int32_t> ids, uint8_t salt, size_t gdn = 1 << 20) {
    ConversationCheckpoint c;
    c.ids = std::move(ids);
    c.gdn.assign_for_overwrite(gdn);
    for (size_t i = 0; i < gdn; ++i) c.gdn.mutable_data()[i] = uint8_t(salt + i);
    c.ple.assign(4096, uint8_t(salt + 1)); c.tails.assign(777, uint8_t(salt + 2));
    c.dead.assign(512, uint8_t(salt + 3)); c.block_pos.assign(4, uint8_t(salt + 4));
    return c;
}
bool same_bytes(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails && a.dead == b.dead && a.block_pos == b.block_pos;
}
ConversationCheckpoint deep(const ConversationCheckpoint& c) {   // the pre-S copy: every block private
    ConversationCheckpoint o = c;
    for (CheckpointBlock* b : {&o.gdn, &o.ple, &o.tails, &o.dead, &o.block_pos}) {
        const CheckpointBlock src = *b; b->assign_for_overwrite(src.size());
        if (src.size()) std::memcpy(b->mutable_data(), src.data(), src.size());
    }
    return o;
}
SavedConversation image(const std::vector<int32_t>& ids, const std::vector<ConversationCheckpoint>& cks) {
    SavedConversation s;
    s.live.ids = ids; s.live.gdn.assign(64, 7);
    for (const auto& c : cks) s.checkpoints.push_back(copy_checkpoint_beating(c));
    s.kv.resize(1); s.kv[0].k.resize(1000);
    return s;
}
}  // namespace

int main() {
    {   // (a)
        ConversationCheckpoint live = ckpt({1, 2, 3}, 10);
        const ConversationCheckpoint before = deep(live);
        ConversationCheckpoint parked = copy_checkpoint_beating(live);
        check(parked.gdn.data() == live.gdn.data() && parked.gdn.shared(), "(a) a park shares the running-state block");
        live.gdn.assign_for_overwrite(live.gdn.size());            // the next save into the same checkpoint (L)
        std::memset(live.gdn.mutable_data(), 0x55, live.gdn.size());
        check(same_bytes(parked, before) && !parked.gdn.shared(), "(a) saving into the source leaves the parked bytes");
        ConversationCheckpoint p2 = copy_checkpoint_beating(parked);
        p2.tails[3] = 0x99;                                         // a write through one holder
        check(parked.tails == before.tails && p2.tails[3] == 0x99, "(a) a write through one holder is private");
        check(live.gdn.unique() && live.gdn.capacity() == live.gdn.size(), "(i) assign_for_overwrite: unique, capacity == size");
    }
    {   // (b), (c), (f)
        std::vector<ConversationCheckpoint> checks{ckpt({1}, 1), ckpt({1, 2}, 2)};
        const std::vector<ConversationCheckpoint> orig{deep(checks[0]), deep(checks[1])};
        ConversationCache cache(1u << 30, 4);
        SavedConversation img = image({1, 2, 3}, checks);
        SavedConversation img_deep = img;
        for (auto& c : img_deep.checkpoints) c = deep(c);
        check(img.bytes() == img_deep.bytes(), "(f) bytes(): shared == deep-copied");
        for (const auto& c : img.checkpoints) check(c.gdn.capacity() == c.gdn.size(), "(i) capacity == size after a park");
        const size_t charge = img.bytes();
        check(cache.put(std::move(img)) && cache.bytes() == charge, "(f) put() charges the shared image as base");
        check(cache.evict_oldest(), "(b) evict");
        check(same_bytes(checks[0], orig[0]) && same_bytes(checks[1], orig[1]), "(b) evicting the parked copy keeps the live bytes");
        SavedConversation img2 = image({1, 2, 3}, checks);
        checks.clear();                                             // the live conversation moves on
        check(same_bytes(img2.checkpoints[0], orig[0]) && same_bytes(img2.checkpoints[1], orig[1]), "(b) the reverse");
        std::vector<ConversationCheckpoint> restored = std::move(img2.checkpoints);   // (c) restore moves them back
        check(same_bytes(restored[0], orig[0]) && same_bytes(restored[1], orig[1]), "(c) restored bytes equal");
    }
    {   // (g) slot copy / borrow
        std::vector<ConversationCheckpoint> a{ckpt({5, 6}, 50)};
        const ConversationCheckpoint orig = deep(a[0]);
        std::vector<ConversationCheckpoint> b; b.push_back(a[0]);   // generate.cpp checks.push_back(*slot_ck) / (*best)
        check(b[0].gdn.data() == a[0].gdn.data(), "(g) a slot copy shares");
        b[0].gdn.assign_for_overwrite(b[0].gdn.size()); std::memset(b[0].gdn.mutable_data(), 1, b[0].gdn.size());
        check(same_bytes(a[0], orig), "(g) rewriting the copy keeps the source");
        std::vector<ConversationCheckpoint> c; c.push_back(a[0]); a.clear();
        check(same_bytes(c[0], orig), "(g) dropping the source keeps the copy");
    }
    {   // (i) split / merge on 2 GPUs keep bytes and sharing
        std::vector<ConversationCheckpoint> cks{ckpt({1}, 3), ckpt({1, 2}, 4)};
        for (auto& c : cks) { c.stage_parts.push_back(ckpt(c.ids, uint8_t(c.ids.size() + 40))); }
        std::vector<ConversationCheckpoint> keep; for (auto& c : cks) keep.push_back(copy_checkpoint_beating(c));
        auto split = conversation_checkpoints_split(std::move(cks), 1);
        std::vector<ConversationCheckpoint> back;
        check(conversation_checkpoints_merge(std::move(split), back) && back.size() == 2, "(i) merge");
        bool ok = true;
        for (size_t i = 0; i < 2; ++i)
            ok = ok && same_bytes(back[i], keep[i]) && back[i].stage_parts.size() == 1 &&
                 same_bytes(back[i].stage_parts[0], keep[i].stage_parts[0]) && back[i].gdn.data() == keep[i].gdn.data();
        check(ok, "(i) split / merge keep bytes and sharing");
    }
    {   // (e) N
        const size_t n = (3u << 24) + 12345;   // 3 full segments and a tail
        { ConversationBuffer poison; poison.resize_for_overwrite(n);
          poison.visit(0, n, [](uint8_t* p, size_t c, size_t) { std::memset(p, 0xA5, c); return true; }); }
        ConversationBuffer z; z.resize(n);
        bool zero = true; z.visit(0, n, [&](const uint8_t* p, size_t c, size_t) { for (size_t i = 0; i < c; ++i) zero = zero && !p[i]; return true; });
        check(zero, "(e) resize() still zero-fills on reused memory");
        ConversationBuffer w; w.resize_for_overwrite(n);
        w.visit(0, n, [](uint8_t* p, size_t c, size_t at) { for (size_t i = 0; i < c; ++i) p[i] = uint8_t(at + i); return true; });
        bool mine = true; w.visit(0, n, [&](const uint8_t* p, size_t c, size_t at) { for (size_t i = 0; i < c; ++i) mine = mine && p[i] == uint8_t(at + i); return true; });
        check(mine && w.size() == n, "(e) resize_for_overwrite: the caller's bytes, whole");
        ConversationBuffer g; g.resize(100, 3); g.resize(200);      // the extend path with a value: zero
        bool ext = true; g.visit(0, 200, [&](const uint8_t* p, size_t c, size_t at) { for (size_t i = 0; i < c; ++i) ext = ext && p[i] == (at + i < 100 ? 3 : 0); return true; });
        check(ext, "(e) resize() on the extend path fills");
        ConversationBuffer h; h.resize(100, 3); h.resize_for_overwrite(200);
        h.visit(100, 100, [](uint8_t* p, size_t c, size_t) { std::memset(p, 9, c); return true; });
        bool ext2 = true; h.visit(0, 200, [&](const uint8_t* p, size_t c, size_t at) { for (size_t i = 0; i < c; ++i) ext2 = ext2 && p[i] == (at + i < 100 ? 3 : 9); return true; });
        check(ext2 && h.bytes() == g.bytes(), "(e) resize_for_overwrite on the extend path: kept bytes, same capacity as resize");
    }
    std::printf("conversation_share_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
