// crashfix-20261117 variants S, N, L (review-fable-crash.md "Ruling on S+N"):
// - DefaultInitAllocator: a vector whose resize(n) without a value leaves new bytes uninitialised (default-init), for buffers
//   the caller overwrites right after (N: fresh K/V segments; L: the live checkpoint's running state).  resize(n, value) still
//   fills, so every caller that relies on zeroed bytes passes the value explicitly.
// - CheckpointBlock: a checkpoint's running-state bytes (gdn, ple, tails, dead, block_pos) as a shared, copy-on-write block
//   (S).  Copying a checkpoint shares its blocks (refcount) instead of deep-copying ~118 MB at every park; any mutator on a
//   shared block first makes it private, so no holder can ever change another holder's bytes.  The vector-like API keeps the
//   existing callers and tests unchanged.  Threading: a single CheckpointBlock object is never used by two threads at once
//   (the STRATA_CKPT_THREADS parts and the slot checkpoints each own theirs); the refcount itself is atomic.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

namespace strata::core {

template<class T> struct DefaultInitAllocator : std::allocator<T> {
    template<class U> struct rebind { using other = DefaultInitAllocator<U>; };
    DefaultInitAllocator() noexcept = default;
    template<class U> DefaultInitAllocator(const DefaultInitAllocator<U>&) noexcept {}
    template<class U> void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) { ::new ((void*) p) U; }
    template<class U, class... A> void construct(U* p, A&&... a) { ::new ((void*) p) U(std::forward<A>(a)...); }
};
template<class T, class U> bool operator==(const DefaultInitAllocator<T>&, const DefaultInitAllocator<U>&) { return true; }

// O (crashfix-20261117, Fable "Ruling on the suite short-prompt delta"): a block may be PENDING - its bytes are still being
// copied in by the checkpoint helper thread (conversation_copy.hpp).  Every access to the bytes waits for the latch; sharing
// the block (a copy, the park) does not.  The helper holds no reference: it writes through a raw Store pointer, and a Store
// whose last reference goes while it is pending waits in its destructor until the helper has opened the latch (the helper
// notifies while holding the mutex, so the Store outlives the notify).
// P1 (Fable "Ruling on the residual short-prompt cost"): the helper's copy is held until the serve loop releases it (after the last
// late join); a reader that reaches a pending latch first releases every held job (the hook), so no reader can wait on a job that
// nobody will release.
inline std::atomic<void (*)()>& ckpt_release_hook() { static std::atomic<void (*)()> h{nullptr}; return h; }
class CheckpointBlock {
public:
    using Bytes = std::vector<uint8_t, DefaultInitAllocator<uint8_t>>;
    struct Store {
        Bytes bytes;
        std::atomic<bool> ready{true};
        std::mutex mu;
        std::condition_variable cv;
        Store() = default;
        explicit Store(Bytes b) : bytes(std::move(b)) {}
        void wait() {
            if (ready.load(std::memory_order_acquire)) return;
            if (auto f = ckpt_release_hook().load(std::memory_order_acquire)) f();   // a held copy: let it run now
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return ready.load(std::memory_order_acquire); });
        }
        void open() {   // the helper, after its copy: under the mutex, so a waiting destructor cannot free us mid-notify
            std::lock_guard<std::mutex> lk(mu);
            ready.store(true, std::memory_order_release);
            cv.notify_all();
        }
        // always through the mutex: the helper may still be inside open() (it set `ready` but holds `mu` until its notify
        // is done), and a Store must not die before that
        ~Store() {
            if (!ready.load(std::memory_order_acquire))
                if (auto f = ckpt_release_hook().load(std::memory_order_acquire)) f();   // a held copy must not wait for ever
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return ready.load(std::memory_order_acquire); });
        }
    };
    CheckpointBlock() = default;
    CheckpointBlock(const std::vector<uint8_t>& v) { if (!v.empty()) p_ = std::make_shared<Store>(Bytes(v.begin(), v.end())); }   // NOLINT
    CheckpointBlock(std::initializer_list<uint8_t> v) { if (v.size()) p_ = std::make_shared<Store>(Bytes(v.begin(), v.end())); }
    CheckpointBlock(const CheckpointBlock&) = default;              // shares (no wait)
    CheckpointBlock& operator=(const CheckpointBlock&) = default;   // shares (no wait)
    CheckpointBlock(CheckpointBlock&& o) noexcept : p_(std::move(o.p_)) {}
    CheckpointBlock& operator=(CheckpointBlock&& o) noexcept { p_ = std::move(o.p_); return *this; }

    size_t size() const { return p_ ? p_->bytes.size() : 0; }       // known while pending
    bool empty() const { return size() == 0; }
    size_t capacity() const { return p_ ? p_->bytes.capacity() : 0; }
    const uint8_t* data() const { if (!p_) return nullptr; p_->wait(); return p_->bytes.data(); }
    // writes go through mutable_data() (copy-on-write); data() never copies, on a const or non-const block
    uint8_t* mutable_data() { return own() ? p_->bytes.data() : nullptr; }
    const uint8_t* begin() const { return data(); }
    const uint8_t* end() const { return data() + size(); }
    uint8_t operator[](size_t i) const { return data()[i]; }
    uint8_t& operator[](size_t i) { own(); return p_->bytes[i]; }
    bool shared() const { if (p_) p_->wait(); return p_ && p_.use_count() > 1; }
    bool unique() const { if (p_) p_->wait(); return !p_ || p_.use_count() == 1; }
    bool pending() const { return p_ && !p_->ready.load(std::memory_order_acquire); }
    void clear() { p_.reset(); }
    // vector semantics: the bytes kept, new ones zero (or `value`)
    void resize(size_t n, uint8_t value = 0) {
        if (n == size()) { own(); return; }
        if (!p_) p_ = std::make_shared<Store>();
        else own();
        if (n > p_->bytes.capacity()) p_->bytes.reserve(n);
        p_->bytes.resize(n, value);
    }
    void assign(size_t n, uint8_t value) { auto st = std::make_shared<Store>(); st->bytes.reserve(n); st->bytes.resize(n, value); p_ = std::move(st); }
    // L: n bytes the caller overwrites in full right after; uninitialised, unique, capacity == n.  A unique block of exactly
    // n bytes (the STRATA_CKPT_POOL reuse) is kept as it is; a shared one is never copied (its bytes are about to go).
    void assign_for_overwrite(size_t n) {
        if (p_) p_->wait();
        if (p_ && p_.use_count() == 1 && p_->bytes.size() == n && p_->bytes.capacity() == n) return;
        auto st = std::make_shared<Store>();
        st->bytes.reserve(n);
        st->bytes.resize(n);   // default-init: no fill
        p_ = std::move(st);
    }
    // O: like assign_for_overwrite, then marked pending; returns the Store the helper writes (raw: it takes no reference)
    Store* begin_pending(size_t n) {
        assign_for_overwrite(n);
        if (!p_) return nullptr;
        p_->ready.store(false, std::memory_order_release);
        return p_.get();
    }
    bool operator==(const CheckpointBlock& o) const {
        if (size() != o.size()) return false;
        if (size() == 0) return true;
        const uint8_t* a = data(); const uint8_t* b = o.data();
        return a == b || std::memcmp(a, b, size()) == 0;
    }
    bool operator==(const std::vector<uint8_t>& v) const {
        return size() == v.size() && (size() == 0 || std::memcmp(data(), v.data(), size()) == 0);
    }

private:
    bool own() {   // make the block private before a write; false when empty
        if (!p_) return false;
        p_->wait();
        if (p_.use_count() > 1) p_ = std::make_shared<Store>(p_->bytes);
        return true;
    }
    std::shared_ptr<Store> p_;
};

} // namespace strata::core
