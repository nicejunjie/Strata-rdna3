// include/strata/core/arena_shm.hpp - [claude] STRATA_ARENA_SHM: the expert arena kept in shared memory across engine starts.
//
// A benchmark run restarts the engine, and ~25 s of every start is reading the same 47 GiB of experts from the SSD into an
// anonymous arena that dies with the process.  STRATA_ARENA_SHM=<name> backs the arena with a shared-memory object instead
// (a /dev/shm file, or a memfd held by a `sleep infinity` keeper when /dev/shm cannot hold it) that outlives the engine; the
// next start whose key matches maps it AS its arena (MAP_SHARED: no copy, RAM is not counted twice) and skips the read.
//
// The object's name carries a 64-bit key of everything the arena's bytes depend on (the pack's metadata and sampled content,
// every source file's path, size, mtime and inode, the arena geometry and a layout version), and its 2 MiB header repeats the
// key, the size and a "complete" state that is written only after a full, successful load.  Any mismatch - another pack or
// quant, a file touched since, a load that died half way - is never mapped: the start loads normally (and an unused stale
// object of the same <name> is deleted, so its RAM comes back).  The caller also compares a few sampled experts against the
// source files after attaching (ArenaExpertSource).
//
// Locking (flock on the object): a creator holds LOCK_EX while it fills the arena, every engine using it holds LOCK_SH for
// its lifetime, and only an object nobody holds can be deleted or rebuilt.  Linux only; elsewhere open() says no.
#pragma once

#include <cstdint>
#include <string>
#if defined(_WIN32)
// The implementation is Linux only (elsewhere open() says so and nothing reads
// keeper_); Windows only needs pid_t to exist for this header to compile.
typedef int pid_t;
#else
#include <sys/types.h>
#endif

namespace strata::core {

class ArenaShm {
public:
    ArenaShm() = default;
    ~ArenaShm();
    ArenaShm(const ArenaShm&) = delete;
    ArenaShm& operator=(const ArenaShm&) = delete;

    /// STRATA_ARENA_SHM (empty: off).  A name of [A-Za-z0-9._-], at most 64 characters.
    static std::string env_name();

    /// Attach to the object for (name, key) when it is complete and matches, else create it (empty, zero-filled,
    /// LOCK_EX held) for the caller to fill.  `bytes`: the arena's size; `desc`: a line for humans (the header keeps it).
    /// False: no shared backing for this start (`note` says why); the caller loads a private arena as before.
    bool open(const std::string& name, uint64_t bytes, uint64_t key, const std::string& desc, std::string& note);

    bool reused() const { return reused_; }      ///< attached to a complete object: the arena is already filled
    uint8_t* data() const { return data_; }      ///< the arena (2 MiB aligned)
    void* mapping() const { return map_; }       ///< the whole mapping (header included), for munmap
    uint64_t mapping_bytes() const { return map_bytes_; }
    /// The PinnedArena that adopted the mapping unmaps it; this object then only keeps the lock.
    void hand_over_mapping() { owns_map_ = false; }
    /// After a fresh load: state "complete", the lock downgraded to shared.  False (and `note`) if the header write failed.
    bool mark_complete(std::string& note);
    /// A reused object failed the sampled check: delete it when no other engine holds it (the next start rebuilds it).
    /// True when deleted.
    bool drop_if_unused();
    const std::string& where() const { return where_; }   ///< the file, or "memfd held by pid N", for the startup print

private:
    void discard_created();   // a created object that never became complete: deleted
    int fd_ = -1;
    void* map_ = nullptr;
    uint64_t map_bytes_ = 0;
    uint8_t* data_ = nullptr;
    bool owns_map_ = true;
    bool reused_ = false;
    bool created_ = false;
    bool complete_ = false;
    std::string path_;        // the /dev/shm file, or the keeper's index file
    bool memfd_ = false;
    pid_t keeper_ = 0;
    std::string where_;
};

}  // namespace strata::core
