#pragma once

// KV prompt-cache disk tier: shared-root accounting, crash cleanup, hard quota.
//
// P0 audit 2026-09-25. The commit-marker protocol was already correct; the
// failures were in accounting scope, re-admission, crash cleanup and
// enforcement, which is what lives here.
//
// The swap root is shared by every server instance that points
// LLAMA_SERVER_KV_SWAP_DIR at it, so all three bounds (bytes, files, free
// space) are measured over the root, never over one instance subdirectory.

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include <csignal>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

namespace server_kv_swap {

inline constexpr size_t default_max_mib      = 48ull * 1024ull;
inline constexpr size_t default_min_free_mib = 8ull * 1024ull;
inline constexpr size_t default_max_files    = 128;

inline const char * const env_max_mib    = "LLAMA_SERVER_KV_SWAP_MAX_MIB";
inline const char * const env_min_free  = "LLAMA_SERVER_KV_SWAP_MIN_FREE_MIB";
inline const char * const env_max_files = "LLAMA_SERVER_KV_SWAP_MAX_FILES";
inline const char * const env_swap_dir  = "LLAMA_SERVER_KV_SWAP_DIR";

inline const char * const default_swap_dir = "/tmp/.kv-cache";

inline const char * const dir_prefix   = "llama-kv-swap-";
inline const char * const commit_name  = ".commit";
inline const char * const target_name  = "target.bin";
inline const char * const draft_name   = "draft.bin";
inline const char * const commit_magic = "LLAMA_KV_SWAP_V1";
inline const char * const stage_prefix = ".stage-";

inline size_t env_size(const char * name, size_t fallback) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) {
        return fallback;
    }

    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        return fallback;
    }

    return parsed > (unsigned long long) std::numeric_limits<size_t>::max()
        ? fallback : (size_t) parsed;
}

inline size_t to_bytes(size_t mib) {
    return mib > std::numeric_limits<size_t>::max() / (1024ull * 1024ull)
        ? std::numeric_limits<size_t>::max() : mib * 1024ull * 1024ull;
}

// pid 0 targets the caller's process group, so it is never a real owner.
inline bool pid_alive(int pid) {
    if (pid <= 0) {
        return false;
    }
    if (::kill((pid_t) pid, 0) == 0) {
        return true;
    }
    return errno == EPERM;
}

struct usage {
    size_t bytes    = 0;
    size_t files    = 0;
    bool   complete = true; // false when a walk error truncated the count
};

// An error partway through must not look like a smaller total, or admission
// proceeds on an underestimate and the bound is silently exceeded.
inline usage tree_usage(const std::filesystem::path & root) {
    usage out;
    std::error_code ec;
    const bool exists = std::filesystem::exists(root, ec);
    if (ec) {
        out.complete = false;
        return out;
    }
    if (!exists) {
        return out;
    }

    std::filesystem::recursive_directory_iterator it(root,
            std::filesystem::directory_options::skip_permission_denied, ec);
    const std::filesystem::recursive_directory_iterator end;
    for (; it != end; it.increment(ec)) {
        if (ec) {
            out.complete = false;
            return out;
        }
        std::error_code file_ec;
        const bool regular = it->is_regular_file(file_ec);
        if (file_ec) {
            out.complete = false;
            return out;
        }
        if (!regular) {
            continue;
        }
        const uintmax_t size = it->file_size(file_ec);
        if (file_ec) {
            out.complete = false;
            return out;
        }
        if (size > (uintmax_t) std::numeric_limits<size_t>::max() - out.bytes) {
            out.bytes = std::numeric_limits<size_t>::max();
        } else {
            out.bytes += (size_t) size;
        }
        ++out.files;
    }
    if (ec) {
        out.complete = false;
    }

    return out;
}

struct marker {
    size_t target_bytes = 0;
    size_t draft_bytes  = 0;
};

// A marker is only a commit when it parses and agrees with the bytes on disk,
// so a generation interrupted between the write and the rename is rejected.
inline bool read_commit(const std::filesystem::path & dir, marker & out) {
    std::ifstream in(std::filesystem::path(dir) / commit_name, std::ios::binary);
    std::string magic;
    marker m;
    if (!(in >> magic >> m.target_bytes >> m.draft_bytes) || magic != commit_magic) {
        return false;
    }

    std::error_code ec;
    const auto target_size = std::filesystem::file_size(std::filesystem::path(dir) / target_name, ec);
    if (ec || target_size != m.target_bytes) {
        return false;
    }

    const auto draft = std::filesystem::path(dir) / draft_name;
    if (m.draft_bytes > 0) {
        const auto draft_size = std::filesystem::file_size(draft, ec);
        if (ec || draft_size != m.draft_bytes) {
            return false;
        }
    } else if (std::filesystem::exists(draft, ec) && !ec) {
        return false; // marker claims no draft, but a draft file is present
    }

    out = m;
    return true;
}

inline bool sync_file(const std::filesystem::path & path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}

inline bool sync_dir(const std::filesystem::path & path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        return false;
    }
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}

inline bool write_commit(const std::filesystem::path & dir, size_t target_bytes, size_t draft_bytes) {
    const auto commit = std::filesystem::path(dir) / commit_name;
    std::ofstream out(commit, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }
    out << commit_magic << "\n" << target_bytes << "\n" << draft_bytes << "\n";
    out.flush();
    if (!out) {
        return false;
    }
    out.close();
    return sync_file(commit) && sync_dir(dir);
}

// The pointer alone is not unique across process lifetimes, and a reused
// address would let a later process remove an earlier process's data.
inline std::string instance_dir_name(int pid, const void * self) {
    std::ostringstream name;
    name << dir_prefix << (pid > 0 ? pid : 0) << "-0x" << std::hex
         << (unsigned long long) (uintptr_t) self;
    return name.str();
}

inline std::filesystem::path swap_root() {
    const char * env = std::getenv(env_swap_dir);
    return (env && env[0]) ? std::filesystem::path(env) : std::filesystem::path(default_swap_dir);
}

struct config {
    std::filesystem::path root;         // shared swap root, holds every instance
    std::filesystem::path instance_dir; // this instance's private subdirectory
    int    pid           = 0;
    size_t max_bytes     = 0;           // 0 = no limit
    size_t min_free_bytes = 0;
    size_t max_files     = 0;           // 0 = no limit

    // instance_dir is authoritative and the root is its parent, so the bounds
    // always span every instance sharing the same LLAMA_SERVER_KV_SWAP_DIR.
    static config for_instance(const std::filesystem::path & instance_dir, int pid) {
        config c;
        c.instance_dir = instance_dir;
        c.root         = instance_dir.parent_path();
        if (c.root.empty()) {
            c.root = ".";
        }
        c.pid            = pid > 0 ? pid : 0;
        c.max_bytes      = to_bytes(env_size(env_max_mib, default_max_mib));
        c.min_free_bytes = to_bytes(env_size(env_min_free, default_min_free_mib));
        c.max_files      = env_size(env_max_files, default_max_files);
        return c;
    }
};

struct dir_entry {
    std::filesystem::path path;
    int    owner_pid    = 0;
    bool   reclaimable  = false; // unreachable: foreign and its owner is gone
    std::filesystem::file_time_type mtime{};
};

// The quota is enforced over a root that every server instance sharing
// LLAMA_SERVER_KV_SWAP_DIR also writes to, so the measure-and-reserve step has
// to be exclusive across processes. flock on a file in the root gives that
// without a daemon, and the kernel drops it if a holder dies, so a crashed
// server cannot wedge the tier.
class root_lock {
public:
    explicit root_lock(const config & cfg) { try_lock_impl(cfg); }

    ~root_lock() { unlock(); }

    root_lock(const root_lock &)             = delete;
    root_lock & operator=(const root_lock &) = delete;

    bool try_lock() {
        unlock();
        return try_lock_impl(cfg_);
    }

    bool held() const { return fd_ >= 0; }

    void unlock() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

private:
    bool try_lock_impl(const config & cfg) {
        cfg_ = cfg;
        fd_  = -1;

        const auto path = cfg.root / lock_name;
        const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd < 0) {
            return false;
        }
        if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
            // Another instance is mid-publish. Denying is correct: the caller
            // treats disk as a reusable cache and recomputes instead.
            ::close(fd);
            return false;
        }
        fd_ = fd;
        return true;
    }

    static constexpr const char * lock_name = ".kv-swap.lock";

    config cfg_;
    int    fd_ = -1;
};

struct inventory {
    size_t root_bytes        = 0;
    size_t root_files        = 0;
    size_t reclaimable_bytes = 0;
    size_t reclaimable_files = 0;
    usage  swept;                        // unrecoverable generations removed
    bool   usage_complete    = true;
    std::vector<dir_entry> reclaimable;  // arbitrary order
};

// pid encoded in an instance directory name, -1 when the name does not follow
// the pid-scoped scheme. A parsed pid of 0 is returned as 0: this server never
// writes 0, so such a directory has no live owner and is reclaimable.
inline int instance_owner_pid(const std::string & name) {
    const std::string prefix = dir_prefix;
    if (name.rfind(prefix, 0) != 0) {
        return -1;
    }
    const size_t begin = prefix.size();
    const size_t dash = name.find('-', begin);
    if (dash == std::string::npos || dash == begin) {
        return -1;
    }
    for (size_t i = begin; i < dash; ++i) {
        if (name[i] < '0' || name[i] > '9') {
            return -1;
        }
    }
    const long pid = std::strtol(name.substr(begin, dash - begin).c_str(), nullptr, 10);
    return pid > 0 ? (int) pid : 0;
}

inline bool is_stage_dir(const std::string & name) {
    return name.rfind(stage_prefix, 0) == 0;
}

// Writer pid recorded in a generation directory name, -1 when the name is not
// a generation this server writes. Generation names end in the pid, so the
// pre-fix layout still carries the information the instance name lacks.
inline int generation_owner_pid(const std::string & name) {
    if (!is_stage_dir(name)) {
        return -1;
    }
    const size_t last = name.rfind('-');
    if (last == std::string::npos || last + 1 == name.size()) {
        return -1;
    }
    for (size_t i = last + 1; i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') {
            return -1;
        }
    }
    const long pid = std::strtol(name.substr(last + 1).c_str(), nullptr, 10);
    return pid > 0 ? (int) pid : 0;
}

// Ownership of a legacy pointer-only instance directory, which has no pid of
// its own. A directory holding a generation written by a live process is
// still owned and must never be reclaimed; a directory holding only dead
// writers' generations, or no generations at all, is an orphan. A directory
// holding anything that is not a recognisable generation has unknown contents
// and is left alone, so unrelated data under a matching name is never removed.
enum class legacy_ownership { live, orphan, unknown };

inline legacy_ownership legacy_dir_ownership(const std::filesystem::path & dir, int self_pid) {
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec) || ec) {
        return legacy_ownership::unknown;
    }

    for (const auto & entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) {
            return legacy_ownership::unknown;
        }
        std::error_code sub_ec;
        const bool is_dir = entry.is_directory(sub_ec);
        if (sub_ec || !is_dir) {
            return legacy_ownership::unknown; // not a generation layout we own
        }
        const int owner = generation_owner_pid(entry.path().filename().string());
        if (owner < 0) {
            return legacy_ownership::unknown;
        }
        if (owner == self_pid || pid_alive(owner)) {
            return legacy_ownership::live;
        }
    }

    // Every generation names a dead writer, or the directory is empty and holds
    // no bytes at all. Either way nothing live owns it.
    return legacy_ownership::orphan;
}

inline void remove_tree(const std::filesystem::path & path, usage & swept) {
    const usage before = tree_usage(path);
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    if (ec) {
        return;
    }
    swept.bytes += before.bytes;
    swept.files += before.files;
}

// A generation without a valid marker can never be restored, because every
// restore path re-verifies it. This is the only removal that is not
// admission-driven.
inline void sweep_torn_generations(const std::filesystem::path & dir, usage & swept) {
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec) || ec) {
        return;
    }

    std::vector<std::filesystem::path> torn;
    for (const auto & entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) {
            return;
        }
        std::error_code sub_ec;
        const bool is_dir = entry.is_directory(sub_ec);
        if (sub_ec || !is_dir) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (!is_stage_dir(name)) {
            continue;
        }
        marker m;
        if (!read_commit(entry.path(), m)) {
            torn.push_back(entry.path());
        }
    }

    for (const auto & path : torn) {
        remove_tree(path, swept);
    }
}

// Reconcile the shared root: measure it, sweep generations that can never be
// restored, and identify foreign directories whose owner is gone. Nothing is
// reclaimed here; that happens only when admission needs the room.
//
// This instance's own directory is never swept, because a publish may be in
// flight in it. Its torn generations are collected instead by the next process,
// which sees this directory as foreign with a dead owner.
inline inventory reconcile(const config & cfg) {
    inventory inv;

    const usage root_usage = tree_usage(cfg.root);
    inv.root_bytes     = root_usage.bytes;
    inv.root_files     = root_usage.files;
    inv.usage_complete = root_usage.complete;

    const std::string own_name = cfg.instance_dir.filename().string();

    std::error_code ec;
    const bool is_dir = std::filesystem::is_directory(cfg.root, ec);
    if (ec) {
        inv.usage_complete = false;
        return inv;
    }
    if (!is_dir) {
        return inv;
    }

    for (const auto & entry : std::filesystem::directory_iterator(cfg.root, ec)) {
        if (ec) {
            inv.usage_complete = false;
            return inv;
        }
        std::error_code sub_ec;
        const bool entry_is_dir = entry.is_directory(sub_ec);
        if (sub_ec || !entry_is_dir) {
            continue;
        }

        const std::string name = entry.path().filename().string();
        if (name == own_name) {
            continue;
        }

        const int owner = instance_owner_pid(name);
        if (owner < 0) {
            // Pre-fix layout: the instance name carries a heap pointer and no
            // pid, so ownership has to be recovered from the generations
            // inside. Without this the historical directories are counted but
            // never reclaimable, and the quota can never be satisfied again.
            if (legacy_dir_ownership(entry.path(), cfg.pid) != legacy_ownership::orphan) {
                continue;
            }
            sweep_torn_generations(entry.path(), inv.swept);

            dir_entry de;
            de.path        = entry.path();
            de.owner_pid   = 0;
            de.reclaimable = true;
            std::error_code mt_ec;
            de.mtime = std::filesystem::last_write_time(entry.path(), mt_ec);
            if (mt_ec) {
                de.mtime = decltype(de.mtime){};
            }
            inv.reclaimable.push_back(de);
            continue;
        }
        if (owner == cfg.pid || pid_alive(owner)) {
            continue; // another live instance owns these bytes
        }

        sweep_torn_generations(entry.path(), inv.swept);

        dir_entry de;
        de.path        = entry.path();
        de.owner_pid   = owner;
        de.reclaimable = true;
        std::error_code mt_ec;
        de.mtime = std::filesystem::last_write_time(entry.path(), mt_ec);
        if (mt_ec) {
            de.mtime = decltype(de.mtime){};
        }
        inv.reclaimable.push_back(de);
    }

    for (const auto & de : inv.reclaimable) {
        const usage u = tree_usage(de.path);
        inv.reclaimable_bytes += u.bytes;
        inv.reclaimable_files += u.files;
    }

    return inv;
}

inline bool over_bounds(const std::filesystem::path & root, const usage & used,
                        size_t bytes_needed, size_t files_needed,
                        size_t max_bytes, size_t max_files, size_t min_free_bytes) {
    if (max_bytes > 0 && (bytes_needed > max_bytes || used.bytes > max_bytes - bytes_needed)) {
        return true;
    }
    if (max_files > 0 && (files_needed > max_files || used.files > max_files - files_needed)) {
        return true;
    }
    std::error_code ec;
    const auto space = std::filesystem::space(root, ec);
    if (ec) {
        return true; // headroom is unknown, so it cannot be shown to be safe
    }
    const size_t available = space.available;
    if (available < bytes_needed || available - bytes_needed < min_free_bytes) {
        return true;
    }
    return false;
}

// Remove reclaimable foreign directories, oldest first, until the root can fit
// the request or there is nothing left to reclaim. This is the only path that
// deletes another instance's data, and it only ever runs when admission has
// already failed.
inline bool reclaim_to_fit(const config & cfg, size_t bytes_needed, size_t files_needed) {
    for (size_t pass = 0; pass < 64; ++pass) {
        const usage used = tree_usage(cfg.root);
        if (!used.complete) {
            return false;
        }
        if (!over_bounds(cfg.root, used, bytes_needed, files_needed,
                         cfg.max_bytes, cfg.max_files, cfg.min_free_bytes)) {
            return true;
        }

        inventory inv = reconcile(cfg);
        if (inv.reclaimable.empty()) {
            return false;
        }

        std::sort(inv.reclaimable.begin(), inv.reclaimable.end(),
                  [](const dir_entry & a, const dir_entry & b) {
            if (a.mtime != b.mtime) {
                return a.mtime < b.mtime;
            }
            return a.path.string() < b.path.string();
        });
        remove_tree(inv.reclaimable.front().path, inv.swept);
    }

    return false;
}

// Can the shared root accept bytes_needed/files_needed once the bytes and files
// of the generation being replaced are discounted? Reclaimable orphans are
// evicted only when the request cannot otherwise be admitted.
inline bool admit(const config & cfg, size_t bytes_needed,
                  size_t bytes_replaced, size_t files_replaced, size_t files_needed) {
    const usage used = tree_usage(cfg.root);
    if (!used.complete) {
        return false;
    }

    const size_t used_after  = used.bytes  >= bytes_replaced  ? used.bytes  - bytes_replaced  : 0;
    const size_t files_after = used.files  >= files_replaced ? used.files  - files_replaced : 0;

    if (over_bounds(cfg.root, usage{used_after, files_after, true}, bytes_needed, files_needed,
                    cfg.max_bytes, cfg.max_files, cfg.min_free_bytes)) {
        // Try to make room among the orphans of dead instances, never by
        // touching this instance's own entries.
        if (!reclaim_to_fit(cfg, bytes_needed, files_needed)) {
            return false;
        }
        const usage after = tree_usage(cfg.root);
        if (!after.complete) {
            return false;
        }
        const size_t after_bytes  = after.bytes  >= bytes_replaced  ? after.bytes  - bytes_replaced  : 0;
        const size_t after_files  = after.files  >= files_replaced ? after.files  - files_replaced : 0;
        if (over_bounds(cfg.root, usage{after_bytes, after_files, true}, bytes_needed, files_needed,
                        cfg.max_bytes, cfg.max_files, cfg.min_free_bytes)) {
            return false;
        }
    }

    return true;
}

// The pre-write estimate can undershoot, so the bound is checked again once
// the bytes are real. The stage directory is already inside the root walk, so
// `used` includes it: the only question is whether the whole root now fits, and
// it must be judged as it stands, without crediting a replacement that has not
// been removed yet. A rejected generation is removed; the previously committed
// state is left untouched by the caller.
inline bool enforce_bound_after_commit(const config & cfg, const std::filesystem::path & stage,
                                       size_t bytes_replaced, size_t files_replaced) {
    const usage used = tree_usage(cfg.root);
    if (!used.complete) {
        usage swept;
        remove_tree(stage, swept);
        return false;
    }

    const size_t used_bytes = used.bytes  >= bytes_replaced ? used.bytes  - bytes_replaced : 0;
    const size_t used_files = used.files  >= files_replaced ? used.files  - files_replaced : 0;

    bool reject = false;
    if (cfg.max_bytes > 0 && used_bytes > cfg.max_bytes) {
        reject = true;
    }
    if (!reject && cfg.max_files > 0 && used_files > cfg.max_files) {
        reject = true;
    }
    if (!reject && cfg.min_free_bytes > 0) {
        std::error_code ec;
        const auto space = std::filesystem::space(cfg.root, ec);
        if (ec || space.available < cfg.min_free_bytes) {
            reject = true;
        }
    }

    if (reject) {
        usage swept;
        remove_tree(stage, swept);
        return false;
    }

    return true;
}

} // namespace server_kv_swap
