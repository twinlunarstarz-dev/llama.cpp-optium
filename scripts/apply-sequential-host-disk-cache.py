#!/usr/bin/env python3
from pathlib import Path
import re


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    s = p.read_text()
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{path}: expected one match, found {n}: {old[:120]!r}")
    p.write_text(s.replace(old, new, 1))


def sub_once(path: str, pattern: str, repl: str) -> None:
    p = Path(path)
    s = p.read_text()
    out, n = re.subn(pattern, repl, s, count=1, flags=re.S)
    if n != 1:
        raise SystemExit(f"{path}: expected one regex match, found {n}: {pattern[:120]!r}")
    p.write_text(out)


# ---------------------------------------------------------------------------
# Disk tier: O_DIRECT first.  If the filesystem/device rejects O_DIRECT, keep a
# read-only shard mmap as the storage source and let the kernel page cache be the
# evictable disk->RAM tier.  mmap does not require the file to fit RAM.  If mmap
# itself is unavailable, use positional pread so concurrent prefetch workers do
# not race through a shared FILE seek position.
# ---------------------------------------------------------------------------
replace_once(
    "src/llama-mmap.cpp",
    """    impl(const char * fname, const char * mode, [[maybe_unused]] const bool use_direct_io = false) : fname(fname) {\n#ifdef __linux__\n        // Try unbuffered I/O for read only\n        if (use_direct_io && std::strcmp(mode, \"rb\") == 0) {\n            if (init_fd()) {\n                return;\n            }\n            LLAMA_LOG_WARN(\"Failed to open file '%s' with error: %s. Falling back to buffered I/O\",\n                           fname, strerror(errno));\n        }\n#endif\n        init_fp(mode);\n    }\n""",
    """    impl(const char * fname, const char * mode, [[maybe_unused]] const bool use_direct_io = false) : fname(fname) {\n#ifdef __linux__\n        direct_io_requested = use_direct_io && std::strcmp(mode, \"rb\") == 0;\n        // Try unbuffered I/O for read only. Filesystems that reject O_DIRECT fall\n        // back to an mmap/pread source below; both remain bounded by the kernel's\n        // page cache and therefore support files much larger than physical RAM.\n        if (direct_io_requested) {\n            if (init_fd()) {\n                return;\n            }\n            LLAMA_LOG_WARN(\"Failed to open file '%s' with O_DIRECT: %s. Falling back to mmap/pread\\n\",\n                           fname, strerror(errno));\n        }\n#endif\n        init_fp(mode);\n#ifdef __linux__\n        if (direct_io_requested) {\n            init_direct_fallback_map();\n        }\n#endif\n    }\n""",
)

replace_once(
    "src/llama-mmap.cpp",
    """    void init_fp(const char * mode) {\n        fp = ggml_fopen(fname.c_str(), mode);\n        if (fp == NULL) {\n            throw std::runtime_error(format(\"failed to open %s: %s\", fname.c_str(), strerror(errno)));\n        }\n        seek(0, SEEK_END);\n        size = tell();\n        seek(0, SEEK_SET);\n    }\n\n    impl(FILE * file) : fname(\"(file*)\"), owns_fp(false) {\n""",
    """    void init_fp(const char * mode) {\n        fp = ggml_fopen(fname.c_str(), mode);\n        if (fp == NULL) {\n            throw std::runtime_error(format(\"failed to open %s: %s\", fname.c_str(), strerror(errno)));\n        }\n        seek(0, SEEK_END);\n        size = tell();\n        seek(0, SEEK_SET);\n    }\n\n#ifdef __linux__\n    void init_direct_fallback_map() {\n#if defined(_POSIX_MAPPED_FILES)\n        if (fp == NULL || size == 0) {\n            return;\n        }\n        const int map_fd = fileno(fp);\n        if (map_fd < 0) {\n            return;\n        }\n        void * mapped = mmap(NULL, size, PROT_READ, MAP_SHARED, map_fd, 0);\n        if (mapped == MAP_FAILED) {\n            LLAMA_LOG_WARN(\"%s: mmap fallback failed for '%s': %s; using pread\\n\",\n                           __func__, fname.c_str(), strerror(errno));\n            return;\n        }\n        direct_fallback_map = mapped;\n        direct_fallback_map_size = size;\n#ifdef MADV_SEQUENTIAL\n        (void) madvise(mapped, size, MADV_SEQUENTIAL);\n#endif\n#ifdef POSIX_FADV_SEQUENTIAL\n        (void) posix_fadvise(map_fd, 0, 0, POSIX_FADV_SEQUENTIAL);\n#endif\n        LLAMA_LOG_INFO(\"%s: sequential mmap fallback enabled for '%s' (%.2f GiB virtual)\\n\",\n                      __func__, fname.c_str(), size / (1024.0 * 1024.0 * 1024.0));\n#else\n        GGML_UNUSED(direct_io_requested);\n#endif\n    }\n#endif\n\n    impl(FILE * file) : fname(\"(file*)\"), owns_fp(false) {\n""",
)

replace_once(
    "src/llama-mmap.cpp",
    """    ~impl() {\n        if (fd != -1) {\n            close(fd);\n        } else if (owns_fp) {\n            std::fclose(fp);\n        }\n    }\n    int fd = -1;\n    std::string fname;\n#endif\n""",
    """    ~impl() {\n#if defined(_POSIX_MAPPED_FILES)\n        if (direct_fallback_map != nullptr) {\n            munmap(direct_fallback_map, direct_fallback_map_size);\n            direct_fallback_map = nullptr;\n            direct_fallback_map_size = 0;\n        }\n#endif\n        if (fd != -1) {\n            close(fd);\n        } else if (owns_fp) {\n            std::fclose(fp);\n        }\n    }\n    int fd = -1;\n    std::string fname;\n    bool direct_io_requested = false;\n    void * direct_fallback_map = nullptr;\n    size_t direct_fallback_map_size = 0;\n#endif\n""",
)

sub_once(
    "src/llama-mmap.cpp",
    r"void llama_file::read_at\(void \* ptr, size_t len, size_t offset\) const \{.*?\n\}\n\nbool llama_file::read_at_padded",
    r'''void llama_file::read_at(void * ptr, size_t len, size_t offset) const {
#ifdef __linux__
    if (len == 0) {
        return;
    }
    if (offset > pimpl->size || len > pimpl->size - offset) {
        throw std::runtime_error("pread range exceeds file size");
    }
    if (pimpl->fd != -1) {
        const size_t alignment = pimpl->alignment;
        const size_t aligned_offset = offset & ~(alignment - 1);
        const size_t prefix = offset - aligned_offset;
        const size_t aligned_size = (prefix + len + alignment - 1) & ~(alignment - 1);
        void * raw = nullptr;
        const int rc = posix_memalign(&raw, alignment, aligned_size);
        if (rc != 0) throw std::runtime_error(format("posix_memalign failed with error %d", rc));
        std::unique_ptr<void, decltype(&free)> buffer(raw, free);
        size_t done = 0;
        while (done < aligned_size) {
            const ssize_t n = pread(pimpl->fd, static_cast<uint8_t *>(raw) + done,
                                    aligned_size - done, aligned_offset + done);
            if (n < 0 && errno == EINTR) continue;
            if (n < 0) throw std::runtime_error(format("direct pread error: %s", strerror(errno)));
            if (n == 0) { memset(static_cast<uint8_t *>(raw) + done, 0, aligned_size - done); break; }
            done += static_cast<size_t>(n);
        }
        memcpy(ptr, static_cast<uint8_t *>(raw) + prefix, len);
        return;
    }
#if defined(_POSIX_MAPPED_FILES)
    if (pimpl->direct_fallback_map != nullptr) {
        GGML_ASSERT(offset <= pimpl->direct_fallback_map_size && len <= pimpl->direct_fallback_map_size - offset);
        memcpy(ptr, static_cast<const uint8_t *>(pimpl->direct_fallback_map) + offset, len);
        return;
    }
#endif
    if (pimpl->fp != nullptr) {
        const int raw_fd = fileno(pimpl->fp);
        if (raw_fd >= 0) {
            size_t done = 0;
            while (done < len) {
                const ssize_t n = pread(raw_fd, static_cast<uint8_t *>(ptr) + done, len - done, offset + done);
                if (n < 0 && errno == EINTR) continue;
                if (n < 0) throw std::runtime_error(format("pread error: %s", strerror(errno)));
                if (n == 0) throw std::runtime_error("unexpectedly reached end of file");
                done += static_cast<size_t>(n);
            }
            return;
        }
    }
#endif
    const size_t saved = tell();
    seek(offset, SEEK_SET);
    const_cast<llama_file *>(this)->read_raw(ptr, len);
    seek(saved, SEEK_SET);
}

bool llama_file::read_at_padded''',
)

# ---------------------------------------------------------------------------
# RAM tier: decode the synthetic logical source in O(1), compute the cache limit
# once, and use frequency + deterministic rank admission rather than LRU scan
# replacement. Dense transformer scans then converge to a stable cached subset;
# sparse/hot weights can still promote when their observed frequency is higher.
# ---------------------------------------------------------------------------
replace_once(
    "src/llama-model.cpp",
    """#include <functional>\n#include <list>\n#include <map>\n""",
    """#include <functional>\n#include <fstream>\n#include <list>\n#include <map>\n""",
)

replace_once(
    "src/llama-model.cpp",
    """    struct direct_io_cache_entry {\n        std::shared_ptr<std::vector<uint8_t>> data;\n        std::list<direct_io_cache_key>::iterator lru;\n    };\n    std::vector<direct_io_region> direct_io_regions;\n    std::vector<std::unique_ptr<llama_file>> direct_io_files;\n    mutable std::mutex direct_io_cache_mutex;\n    mutable std::unordered_map<direct_io_cache_key, direct_io_cache_entry, direct_io_cache_key_hash> direct_io_cache;\n    mutable std::list<direct_io_cache_key> direct_io_cache_lru;\n    mutable size_t direct_io_cache_bytes = 0;\n    size_t direct_io_cache_limit = (size_t) 4 * 1024 * 1024 * 1024;\n};\n\nllama_model::llama_model(const llama_model_params & params) : params(params), pimpl(std::make_unique<impl>()) {\n""",
    """    using direct_io_cache_rank_key = std::pair<uint64_t, size_t>;\n    using direct_io_cache_rank = std::multimap<direct_io_cache_rank_key, direct_io_cache_key>;\n    struct direct_io_cache_entry {\n        std::shared_ptr<std::vector<uint8_t>> data;\n        uint64_t frequency = 0;\n        direct_io_cache_rank::iterator rank;\n    };\n    std::vector<direct_io_region> direct_io_regions;\n    std::vector<std::unique_ptr<llama_file>> direct_io_files;\n    mutable std::mutex direct_io_cache_mutex;\n    mutable std::unordered_map<direct_io_cache_key, direct_io_cache_entry, direct_io_cache_key_hash> direct_io_cache;\n    mutable std::unordered_map<direct_io_cache_key, uint64_t, direct_io_cache_key_hash> direct_io_cache_frequency;\n    mutable direct_io_cache_rank direct_io_cache_ranks;\n    mutable size_t direct_io_cache_bytes = 0;\n    size_t direct_io_cache_limit = 0;\n};\n\nstatic size_t llama_sequential_host_cache_limit() {\n    const char * env = getenv(\"LLAMA_SEQUENTIAL_HOST_CACHE_MB\");\n    if (env != nullptr && env[0] != '\\0') {\n        char * end = nullptr;\n        const unsigned long long mb = strtoull(env, &end, 10);\n        if (end != env) {\n            return mb > SIZE_MAX / (1024ull * 1024ull) ? SIZE_MAX : (size_t) mb * 1024ull * 1024ull;\n        }\n    }\n\n    size_t available = 0;\n#if defined(__linux__)\n    std::ifstream f(\"/proc/meminfo\");\n    std::string key;\n    uint64_t value = 0;\n    std::string unit;\n    while (f >> key >> value >> unit) {\n        if (key == \"MemAvailable:\") {\n            available = value > SIZE_MAX / 1024 ? SIZE_MAX : (size_t) value * 1024;\n            break;\n        }\n    }\n#endif\n    if (available == 0) {\n        return (size_t) 4 * 1024 * 1024 * 1024;\n    }\n    const size_t cap = (size_t) 16 * 1024 * 1024 * 1024;\n    return std::min(cap, available / 4);\n}\n\nllama_model::llama_model(const llama_model_params & params) : params(params), pimpl(std::make_unique<impl>()) {\n    if (params.sequential_load) {\n        pimpl->direct_io_cache_limit = llama_sequential_host_cache_limit();\n    }\n""",
)

replace_once(
    "src/llama-model.cpp",
    """    if (pimpl->sequential_load && ml.use_direct_io) {\n        for (auto & file : ml.files) {\n            pimpl->direct_io_files.emplace_back(std::move(file));\n        }\n    }\n\n    return true;\n}\n""",
    """    if (pimpl->sequential_load && ml.use_direct_io) {\n        for (auto & file : ml.files) {\n            pimpl->direct_io_files.emplace_back(std::move(file));\n        }\n        LLAMA_LOG_INFO(\"%s: sequential host weight cache = %.2f GiB (VRAM > RAM > disk)\\n\",\n            __func__, pimpl->direct_io_cache_limit / (1024.0 * 1024.0 * 1024.0));\n    }\n\n    return true;\n}\n""",
)

sub_once(
    "src/llama-model.cpp",
    r"bool llama_model::read_sequential_weight_padded\(\n        const void \* logical_src, void \* dst, size_t capacity, size_t size, size_t \* data_offset\) const \{.*?\n\}\n\nstd::map<ggml_backend_buffer_type_t, size_t> llama_model::memory_breakdown",
    r'''bool llama_model::read_sequential_weight_padded(
        const void * logical_src, void * dst, size_t capacity, size_t size, size_t * data_offset) const {
    if (dst == nullptr || data_offset == nullptr || capacity < size) return false;
    static_assert(sizeof(uintptr_t) >= 8, "sequential direct-I/O logical addresses require 64-bit uintptr_t");

    const uintptr_t begin = reinterpret_cast<uintptr_t>(logical_src);
    const impl::direct_io_cache_key key{begin, size};
    const size_t tie = impl::direct_io_cache_key_hash{}(key);

    uint64_t frequency = 1;
    std::shared_ptr<std::vector<uint8_t>> cached;
    {
        std::lock_guard<std::mutex> lock(pimpl->direct_io_cache_mutex);
        auto & seen = pimpl->direct_io_cache_frequency[key];
        if (seen != UINT64_MAX) {
            ++seen;
        }
        frequency = seen;

        auto hit = pimpl->direct_io_cache.find(key);
        if (hit != pimpl->direct_io_cache.end()) {
            cached = hit->second.data;
            pimpl->direct_io_cache_ranks.erase(hit->second.rank);
            hit->second.frequency = frequency;
            hit->second.rank = pimpl->direct_io_cache_ranks.emplace(
                impl::direct_io_cache_rank_key{frequency, tie}, key);
        }
    }
    if (cached) {
        memcpy(dst, cached->data(), size);
        *data_offset = 0;
        return true;
    }

    // The synthetic pointer is constructed as ((file_index + 1) << 48) | file_offset.
    // Decode it directly instead of linearly scanning every tensor extent for every
    // 16-256 MiB scheduler chunk. The scheduler only derives these addresses from
    // registered sequential GGUF tensors; file bounds provide the final validation.
    constexpr uintptr_t file_offset_mask = (uintptr_t(1) << 48) - 1;
    const uintptr_t file_tag = begin >> 48;
    if (file_tag == 0) {
        return false;
    }
    const size_t file_idx = (size_t) (file_tag - 1);
    const size_t file_offset = (size_t) (begin & file_offset_mask);
    if (file_idx >= pimpl->direct_io_files.size()) {
        return false;
    }
    llama_file * file = pimpl->direct_io_files[file_idx].get();
    if (file_offset > file->size() || size > file->size() - file_offset) {
        return false;
    }
    if (!file->read_at_padded(dst, capacity, size, file_offset, data_offset)) {
        return false;
    }

    const size_t limit = pimpl->direct_io_cache_limit;
    if (limit == 0 || size > limit) {
        return true;
    }

    bool maybe_admit = false;
    {
        std::lock_guard<std::mutex> lock(pimpl->direct_io_cache_mutex);
        if (pimpl->direct_io_cache.find(key) != pimpl->direct_io_cache.end()) {
            return true;
        }
        const impl::direct_io_cache_rank_key candidate{frequency, tie};
        maybe_admit = pimpl->direct_io_cache_bytes <= limit - size;
        if (!maybe_admit && !pimpl->direct_io_cache_ranks.empty()) {
            // Only displace a lower-ranked resident. Equal-frequency cyclic scans
            // converge to a deterministic subset instead of churning every layer.
            maybe_admit = candidate > pimpl->direct_io_cache_ranks.begin()->first;
        }
    }
    if (!maybe_admit) {
        return true;
    }

    auto data = std::make_shared<std::vector<uint8_t>>(size);
    memcpy(data->data(), static_cast<uint8_t *>(dst) + *data_offset, size);

    {
        std::lock_guard<std::mutex> lock(pimpl->direct_io_cache_mutex);
        if (pimpl->direct_io_cache.find(key) != pimpl->direct_io_cache.end()) {
            return true;
        }
        const impl::direct_io_cache_rank_key candidate{frequency, tie};
        while (pimpl->direct_io_cache_bytes > limit - size) {
            if (pimpl->direct_io_cache_ranks.empty() || candidate <= pimpl->direct_io_cache_ranks.begin()->first) {
                return true;
            }
            auto rank_it = pimpl->direct_io_cache_ranks.begin();
            const auto victim_key = rank_it->second;
            auto victim = pimpl->direct_io_cache.find(victim_key);
            GGML_ASSERT(victim != pimpl->direct_io_cache.end());
            pimpl->direct_io_cache_bytes -= victim->second.data->size();
            pimpl->direct_io_cache_ranks.erase(rank_it);
            pimpl->direct_io_cache.erase(victim);
        }

        auto rank_it = pimpl->direct_io_cache_ranks.emplace(candidate, key);
        pimpl->direct_io_cache.emplace(key, impl::direct_io_cache_entry{data, frequency, rank_it});
        pimpl->direct_io_cache_bytes += size;
    }
    return true;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_model::memory_breakdown''',
)

print("sequential host/disk tier optimization applied")
