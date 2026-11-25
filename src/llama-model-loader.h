#pragma once

#include "llama.h"

#include "llama-impl.h"
#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <chrono>
#include <deque>
#include <set>
#include <condition_variable>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <memory>
#include <condition_variable>
#include <unordered_map>

// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,
};

const char * llama_file_version_name(llama_fver version);

// Forward declarations for MoE tiered placement
struct moe_placement_plan;
struct moe_ram_arena;

struct llama_model_loader {
    // Holds information on a model weight
    struct llama_tensor_weight {
        uint16_t  idx; // source file index
        size_t   offs; // tensor data offset in the original file

        ggml_tensor * tensor;

        llama_tensor_weight(const llama_file * file, uint16_t idx, const struct gguf_context * gguf_ctx, ggml_tensor * tensor) : idx(idx), tensor(tensor) {
            const int tensor_idx = gguf_find_tensor(gguf_ctx,  ggml_get_name(tensor));
            if (tensor_idx < 0) {
                throw std::runtime_error(format("tensor '%s' not found in the model", ggml_get_name(tensor)));
            }

            offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
            if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
                throw std::runtime_error(format("tensor '%s' data is not within the file bounds, model is corrupted or incomplete", ggml_get_name(tensor)));
            }
        }
    };

    // custom comparator to sort weights more nicely by layer
    struct weight_name_comparer {
        bool operator()(const std::string & a, const std::string & b) const {
            int a_layer = -1;
            int b_layer = -1;
            sscanf(a.c_str(), "blk.%d.", &a_layer);
            sscanf(b.c_str(), "blk.%d.", &b_layer);
            if (a_layer != b_layer) {
                return a_layer < b_layer;
            }
            return a < b;
        }
    };

    static const int TENSOR_NOT_REQUIRED    = 1 << 0;
    static const int TENSOR_DUPLICATED      = 1 << 1;
    static const int TENSOR_SKIP            = 1 << 2;
    static const int TENSOR_SKIP_IF_VIRTUAL = 1 << 3;
    static const int TENSOR_ALLOW_RESHAPE   = 1 << 4;
    static const int TENSOR_READ_LAZY       = 1 << 5; // read rows on demand instead of loading whole tensor; requires mmap for now

    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;

    uint64_t n_elements = 0;
    size_t   n_bytes    = 0;

    bool use_mmap = false;
    bool use_direct_io = false;
    bool check_tensors;
    bool no_alloc;
    bool load_mtp = false;
    bool sequential_load = false;

    // handle TENSOR_READ_LAZY
    // use case: keep PLE / engrams embd tensors on disk, read them on demand
    struct lazy_read {
        // set by the caller before the create_tensor() calls
        enum llama_lazy_mode mode = LLAMA_LAZY_MODE_OFF;

        // decide whether this tensor is read lazily
        // pass w to also record it, or nullptr to only ask
        // force: bypass the lazy-mode and size thresholds; used for MoE expert
        // tensors under tiered placement (RAM arena or mmap page cache serves
        // them on demand, so they must never be eagerly loaded into host RAM)
        bool add(const std::string & name, const ggml_tensor * t, const llama_tensor_weight * w, bool force = false);

        bool any() const {
            return !ranges.empty();
        }

        bool has(const ggml_tensor * t) const {
            return tensors.count(ggml_get_name(t)) > 0;
        }

        const llama_mmap::ranges & for_file(uint32_t idx) const {
            static const llama_mmap::ranges none;

            const auto it = ranges.find(idx);
            return it == ranges.end() ? none : it->second;
        }

        // lazy tensors are gathered on the host, so no offload setting applies to them
        static ggml_backend_buffer_type_t buft();

    private:
        std::map<uint32_t, llama_mmap::ranges> ranges;
        std::set<std::string>                  tensors;
    } lazy;

    llama_files files;
    llama_ftype ftype;
    llama_fver  fver;

    llama_mmaps mappings;

    std::map<std::string, llama_tensor_weight, weight_name_comparer> weights_map;
    std::unordered_map<std::string, llama_model_kv_override> kv_overrides;
    const llama_model_tensor_buft_override * tensor_buft_overrides;
    std::vector<llama_model_tensor_buft_override> moe_overrides_owned;
    std::deque<std::string> moe_pattern_strings; // stable refs: overrides store raw c_str()
    std::unique_ptr<moe_placement_plan> moe_plan;
    moe_ram_arena * moe_arena = nullptr; // owned by g_moe_ram_arena

    // Point every RAM-tier expert tensor at the arena block that holds it, so the
    // MoE CPU branch can compute those experts straight out of host RAM instead of
    // streaming them over PCIe. Must run after load_all_data() (arena sizes final).
    void rebind_moe_arena_tensors();
    // Pread (and optionally lock) the arena RAM tier before the first token.
    void preload_moe_arena(size_t n_workers, bool lock_ram);

    gguf_context_ptr metadata_ptr;
    struct gguf_context * metadata; // either metadata_ptr.get() or externally set
    llama_model_set_tensor_data_t set_tensor_data;
    void * set_tensor_data_ud;
    std::vector<ggml_context_ptr> contexts;

    std::string arch_name;
    LLM_KV      llm_kv    = LLM_KV(LLM_ARCH_UNKNOWN);

    size_t size_done = 0;
    size_t size_data = 0;
    std::vector<std::pair<size_t, size_t>> mmaps_used;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    // lazy tensors need dedicated context
    struct ctx_key {
        ggml_backend_buffer_type_t buft;
        bool lazy;
    };

    struct ctx_key_comparator {
        bool operator()(const ctx_key & lhs, const ctx_key & rhs) const {
            if (lhs.lazy != rhs.lazy) {
                return lhs.lazy < rhs.lazy;
            }
            return strcmp(ggml_backend_buft_name(lhs.buft), ggml_backend_buft_name(rhs.buft)) < 0;
        }
    };

    std::map<ctx_key, ggml_context_ptr, ctx_key_comparator> ctx_map;

    // track tensors that had to be moved for debugging:
    size_t n_tensors_moved = 0;
    std::string first_tensor_moved_name;
    std::string first_tensor_moved_type_name;
    ggml_backend_buffer_type_t first_moved_from_buft = nullptr;
    ggml_backend_buffer_type_t first_moved_to_buft = nullptr;

    // Capture file offset for RAM-tier expert tensor
    void capture_expert_file_offset(const char * tensor_name, size_t file_offset, size_t tensor_size, struct ggml_tensor * cur, uint16_t shard_idx = 0);

    llama_model_loader(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & fname,
        std::vector<std::string> & splits, // optional, only need if the split does not follow naming scheme
        FILE * file,
        llama_load_mode load_mode,
        bool check_tensors,
        bool no_alloc,
        bool load_mtp,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p,
        // MoE tiered placement parameters
        size_t moe_ram_mib,
        const char * moe_placement,
        bool moe_pin_decode,
        const float * tensor_split,
        int n_devices,
        const char * moe_hot);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const std::string & key, T & result, bool required = true);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_arr(const std::string & key, std::vector<T> & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_arr(const std::string & key, std::array<T, N_MAX> & result, bool required = true);

    template<typename T>
    bool get_arr(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_key(const std::string & key, T & result, bool required = true);

    template<typename T>
    bool get_key(enum llm_kv kid, T & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, bool required = true);

    template<typename T>
    bool get_key_or_arr(enum llm_kv kid, T & result, uint32_t n, bool required = true);

    bool get_key_or_arr(enum llm_kv kid, uint32_t & result, bool required = true);

    std::string get_arch_name() const;

    enum llm_arch get_arch() const;

    const llama_tensor_weight * get_weight(const char * name) const;

    const llama_tensor_weight & require_weight(const char * name) const;

    struct ggml_tensor * get_tensor_meta(const char * name) const;

    struct ggml_tensor * require_tensor_meta(const std::string & name) const;

    const struct ggml_tensor * check_tensor_dims(
            const std::string & name,
            const std::vector<int64_t> & ne,
            bool required,
            bool allow_reshape) const;

    struct ggml_tensor * create_tensor(
        const llama_hparams & hparams, const buft_list_t * buft_list_cpu, const buft_list_t * buft_list_input, const buft_list_t * buft_list_output,
        const buft_list_t * buft_list_layer, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags);

    void done_getting_tensors(bool partial = false) const;

    void init_mappings(bool prefetch = true, llama_mlocks * mlock_mmaps = nullptr);

    // byte range within a mapped file
    struct mapping_range {
        size_t first; // inclusive
        size_t last;  // exclusive
    };

    // sorted, non-overlapping ranges of file 'idx' that back the tensors of 'ctx'
    // ranges owned by other contexts are excluded, so that a buffer wrapping them won't pin
    // data that this context never reads (lazy tensors, -cmoe tensors, ...)
    std::vector<mapping_range> get_mapping_ranges(uint32_t idx, ggml_context * ctx) const;

    // release a weight's mmap pages
    void unmap_weight(const llama_tensor_weight & w) const;

    // read a byte range of a weight's data
    // with mmap, returns a pointer into the mapping, otherwise reads into buf and returns buf
    const void * load_data_range(const llama_tensor_weight & w, size_t offs, size_t size, void * buf) const;

    // Returns false if cancelled by progress_callback
    bool load_all_data(
            struct ggml_context * ctx,
            std::vector<ggml_backend_buffer_t> & bufs,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data);

    std::string ftype_name() const;

    void print_info() const;
};

// MoE Tiered Placement Budget Solver (Phase 1)
// Expert tier: 0=VRAM, 1=RAM, 2=DISK
enum moe_expert_tier { MOE_TIER_VRAM = 0, MOE_TIER_RAM = 1, MOE_TIER_DISK = 2 };

struct moe_placement_plan {
    // Phase 1: layer-granular budgets
    std::vector<size_t> vram_expert_budget;
    size_t ram_expert_budget = 0;
    int n_layers_vram = 0;
    int n_layers_ram = 0;
    int n_layers_disk = 0;
    size_t expert_bytes_per_layer = 0;
    size_t total_expert_bytes = 0;
    size_t pinned_decode_bytes = 0;
    std::vector<size_t> kv_compute_reserve;
    std::vector<std::string> report_lines;
    bool fits = false;
    std::string error;

    // Phase 1b: precise per-layer / per-device accounting (filled when the
    // solver runs with a model loader, i.e. it can read real tensor sizes).
    // layer_tier[l] = 0 VRAM, 1 RAM, 2 DISK. Placement is non-contiguous
    // because every expert layer must stay on its attention layer's device.
    std::vector<uint8_t> layer_tier;
    std::vector<size_t>  layer_expert_bytes;
    std::vector<size_t>  device_total;      // total VRAM per accelerator
    std::vector<size_t>  device_nonexpert;  // real non-expert weight bytes
    std::vector<size_t>  device_committed;  // kv + compute + draft + mmproj + reserve
    std::vector<size_t>  device_budget;     // leftover for experts
    bool precise = false;

    // Phase 2: expert-granular placement
    // expert_tier[layer * n_expert + expert] = tier (0=VRAM, 1=RAM, 2=DISK)
    std::vector<uint8_t> expert_tier;
    // expert_slot[layer * n_expert + expert] = slot index within tier (for VRAM/RAM buffer indexing)
    std::vector<int32_t> expert_slot;
    // Per-tier expert counts
    std::vector<size_t> tier_expert_counts; // size 3: VRAM, RAM, DISK
    // Per-tier bytes per layer
    std::vector<size_t> tier_bytes_per_layer; // size 3
    // Profile used for ranking (activation counts per expert)
    std::vector<uint32_t> profile_counts; // [n_layer * n_expert]
    int n_expert = 0;
    int n_layer = 0;
};

// Solve MoE expert placement from metadata and params (Phase 1: layer-granular)
moe_placement_plan moe_solve_placement(
        struct gguf_context * metadata,
        const float * tensor_split,
        int n_devices,
        size_t moe_ram_mib,
        const char * /*moe_placement_mode*/,
        bool moe_pin_decode,
        int n_layer,
        int n_expert,
        int n_expert_used,
        llama_model_loader * loader = nullptr);

// Phase 2: expert-granular solver with profile
moe_placement_plan moe_solve_placement_expert_granular(
        struct gguf_context * metadata,
        const float * tensor_split,
        int n_devices,
        size_t moe_ram_mib,
        const char * moe_placement_mode,
        bool moe_pin_decode,
        int n_layer,
        int n_expert,
        int n_expert_used,
        const char * moe_profile_path);

// Note: loader is used to store pattern strings for lifetime management
std::vector<llama_model_tensor_buft_override> moe_plan_to_overrides(
        llama_model_loader * loader,
        const moe_placement_plan & plan,
        int n_layer);

// Convert expert-granular plan to tensor buffer type overrides
std::vector<llama_model_tensor_buft_override> moe_plan_to_overrides_expert_granular(
        llama_model_loader * loader,
        const moe_placement_plan & plan,
        int n_layer,
        int n_expert);

// ============================================================================
// Phase 3: Bounded RAM Arena for MoE experts
// ============================================================================

// Per-expert slot in the RAM arena
struct moe_ram_slot {
    int layer = -1;
    int expert = -1;
    std::string tensor_type;     // "gate", "up", "down", "gate_up"
    size_t file_offset = 0;      // Offset in GGUF file for this expert's tensor data
    size_t size = 0;             // Size of expert tensor data in bytes
    uint16_t shard_idx = 0;      // Source GGUF shard (index into shard_fds)
    // data pointer now points into the combined buffer (set when resident)
    void * data = nullptr;       // Pointer to data in arena (when resident)
    uint64_t last_access = 0;    // For LRU eviction
    bool resident = false;       // Whether data is currently in arena
};

// Combined buffer for a layer's expert tensor (gate, up, down, or gate_up)
// Holds all experts contiguously with the same stride as the original tensor
struct moe_combined_buffer {
    int layer = -1;
    std::string tensor_type;     // "gate", "up", "down", "gate_up"
    std::vector<uint8_t> buffer; // Combined buffer for all experts
    // The expert tensors are handed to ggml as host weights, so the first row has
    // to satisfy GGML's tensor alignment. `offset` aligns buffer.data() without
    // changing the contiguous per-expert layout: expert e lives at base()+e*stride.
    size_t offset = 0;
    size_t alloc = 0;
    uint8_t * base() { return buffer.data() + offset; }
    const uint8_t * base() const { return buffer.data() + offset; }
    size_t expert_size = 0;      // Size of one expert in bytes
    size_t stride = 0;           // Stride between experts (nb[2])
    std::vector<bool> resident;  // Per-expert resident flags
    uint64_t last_access = 0;    // For LRU (whole buffer)
};

// Per-layer expert geometry. Models such as nemotron_h_moe declare a different
// expert_feed_forward_length per block, so the byte size of one expert is a
// property of (layer, tensor type), not a global constant. The arena keeps this
// table so every pread and every upload uses the layer's own stride.
struct moe_layer_geom {
    int    n_expert = 0;             // experts in this layer (0 = not yet captured)
    std::array<size_t, 4> bytes {};  // one expert's bytes per type: gate_up, gate, up, down
    bool   known = false;            // at least one expert tensor of this layer was captured
};

// Arena identity. Expert geometry is part of the key: two models (or a target and
// its MTP/draft head) that share block/expert counts but differ in per-layer
// expert width must never resolve to the same arena.
struct moe_arena_key {
    int      n_layer   = 0;
    int      n_expert  = 0;
    uint64_t geom_hash = 0;
    bool operator<(const moe_arena_key & o) const {
        if (n_layer   != o.n_layer)   return n_layer   < o.n_layer;
        if (n_expert  != o.n_expert)  return n_expert  < o.n_expert;
        return geom_hash < o.geom_hash;
    }
};

// Fingerprint of the per-layer expert geometry. The loader calls this with the
// GGUF arrays and the inference side with llama_hparams (read from the same
// keys), so the two sides agree without sharing objects. Absent arrays hash as
// zeros on both sides, i.e. "uniform geometry".
inline uint64_t moe_arena_geom_hash(int n_layer, int n_expert, const uint32_t * n_ff_exp, const uint32_t * n_expert_used) {
    uint64_t h = 1469598103934665603ull; // FNV-1a 64
    auto mix = [&h](uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            h ^= (v >> (8*i)) & 0xffull;
            h *= 1099511628211ull;
        }
    };
    mix((uint64_t) (uint32_t) n_layer);
    mix((uint64_t) (uint32_t) n_expert);
    for (int i = 0; i < n_layer; ++i) {
        mix(n_ff_exp      != nullptr ? (uint64_t) n_ff_exp[i]      : 0ull);
        mix(n_expert_used != nullptr ? (uint64_t) n_expert_used[i] : 0ull);
    }
    return h;
}

// Bounded RAM arena for MoE experts on disk tier
// Fixed size (--moe-ram-mib), holds expert tensors loaded on-demand via pread
// DISK-tier expert tensor metadata (no arena buffer; served by direct pread)
struct moe_disk_tensor {
    size_t base_offset = 0;      // file offset of expert 0
    size_t stride = 0;           // bytes per expert
    uint16_t shard_idx = 0;      // source GGUF shard
};
struct moe_ram_arena {
    size_t capacity_bytes = 0;           // Total arena capacity (--moe-ram-mib)
    size_t used_bytes = 0;               // Currently used bytes
    std::vector<moe_ram_slot> slots;     // All slots (one per RAM-tier expert)
    std::vector<size_t> lru_list;        // LRU order (indices into slots)
    std::mutex mutex;                    // Thread safety for async prefetch
    std::atomic<bool> shutdown = false;  // Signal worker to stop
    std::thread prefetch_thread;         // Async prefetch worker
    int model_fd = -1;                  // File descriptor for model file (for pread)
    std::vector<int> shard_fds;          // One fd per GGUF shard (index = shard idx)

    // Combined buffers: key = (layer, tensor_type), value = buffer
    // tensor_type: "gate", "up", "down", "gate_up"
    std::map<std::pair<int, std::string>, moe_combined_buffer> combined_buffers;

    // DISK-tier expert tensors: key = (layer, tensor_type)
    std::map<std::pair<int, std::string>, moe_disk_tensor> disk_meta;

    // Per-layer expert geometry (index = block index). Filled at capture time
    // from the real tensors, so each layer is addressed with its own stride.
    std::vector<moe_layer_geom> layer_geom;
    int      n_layer_total = 0;
    uint64_t geom_hash     = 0;
    // Set when a layer's geometry contradicts what the arena recorded. Serving
    // is then refused for the whole arena so every read falls back to the mmap
    // path instead of handing out weights of the wrong shape.
    bool     disabled      = false;

    static int type_index(const char * tensor_type) {
        return tensor_type == nullptr ? -1 :
               strcmp(tensor_type, "gate_up") == 0 ? 0 :
               strcmp(tensor_type, "gate")    == 0 ? 1 :
               strcmp(tensor_type, "up")      == 0 ? 2 :
               strcmp(tensor_type, "down")    == 0 ? 3 : -1;
    }
    // Bytes of one expert tensor of `tensor_type` in `layer`, 0 if unknown.
    size_t layer_expert_bytes(int layer, const char * tensor_type) const {
        const int t = type_index(tensor_type);
        if (t < 0 || layer < 0 || layer >= (int) layer_geom.size()) {
            return 0;
        }
        return layer_geom[layer].bytes[t];
    }
    int layer_n_expert(int layer) const {
        if (layer < 0 || layer >= (int) layer_geom.size()) {
            return 0;
        }
        return layer_geom[layer].n_expert;
    }
    // Record one layer's expert geometry, validating it against what is already
    // known for that layer. Returns false when the layer contradicts itself (the
    // arena is then disabled: no expert is served with a wrong stride).
    bool note_layer_geom(int layer, int n_expert, const char * tensor_type, size_t bytes);
    void disable(const char * reason);

    // host-RAM LRU in front of disk-tier expert preads
    struct disk_lru_ent {
        std::shared_ptr<std::vector<uint8_t>> buf;
        uint64_t last = 0;
        size_t   size = 0;
    };
    std::mutex disk_lru_mu;
    std::unordered_map<uint64_t, disk_lru_ent> disk_lru;
    size_t   disk_lru_bytes = 0;
    size_t   disk_lru_cap = 0;
    uint64_t disk_lru_clock = 0;
    uint64_t disk_lru_hits = 0;
    uint64_t disk_lru_misses = 0;
    std::string warm_profile_path;
    int      warm_n_expert = 0;
    size_t   warm_budget_mib = 8192;
    std::atomic<bool> warm_done{false};

    // routing-lookahead warm state
    int      warm_hist_n = 4;                 // recent routed-id sets kept per layer
    size_t   warm_budget = 96ull << 20;       // rolling-window byte budget
    size_t   warm_win_bytes = 0;              // bytes queued in the current window
    std::chrono::steady_clock::time_point warm_win_t0{};
    std::mutex warm_mu;
    std::map<int, std::deque<std::vector<int32_t>>> warm_hist;
    std::unordered_map<uint64_t, uint64_t> warm_armed; // key -> event armed
    std::atomic<uint64_t> warm_event_a{0};
    std::atomic<uint64_t> warm_pred{0}, warm_useful{0}, warm_labytes{0}, warm_dedup{0}, warm_fskip{0}, warm_late{0};
    uint64_t warm_next_stat = 16384;

    // staged warm ring: workers pread predicted misses into disk_lru so the
    // serve path hits host RAM instead of NVMe (Strata's staged variant)
    struct warm_qjob { int il; int tidx; int e; uint64_t ev; };
    std::mutex warm_q_mu;
    std::condition_variable warm_q_cv;
    std::deque<warm_qjob> warm_q;
    std::set<uint64_t> warm_q_keys;
    std::vector<std::thread> warm_workers;
    std::atomic<bool> warm_stop{false};
    size_t   warm_ring_mib = 1024;
    void warm_stage_worker();
    bool warm_stage(int il, int tidx, int expert);

    static uint64_t warm_type_idx(const char * type) {
        return strcmp(type, "gate_up") == 0 ? 0 : strcmp(type, "gate") == 0 ? 1 :
               strcmp(type, "up") == 0      ? 2 : 3;
    }
    static uint64_t warm_key(int il, uint64_t ti, int e) {
        return ((uint64_t)(uint32_t) il << 40) | (ti << 32) | (uint32_t) e;
    }

    // Initialize arena with given capacity and expert metadata
    bool init(size_t capacity_mib, const moe_placement_plan & plan, int n_layer, int n_expert, const char * model_path);
    bool init_sharded(size_t capacity_mib, const moe_placement_plan & plan, int n_layer, int n_expert, const std::vector<int> & shard_fds_in);

    // Load expert into arena (blocking, for synchronous access)
    // Returns pointer to expert data in combined buffer, or nullptr if not in RAM tier
    void * load_expert(int layer, int expert, const char * tensor_type = nullptr);

    // Ensure all selected experts for a layer and tensor type are loaded
    // Called before moe_cache uploads experts to GPU
    bool ensure_experts_loaded(int layer, const char * tensor_type, const int * expert_ids, int n_experts);

    // Get combined buffer pointer for a layer and tensor type
    // Returns nullptr if not a RAM-tier tensor
    void * get_combined_buffer(int layer, const char * tensor_type);

    // Direct pread of one expert (any tier) into dst; returns bytes read, 0 on miss.
    // Thread-safe for concurrent use: read-only map + positional pread.
    // dst_cap is the caller's buffer size for one expert of THIS layer; the read
    // is refused if the layer's own stride would overflow it.
    size_t read_expert_direct(int layer, const char * tensor_type, int expert, void * dst, size_t dst_cap);
    size_t read_expert_direct(int layer, const char * tensor_type, int expert, void * dst) {
        return read_expert_direct(layer, tensor_type, expert, dst, (size_t) -1);
    }

    // Profile-guided page-cache warm: issue POSIX_FADV_WILLNEED for the hottest
    // disk-tier experts (by placement profile) so cold topic switches hit RAM.
    void warm_disk_from_profile(const char * profile_path, int n_expert_in, size_t budget_mib);
    bool warm_ready() const { return warm_done.load(); }
    void warm_mark_done() { warm_done.store(true); }
    int  n_expert_total() const { return warm_n_expert; }
    void warm_arm(const char * profile_path, int n_expert_in, size_t budget_mib) {
        if (profile_path && *profile_path && n_expert_in > 0) {
            warm_profile_path = profile_path;
            warm_n_expert = n_expert_in;
            warm_budget_mib = budget_mib;
        }
    }

    // Disk-tier expert read with a host-RAM LRU cache in front of the disk.
    // Repeat uses of a disk-tier expert are served from RAM instead of re-preading
    // the GGUF shard on every use. Cap via LLAMA_MOE_DISK_LRU_MIB (default 8192, 0 = off).
    size_t read_expert_cached(int layer, const char * tensor_type, int expert, void * dst, size_t dst_cap);
    size_t read_expert_cached(int layer, const char * tensor_type, int expert, void * dst) {
        return read_expert_cached(layer, tensor_type, expert, dst, (size_t) -1);
    }

    // Prefetch expert asynchronously (non-blocking)
    void prefetch_expert(int layer, int expert, const char * tensor_type = nullptr);

    // ---- routing-lookahead warm (adapted from Strata's RouterLookahead) ----
    // Records the routed expert ids of layer `il` for this event (one call per
    // copy-callback invocation per layer). History is newest-entry-first.
    void warm_note(int il, const int32_t * ids, int n);
    // Warms (POSIX_FADV_WILLNEED) the disk-tier experts of layers il+1..il+ahead
    // that recent history routed, skipping RAM-resident ones, within a rolling
    // byte budget. Prediction only warms pages: it never changes what is computed.
    void warm_lookahead(int il, int ahead);
    // Credits a warmed prediction when the expert is actually served from disk
    bool warm_credit(int il, const char * type, int e);
    // true if expert e of (il, type) currently sits in the arena's RAM tier
    bool expert_resident(int il, const char * type, int e) const;

    // true only when expert e is resident in EVERY one of `types` for layer `il`.
    // The CPU expert branch reads all expert tensors of a layer, so a partially
    // resident expert must not be scheduled there (it would read zero pages).
    bool expert_resident_all(int layer, int expert, const std::vector<std::string> & types) const;

    // Pread the RAM tier in full, so every arena-backed expert is present before
    // the first token and the CPU branch never depends on a fill side effect.
    // Also mlock()s the populated blocks when `lock_ram` is set (best effort).
    bool preload_all(size_t n_workers, bool lock_ram);

    // Evict least recently used expert(s) to make room for new data
    void evict_lru(size_t need_bytes);

    // Shutdown prefetch worker
    void shutdown_worker();

    // Destructor
    ~moe_ram_arena();
};

// Global arena instance (owned by llama_model_loader)
extern moe_ram_arena * g_moe_ram_arena;
// Keyed by (blocks, experts, per-layer geometry fingerprint): heterogeneous
// expert widths get their own arena and can never alias a uniform one.
extern std::map<moe_arena_key, std::unique_ptr<moe_ram_arena>> g_moe_arenas;

// Early MoE solver - called from common.cpp before standard parameter fitting
// Returns true on success, false on failure (error in plan.error)
bool llama_moe_solve_early(
        const char * model_path,
        const float * tensor_split,
        int n_devices,
        size_t moe_ram_mib,
        const char * moe_placement_mode,
        bool moe_pin_decode,
        const char * moe_profile_path,
        int & n_layer_out,
        int & n_expert_out,
        int & n_expert_used_out,
        std::vector<llama_model_tensor_buft_override> & overrides_out);
