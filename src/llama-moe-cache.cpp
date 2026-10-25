#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>

// one MoE cache per process: a file-scope lock is enough to serialise the
// profile counters between the scheduler's copy() threads and the flusher.
static std::mutex g_moe_profile_mu;
#include <thread>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>

// Include for MoE RAM arena integration
#include "llama-model-loader.h"

// Global arena instance
// arena globals are declared in llama-model-loader.h

// Helper to extract tensor type from expert tensor name
// e.g., "blk.0.ffn_gate_exps.weight" -> "gate"
static const char * moe_get_tensor_type(const char * tensor_name) {
    if (!tensor_name) return nullptr;
    const char * p = strstr(tensor_name, "ffn_");
    if (!p) return nullptr;
    p += 4; // skip "ffn_"
    if (strncmp(p, "gate_up", 7) == 0) return "gate_up";
    if (strncmp(p, "gate", 4) == 0) return "gate";
    if (strncmp(p, "up", 2) == 0) return "up";
    if (strncmp(p, "down", 4) == 0) return "down";
    return nullptr;
}

namespace {

// ~ expansion, so presets can use ~/.cache/llama.cpp/...
static std::string moe_profile_expand(const std::string & path) {
    if (path.empty() || path[0] != '~') {
        return path;
    }
    const char * home = getenv("HOME");
    return home != nullptr ? std::string(home) + path.substr(1) : path;
}

static int64_t moe_json_int(const std::string & s, const char * key, int64_t def) {
    const std::string k = std::string("\"") + key + "\"";
    const size_t pos = s.find(k);
    if (pos == std::string::npos) {
        return def;
    }
    const size_t colon = s.find(':', pos + k.size());
    if (colon == std::string::npos) {
        return def;
    }
    char * end = nullptr;
    const long long v = strtoll(s.c_str() + colon + 1, &end, 10);
    return end == s.c_str() + colon + 1 ? def : (int64_t) v;
}

// reads the JSON written by moe_profile_write; tolerant scanner for the known schema
static bool moe_profile_read(const std::string & path, int32_t & n_layer, int32_t & n_expert, std::vector<uint32_t> & counts, uint64_t & tokens) {
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    const int64_t nl = moe_json_int(s, "n_layer", -1);
    const int64_t ne = moe_json_int(s, "n_expert", -1);
    tokens = (uint64_t) std::max<int64_t>(0, moe_json_int(s, "tokens", 0));
    const size_t key = s.find("\"counts\"");
    if (nl <= 0 || ne <= 0 || nl > 4096 || ne > 65536 || key == std::string::npos) {
        return false;
    }
    // write the out-parameters: the locals above shadow them, so without this the
    // caller sees n_layer/n_expert == 0 and rejects a perfectly parsed profile
    n_layer = (int32_t) nl;
    n_expert = (int32_t) ne;
    const char * p = s.c_str() + key;
    p = strchr(p, '[');
    if (p == nullptr) {
        return false;
    }
    ++p;

    counts.assign((size_t) nl * (size_t) ne, 0);
    int32_t row = 0;
    while (*p != 0) {
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',') {
            ++p;
        }
        if (*p == ']' || *p == 0) {
            break;
        }
        if (*p != '[') {
            return false;
        }
        ++p;
        for (int32_t e = 0; e < (int32_t) ne; ++e) {
            while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') {
                ++p;
            }
            char * end = nullptr;
            const unsigned long long v = strtoull(p, &end, 10);
            if (end == p) {
                return false;
            }
            counts[(size_t) row * (size_t) ne + (size_t) e] = (uint32_t) std::min<unsigned long long>(v, UINT32_MAX);
            p = end;
            while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') {
                ++p;
            }
            if (*p == ',') {
                ++p;
            }
        }
        while (*p != 0 && *p != ']') {
            ++p;
        }
        if (*p == ']') {
            ++p;
        }
        ++row;
    }
    return row == (int32_t) nl;
}

// adds the counts to the file if it exists, so profiles accumulate across runs
static bool moe_profile_write(const std::string & path, int32_t n_layer, int32_t n_expert, int32_t n_expert_used, const std::vector<uint32_t> & counts, uint64_t tokens) {
    if (path.empty() || counts.size() != (size_t) n_layer * (size_t) n_expert) {
        return false;
    }
    std::vector<uint32_t> total = counts;
    uint64_t total_tokens = tokens;

    int32_t nl = 0;
    int32_t ne = 0;
    std::vector<uint32_t> prev;
    uint64_t prev_tokens = 0;
    if (moe_profile_read(path, nl, ne, prev, prev_tokens) && nl == n_layer && ne == n_expert) {
        for (size_t i = 0; i < total.size(); ++i) {
            total[i] += prev[i];
        }
        total_tokens += prev_tokens;
    }

    std::error_code ec;
    const std::filesystem::path fp(path);
    if (!fp.parent_path().empty()) {
        std::filesystem::create_directories(fp.parent_path(), ec);
    }
    std::ofstream f(path, std::ios::trunc);
    if (!f) {
        return false;
    }
    f << "{\n  \"version\": 1,\n"
      << "  \"n_layer\": " << n_layer << ",\n"
      << "  \"n_expert\": " << n_expert << ",\n"
      << "  \"n_expert_used\": " << n_expert_used << ",\n"
      << "  \"tokens\": " << total_tokens << ",\n"
      << "  \"counts\": [\n";
    for (int32_t l = 0; l < n_layer; ++l) {
        f << "    [";
        for (int32_t e = 0; e < n_expert; ++e) {
            f << total[(size_t) l * (size_t) n_expert + (size_t) e];
            if (e + 1 < n_expert) {
                f << ",";
            }
        }
        f << "]" << (l + 1 < n_layer ? "," : "") << "\n";
    }
    f << "  ]\n}\n";
    return true;
}

// LRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_lru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // doubly linked list of the slots, head is the least recently used
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head = -1;
    int32_t tail = -1;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    // optional activation counters, indexed il*n_expert + expert
    uint32_t * counts = nullptr;

    // optional pinned set (il*n_expert + expert), pinned experts are never evicted while unpinned slots exist
    const uint8_t * pinned = nullptr;

    // the slot to evict: empty first, then the least recently used unpinned one
    int32_t victim() const {
        for (int32_t s = head; s >= 0; s = next[s]) {
            if (key_of[s] < 0) {
                return s;
            }
        }
        for (int32_t s = head; s >= 0; s = next[s]) {
            if (pinned == nullptr || pinned[key_of[s]] == 0) {
                return s;
            }
        }
        return head;
    }

    // puts expert e of layer il into slot s, used to seed the cache from a profile
    void place(int32_t il, int32_t e, int32_t s) {
        if (key_of[s] >= 0) {
            slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
        }
        key_of[s] = il*n_expert + e;
        slot_map[il][e] = s;
        touch(s);
    }

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = 0;
        tail = n_slots - 1;
        seen.assign(n_expert, 0);
    }

    // move slot s to the tail (most recently used)
    void touch(int32_t s) {
        if (s == tail) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head = next[s];
        }
        prev[next[s]] = prev[s];

        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, the misses evict the least recently used experts
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (counts != nullptr) {
                std::lock_guard<std::mutex> lk(g_moe_profile_mu);
                counts[(size_t) il*n_expert + ids[i]]++;
            }
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // hits go to the tail first, so the head can be evicted below
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = victim();
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            touch(s);
            fills.push_back({ e, s });
        }
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct moe_timing_t {
    std::atomic<uint64_t> n{0}, ids_us{0}, fill_us{0}, total_us{0}, stage_sync_us{0}, stage_sync_n{0};
};
static moe_timing_t g_tm;

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;

        // pinned staging for DISK-tier fills: pread into pinned memory so the
        // H2D copy is true async DMA (mmap sources page-fault and copy synchronously)
        ggml_backend_buffer_ptr buf_stage; // two chunk regions, double buffered
        size_t stage_experts = 32;
        bool stage_outstd[2] = { false, false }; // chunk region has queued H2D copies
        int  stage_flip    = 0;                  // which chunk region to fill next
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    // batches up to three ubatches worth still gain from the cache: prompt
    // eval ubatches of this size route to a few hundred distinct experts
    // per layer, far fewer than the full expert space
    static constexpr int64_t max_batch = 384;

    int32_t n_expert_used;
    int32_t n_expert = 0;
    moe_ram_arena * arena = nullptr; // this model's arena (target vs MTP head)

    std::string profile_path; // record activation counts here
    std::string hot_path;     // pin the hot experts from this profile
    std::vector<uint32_t> counts; // [n_layer*n_expert] token-weighted activation counts
    std::vector<uint8_t>  pinned; // [n_layer*n_expert] 1 if the expert is pinned
    uint64_t profile_tokens = 0;

    // periodic flush of the hot profile, so a long-running server keeps learning
    // which experts are actually selected instead of only recording at shutdown
    std::chrono::steady_clock::time_point last_profile_flush = std::chrono::steady_clock::now();
    uint64_t profile_flush_interval_s = 60;
    uint64_t profile_flush_calls      = 0;
    uint64_t profile_tokens_at_flush  = 0;

    // copy() is invoked from the scheduler for several layers at once, so the
    // counters and the periodic profile write must be serialised. Without this
    // the flush raced the per-token increments (std::fill vs counts[...]++) and
    // corrupted the counters, which surfaced later as a CUDA illegal access.
    std::vector<uint32_t> flushed_counts; // counts already written to the profile

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
         const char * profile_path_, const char * hot_path_) :
            n_expert_used(model.hparams.n_expert_used_max()), n_expert((int32_t) model.hparams.n_expert), layers(model.layers.size()) {
        {
            const auto it = g_moe_arenas.find({ (int) model.hparams.n_layer(), (int) model.hparams.n_expert });
            if (it != g_moe_arenas.end()) {
                arena = it->second.get();
            }
        }
        // 'auto' resolves to <cache dir>/moe-profiles/<model>.json, one file per preset/model
        auto resolve = [&](const char * in) {
            std::string p = moe_profile_expand(in != nullptr ? in : "");
            if (p != "auto") {
                return p;
            }
            const char * cache = getenv("LLAMA_CACHE");
            const char * home  = getenv("HOME");
            const std::string dir = cache != nullptr ? cache : (home != nullptr ? std::string(home) + "/.cache/llama.cpp" : ".");
            std::string name = model.name;
            for (char & c : name) {
                if (!isalnum((unsigned char) c) && c != '-' && c != '_' && c != '.') {
                    c = '-';
                }
            }
            if (name.empty() || name == "n/a") {
                name = "model";
            }
            return dir + "/moe-profiles/" + name + ".json";
        };
        profile_path = resolve(profile_path_);
        hot_path     = resolve(hot_path_);
        if (const char * iv = getenv("LLAMA_MOE_PROFILE_FLUSH_S")) {
            const long v = strtol(iv, nullptr, 10);
            if (v > 0) {
                profile_flush_interval_s = (uint64_t) v;
            }
        }

        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        if (!profile_path.empty() || !hot_path.empty()) {
            counts.assign((size_t) layers.size() * (size_t) n_expert, 0);
            flushed_counts.assign((size_t) layers.size() * (size_t) n_expert, 0);
            pinned.assign((size_t) layers.size() * (size_t) n_expert, 0);
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots);
            if (!counts.empty()) {
                g.lru.counts = counts.data();
            }
            if (!pinned.empty()) {
                g.lru.pinned = pinned.data();
            }
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size();
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
                // pinned staging: two chunks of stage_experts experts, double buffered
                // so a pread batch overlaps the previous batch's H2D copy
                {
                    const size_t dev_idx = (size_t) (&d - devices.data());
                    size_t max_es = 0;
                    for (const group & gg : groups) {
                        if (gg.id == (int32_t) dev_idx && !gg.ref.empty()) {
                            for (const ggml_tensor * t : gg.ref) {
                                max_es = std::max(max_es, (size_t) t->nb[2]);
                            }
                        }
                    }
                    if (max_es > 0) {
                        ggml_backend_buffer_type_t buft_pinned = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(d.backend));
                        if (buft_pinned == nullptr) {
                            buft_pinned = ggml_backend_cpu_buffer_type();
                        }
                        const size_t stage_bytes = 2*d.stage_experts*max_es;
                        d.buf_stage.reset(ggml_backend_buft_alloc_buffer(buft_pinned, stage_bytes));
                        if (!d.buf_stage) {
                            LLAMA_LOG_WARN("%s: failed to allocate %.2f MiB of pinned staging, DISK-tier fills use the mmap path\n",
                                           __func__, stage_bytes/1024.0/1024.0);
                        }
                    }
                }
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
        }

        seed_from_profile();

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    // place the most activated experts of each layer in the cache and pin them, so the LRU only rotates the cold rest
    void seed_from_profile() {
        if (hot_path.empty() || pinned.empty()) {
            return;
        }
        // Escape hatch: pre-pinning a large hot set faults in sustained decode on
        // this fork (CUDA illegal access in the MoE matvec after the fill). The
        // profile is still learned and still drives tier placement; only the
        // eager VRAM seed is skipped.
        if (getenv("LLAMA_MOE_NO_PIN") != nullptr) {
            LLAMA_LOG_WARN("%s: LLAMA_MOE_NO_PIN set, skipping profile seeding\n", __func__);
            return;
        }
        int32_t nl = 0;
        int32_t ne = 0;
        std::vector<uint32_t> hot;
        uint64_t selections = 0;
        const bool seed_read_ok = moe_profile_read(hot_path, nl, ne, hot, selections);
        LLAMA_LOG_INFO("seed_from_profile: read_ok=%d nl=%d ne=%d tokens=%llu hot=%zu cache_ne=%d layers=%zu\n",
                       (int) seed_read_ok, nl, ne, (unsigned long long) selections, hot.size(), n_expert, layers.size());
        if (!seed_read_ok || ne != n_expert || nl <= 0 || nl > (int32_t) layers.size()) {
            LLAMA_LOG_WARN("%s: could not read MoE profile '%s', starting with a cold cache\n", __func__, hot_path.c_str());
            return;
        }
        // models with an MTP/nextn block have one more layer than the profile covers;
        // accept a prefix profile and treat the missing tail layers as cold
        if (nl < (int32_t) layers.size()) {
            hot.resize((size_t) layers.size() * (size_t) n_expert, 0);
        }

        size_t n_seeded = 0;
        size_t n_bytes  = 0;
        struct seed_job { int32_t il; int32_t e; int32_t slot; };
        std::vector<seed_job> seed_jobs;
        for (group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            // Keep a real free pool, not just one selection. Layers whose profile counts
            // are all zero (never cache-managed before) get nothing pinned, so a 10-slot
            // free pool makes them thrash every token and can hand the kernel slots it
            // cannot serve. A fifth of the cache stays unpinned and rotates.
            const int32_t free_pool = std::max<int32_t>(n_expert_used * 8, g.n_slots / 5);
            const int32_t budget    = std::max<int32_t>(0, g.n_slots - std::min<int32_t>(free_pool, g.n_slots));
            if (budget == 0) {
                continue;
            }

            std::vector<double> mass(g.layers.size(), 0.0);
            double total = 0.0;
            for (size_t li = 0; li < g.layers.size(); ++li) {
                const int32_t il = g.layers[li];
                for (int32_t e = 0; e < n_expert; ++e) {
                    mass[li] += hot[(size_t) il*n_expert + e];
                }
                total += mass[li];
            }
            if (total <= 0.0) {
                continue;
            }

            // pin budget per layer proportional to its activation mass, then hand out the remainder
            std::vector<int32_t> take(g.layers.size(), 0);
            std::vector<std::pair<double, size_t>> frac;
            int32_t assigned = 0;
            for (size_t li = 0; li < g.layers.size(); ++li) {
                const double exact = budget * mass[li] / total;
                take[li] = (int32_t) exact;
                assigned += take[li];
                frac.emplace_back(exact - take[li], li);
            }
            std::sort(frac.begin(), frac.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
            for (size_t i = 0; assigned < budget && i < frac.size(); ++i) {
                take[frac[i].second]++;
                assigned++;
            }

            int32_t slot = 0;
            for (size_t li = 0; li < g.layers.size() && slot < budget; ++li) {
                const int32_t il = g.layers[li];
                if (take[li] == 0 || layers[il].slot_map == nullptr) {
                    continue;
                }
                std::vector<std::pair<uint32_t, int32_t>> rank;
                rank.reserve(n_expert);
                for (int32_t e = 0; e < n_expert; ++e) {
                    const uint32_t c = hot[(size_t) il*n_expert + e];
                    if (c > 0) {
                        rank.emplace_back(c, e);
                    }
                }
                std::sort(rank.begin(), rank.end(), [](const auto & a, const auto & b) {
                    return a.first != b.first ? a.first > b.first : a.second < b.second;
                });
                const int32_t n = std::min<int32_t>(take[li], (int32_t) rank.size());
                for (int32_t i = 0; i < n && slot < budget; ++i) {
                    const int32_t e = rank[i].second;
                    g.lru.place(il, e, slot);
                    pinned[(size_t) il*n_expert + e] = 1;
                    seed_jobs.push_back({ il, e, slot });
                    n_seeded++;
                    slot++;
                }
            }
        }

        // Upload the pinned experts. Grouped per (layer, expert tensor) span so a
        // whole span is uploaded in one backend tensor_set call; span reads use
        // the parallel pread workers on the shard fds instead of faulting the
        // mmap view expert by expert (that path ran at a fraction of the drives'
        // random-read capacity and dominated the load time).
        if (!seed_jobs.empty()) {
            auto expert_at = [](const std::vector<seed_job> & jobs_, int32_t slot) -> int32_t {
                for (const auto & j : jobs_) {
                    if (j.slot == slot) {
                        return j.e;
                    }
                }
                return -1;
            };
            std::map<std::pair<int32_t,int32_t>, std::vector<seed_job>> type_jobs; // (il, ip)
            for (const auto & j : seed_jobs) {
                for (size_t ip = 0; ip < layers[j.il].experts.size(); ++ip) {
                    type_jobs[{ j.il, (int32_t) ip }].push_back(j);
                }
            }
            for (auto & [k, jobs] : type_jobs) {
                const auto [il, ip] = k;
                const layer & l = layers[il];
                group * g = nullptr;
                for (group & gg : groups) {
                    for (int32_t ili : gg.layers) {
                        if (ili == il) { g = &gg; break; }
                    }
                    if (g) break;
                }
                if (!g || g->n_slots == 0 || ip >= g->banks.size()) {
                    continue;
                }
                ggml_tensor       * bank = g->banks[ip];
                const ggml_tensor * w    = l.experts[ip];
                const char * wname = w->name ? w->name : "";
                const char * wtype = strstr(wname, "ffn_gate_up_exps") ? "gate_up" :
                                     strstr(wname, "ffn_gate_exps")    ? "gate" :
                                     strstr(wname, "ffn_up_exps")      ? "up" :
                                     strstr(wname, "ffn_down_exps")    ? "down" : nullptr;
                const size_t stride = w->nb[2];
                // consecutive slots of one layer form a span within the bank
                std::vector<std::pair<int32_t,int32_t>> spans; // (slot0, n)
                int32_t cur0 = -2;
                int32_t curl = 0;
                std::sort(jobs.begin(), jobs.end(), [](const seed_job & a, const seed_job & b) {
                    return a.slot < b.slot;
                });
                for (const auto & j : jobs) {
                    if (j.slot == cur0 + curl) {
                        curl++;
                    } else {
                        if (curl > 0) spans.emplace_back(cur0, curl);
                        cur0 = j.slot;
                        curl = 1;
                    }
                }
                if (curl > 0) spans.emplace_back(cur0, curl);

                for (const auto & [slot0, count] : spans) {
                    if (!arena || !wtype) {
                        for (int32_t i = 0; i < count; ++i) {
                            // slots in this span belong to this layer, so the
                            // expert id comes from the jobs we collected
                            const int32_t e = expert_at(jobs, slot0 + i);
                            if (e < 0) {
                                continue;
                            }
                            ggml_backend_tensor_set(bank,
                                (const uint8_t *) w->data + (size_t) e*stride,
                                (size_t) (slot0 + i)*bank->nb[2], stride);
                            n_bytes += stride;
                        }
                        continue;
                    }
                    // reuse the device's pinned double-buffered staging (the same
                    // chunks the DISK-tier fills use): pread the slice in parallel
                    // workers, then queue one H2D copy; the next slice reads into
                    // the other buffer while the copy drains async
                    ggml_backend_buffer_t stage_buf = devices[g->id].buf_stage.get();
                    if (!stage_buf) {
                        for (int32_t i = 0; i < count; ++i) {
                            const int32_t e = expert_at(jobs, slot0 + i);
                            if (e < 0) {
                                continue;
                            }
                            ggml_backend_tensor_set(bank,
                                (const uint8_t *) w->data + (size_t) e*stride,
                                (size_t) (slot0 + i)*bank->nb[2], stride);
                            n_bytes += stride;
                        }
                        continue;
                    }
                    device & d = devices[g->id];
                    const size_t es = std::min(stride, (size_t) bank->nb[2]);
                    uint8_t * stage_base = (uint8_t *) ggml_backend_buffer_get_base(stage_buf);
                    const size_t K = d.stage_experts;
                    const int n_workers = (int) std::min<size_t>(K, std::max(1u, std::thread::hardware_concurrency() / 2));
                    for (int32_t s = 0; s < count; s += (int32_t) K) {
                        const int32_t ci = (int32_t) std::min<size_t>((size_t) count - s, K);
                        const int f = d.stage_flip;
                        d.stage_flip ^= 1;
                        if (d.stage_outstd[f]) {
                            ggml_backend_synchronize(d.backend);
                            d.stage_outstd[f] = false;
                        }
                        // resolve expert ids for this chunk in slot order
                        std::vector<int32_t> ids(ci, -1);
                        for (const auto & j : jobs) {
                            if (j.slot >= slot0 + s && j.slot < slot0 + s + ci) {
                                ids[j.slot - slot0 - s] = j.e;
                            }
                        }
                        const int f0 = f;
                        std::atomic<int32_t> next(0);
                        std::atomic<bool> any_fail(false);
                        auto worker = [&]() {
                            for (;;) {
                                const int32_t k = next.fetch_add(1);
                                if (k >= ci) {
                                    break;
                                }
                                uint8_t * dstp = stage_base + (size_t) f0*K*es + (size_t) k*es;
                                if (ids[k] < 0) {
                                    continue;
                                }
                                if (arena->read_expert_direct(il, wtype, ids[k], dstp) == 0) {
                                    any_fail.store(true);
                                    break;
                                }
                            }
                        };
                        if (n_workers <= 1) {
                            worker();
                        } else {
                            std::vector<std::thread> pool;
                            for (int t = 1; t < n_workers; ++t) {
                                pool.emplace_back(worker);
                            }
                            worker();
                            for (auto & th : pool) {
                                th.join();
                            }
                        }
                        if (any_fail.load()) {
                            // mmap fallback copies for the failed chunk
                            for (int32_t k = 0; k < ci; ++k) {
                                const int32_t e = ids[k];
                                if (e < 0) {
                                    continue;
                                }
                                ggml_backend_tensor_set(bank,
                                    (const uint8_t *) w->data + (size_t) e*stride,
                                    (size_t) (slot0 + s + k)*bank->nb[2], es);
                                n_bytes += es;
                            }
                            continue;
                        }
                        uint8_t * src = stage_base + (size_t) f0*K*es;
                        ggml_backend_tensor_set_async(d.backend, bank, src,
                            (size_t) (slot0 + s)*bank->nb[2], (size_t) ci*es);
                        n_bytes += (size_t) ci * es;
                        d.stage_outstd[f] = true;
                    }
                }
            }
            // flush the last H2D copies queued for each device
            for (device & d : devices) {
                if (d.backend) {
                    ggml_backend_synchronize(d.backend);
                }
            }
        }

        LLAMA_LOG_INFO("%s: pinned %zu experts (%.2f MiB) from '%s' (%llu recorded selections)\n", __func__,
            n_seeded, n_bytes/1024.0/1024.0, hot_path.c_str(), (unsigned long long) selections);
    }

    // write the accumulated activation counts to the profile and clear them.
    // moe_profile_write adds to the file, so the in-memory counts must be zeroed
    // afterwards or every flush would count the same selections again.
    bool flush_profile() {
        if (profile_path.empty() || counts.empty()) {
            return false;
        }

        // Snapshot only what is new since the previous flush, then release the
        // lock before touching the filesystem: the write must not stall decode.
        std::vector<uint32_t> delta;
        uint64_t tokens = 0;
        {
            std::lock_guard<std::mutex> lk(g_moe_profile_mu);
            if (profile_tokens == profile_tokens_at_flush) {
                return false; // nothing new since the last flush
            }
            if (flushed_counts.size() != counts.size()) {
                flushed_counts.assign(counts.size(), 0);
            }
            delta.resize(counts.size());
            bool any = false;
            for (size_t i = 0; i < counts.size(); ++i) {
                delta[i] = counts[i] - flushed_counts[i];
                any = any || delta[i] != 0;
            }
            if (!any) {
                profile_tokens_at_flush = profile_tokens;
                return false;
            }
            tokens = profile_tokens;
            profile_tokens_at_flush = profile_tokens;
            flushed_counts = counts;
        }

        if (!moe_profile_write(profile_path, (int32_t) layers.size(), n_expert, n_expert_used, delta, tokens)) {
            LLAMA_LOG_WARN("llama_moe_cache: failed to flush MoE profile to '%s'\n", profile_path.c_str());
            return false;
        }
        LLAMA_LOG_INFO("llama_moe_cache: flushed MoE profile to '%s' (%llu selections)\n",
            profile_path.c_str(), (unsigned long long) tokens);
        last_profile_flush = std::chrono::steady_clock::now();
        return true;
    }

    // called from the expert copy path; cheap because the clock is only read
    // every 32 calls and the profile write happens at most once per interval
    void maybe_flush_profile() {
        if (profile_path.empty() || counts.empty()) {
            return;
        }
        if (++profile_flush_calls % 32 != 0) {
            return;
        }
        if (std::chrono::steady_clock::now() - last_profile_flush < std::chrono::seconds(profile_flush_interval_s)) {
            return;
        }
        flush_profile();
    }

    ~impl() {
        log_stats();
        flush_profile();
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, ggml_backend_sched_t sched) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        // the ids in the split graph are scheduler copies; resolve to the original tensor,
        // which is already valid: the producing split ran before this copy callback fires
        ggml_backend_t sel_backend = backend;
        if (sched) {
            const ggml_tensor * sel_src = ggml_backend_sched_copy_source(sched, sel);
            if (sel_src) {
                sel = sel_src;
                ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, const_cast<ggml_tensor *>(sel_src));
                if (b) {
                    sel_backend = b;
                }
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        const auto tm0 = std::chrono::steady_clock::now();
        auto tm_mark = tm0;
        auto us_since = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
            return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
        };
        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(sel_backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(sel_backend);
        {
            const auto t_now = std::chrono::steady_clock::now();
            g_tm.ids_us.fetch_add(us_since(tm_mark, t_now), std::memory_order_relaxed);
            tm_mark = t_now;
        }

        {
            static std::atomic<int> moez_calls{0};
            if (moez_calls.fetch_add(1) < 24) {
                int32_t * slots_dbg = g.lru.slot_map[il];
                int n_mapped = 0;
                for (int32_t e = 0; e < n_expert && slots_dbg; ++e) {
                    if (slots_dbg[e] >= 0) n_mapped++;
                }
                LLAMA_LOG_WARN("MOEZ il=%d slots_ptr=%p n_slots=%d mapped_pre=%d ids0=%d ids1=%d nids=%zu\n",
                    il, (void*) slots_dbg, g.n_slots, n_mapped, ids.empty() ? -9 : ids[0], ids.size() > 1 ? ids[1] : -9, ids.size());
            }
        }

        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        if (getenv("LLAMA_MOE_DEBUG")) {
            fprintf(stderr, "MOECACHE il=%d n_slots=%d ids=%zu hits=%zu fills=%zu", il, g.n_slots, ids.size(), n_hit, fills.size());
            for (size_t i = 0; i < ids.size() && i < 12; ++i) {
                fprintf(stderr, " e%d->s%d", ids[i], g.lru.slot_map[il][ids[i]]);
            }
            fprintf(stderr, "\n");
        }
        if (!counts.empty()) {
            profile_tokens += ids.size() / (size_t) std::max(1, n_expert_used);
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        
        // Before uploading, ensure RAM-tier experts are loaded from disk into moe_ram_arena combined buffers
        if (arena) {
            // Get expert IDs from fills
            std::vector<int> expert_ids;
            expert_ids.reserve(fills.size());
            for (const auto & fill : fills) {
                expert_ids.push_back(fill.expert);
            }
            
            // For each expert tensor type, ensure experts are loaded
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                const char * tensor_type = moe_get_tensor_type(w->name);
                if (tensor_type) {
                    arena->ensure_experts_loaded(il, tensor_type, expert_ids.data(), (int)expert_ids.size());
                }
            }
        }
        
        const auto tm_fill0 = std::chrono::steady_clock::now();
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];

            // RAM tier: source is the arena combined buffer (resident, or loaded
            // above by the parallel preads of ensure_experts_loaded)
            const char * tensor_type = moe_get_tensor_type(w->name);
            void * src_data = w->data;
            bool have_combined = false;
            {
                static std::atomic<int> dbg_es{0};
                if (dbg_es.fetch_add(1) < 16) {
                    LLAMA_LOG_WARN("MOEES il=%d type=%s src_es=%zu bank_es=%zu bank_bytes=%zu n_slots=%d fills=%zu first=%d last=%d\n",
                        il, tensor_type ? tensor_type : "?", expert_size, (size_t) bank->nb[2], ggml_nbytes(bank),
                        g.n_slots, fills.size(),
                        fills.empty() ? -1 : fills.front().slot,
                        fills.empty() ? -1 : fills.back().slot);
                }
            }
            if (tensor_type && arena) {
                void * combined = arena->get_combined_buffer(il, tensor_type);
                if (combined) {
                    src_data = combined;
                    have_combined = true;
                }
            }

            {
                static std::atomic<int> dbg_fill{0};
                if (dbg_fill.fetch_add(1) < 24) {
                    const bool has_dm = tensor_type && arena &&
                        (arena->disk_meta.count(std::make_pair(il, std::string(tensor_type))) > 0);
                    LLAMA_LOG_INFO("MOEFILL il=%d type=%s fills=%zu comb=%d stage=%p dm=%d\n",
                                   il, tensor_type ? tensor_type : "?", fills.size(),
                                   (int) have_combined, (void *) devices[g.id].buf_stage.get(), (int) has_dm);
                }
            }
            if (!have_combined && tensor_type && arena &&
                devices[g.id].buf_stage != nullptr) {
                // DISK tier: stream the experts through pinned staging in chunks;
                // the preads run in parallel workers and overlap the H2D copies of
                // the previous chunk (reading through the mmap view instead page
                // faults one 4 KiB page at a time inside the synchronous copy)
                device & dev = devices[g.id];
                const size_t K = dev.stage_experts;
                uint8_t * stage_base = (uint8_t *) ggml_backend_buffer_get_base(dev.buf_stage.get());
                const int n_workers = (int) std::min<size_t>(K, std::max(1u, std::thread::hardware_concurrency() / 2));
                size_t i = 0;
                while (i < fills.size()) {
                    const size_t c0 = i;
                    size_t ci = 0;
                    while (i < fills.size() && ci < K) {
                        i++;
                        ci++;
                    }
                    const int f = dev.stage_flip;
                    dev.stage_flip ^= 1;
                    if (dev.stage_outstd[f]) {
                        // this chunk region still has queued copies from an earlier
                        // layer; wait for them before overwriting the pinned memory
                        {
                            const auto s0 = std::chrono::steady_clock::now();
                            ggml_backend_synchronize(backend);
                            g_tm.stage_sync_us.fetch_add(us_since(s0, std::chrono::steady_clock::now()), std::memory_order_relaxed);
                            g_tm.stage_sync_n.fetch_add(1, std::memory_order_relaxed);
                        }
                        dev.stage_outstd[f] = false;
                    }
                    std::atomic<size_t> next(c0);
                    std::atomic<bool> any_fail(false);
                    auto worker = [&]() {
                        for (;;) {
                            const size_t k = next.fetch_add(1);
                            if (k >= c0 + ci) {
                                return;
                            }
                            void * dstp = stage_base + (size_t) f*K*expert_size + (k - c0)*expert_size;
                            if (arena->read_expert_direct(il, tensor_type, fills[k].expert, dstp) == 0) {
                                any_fail.store(true);
                                return;
                            }
                        }
                    };
                    // decode fills are usually 1-4 experts: spawning a pool costs
                    // more than the preads themselves, so run those inline
                    if (n_workers <= 1 || ci <= 4) {
                        worker();
                    } else {
                        std::vector<std::thread> pool;
                        for (int t = 1; t < n_workers; ++t) {
                            pool.emplace_back(worker);
                        }
                        worker();
                        for (auto & th : pool) {
                            th.join();
                        }
                    }
                    if (any_fail.load()) {
                        // no file metadata for this layer/type: mmap fallback
                        for (size_t k = c0; k < c0 + ci; ++k) {
                            const size_t bank_es = bank->nb[2];
                            const size_t cpy     = std::min(expert_size, bank_es);
                            if (fills[k].slot < 0 || fills[k].slot >= g.n_slots) {
                                continue;
                            }
                            ggml_backend_tensor_set_async(backend, bank,
                                (const uint8_t *) src_data + fills[k].expert*expert_size,
                                (size_t) fills[k].slot*bank_es, cpy);
                            bytes += cpy;
                        }
                        continue;
                    }
                    for (size_t k = c0; k < c0 + ci; ++k) {
                        const size_t bank_es = bank->nb[2];
                        const size_t cpy     = std::min(expert_size, bank_es);
                        if (fills[k].slot < 0 || fills[k].slot >= g.n_slots) {
                            LLAMA_LOG_WARN("MOEFILL: slot %d out of range (n_slots=%d) il=%d type=%s, skipping\n",
                                           fills[k].slot, g.n_slots, il, tensor_type ? tensor_type : "?");
                            continue;
                        }
                        ggml_backend_tensor_set_async(backend, bank,
                            stage_base + (size_t) f*K*expert_size + (k - c0)*expert_size,
                            (size_t) fills[k].slot*bank_es, cpy);
                        bytes += cpy;
                    }
                    dev.stage_outstd[f] = true;
                }
                continue;
            }

            // RAM tier (or arena-less): upload from the resident source
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                const size_t bank_es = bank->nb[2];
                const size_t cpy     = std::min(expert_size, bank_es);
                if (fills[i].slot < 0 || fills[i].slot + (int32_t) n > g.n_slots) {
                    i += n;
                    continue;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) src_data + fills[i].expert*expert_size, (size_t) fills[i].slot*bank_es, std::min(cpy*n, bank_es*n));
                bytes += n*cpy;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        {
            const auto t_now = std::chrono::steady_clock::now();
            g_tm.fill_us.fetch_add(us_since(tm_fill0, t_now), std::memory_order_relaxed);
            g_tm.total_us.fetch_add(us_since(tm0, t_now), std::memory_order_relaxed);
            const uint64_t n = g_tm.n.fetch_add(1, std::memory_order_relaxed) + 1;
            if (n % 32 == 0) {
                LLAMA_LOG_WARN("MOETM layers=%llu avg_total=%.2fms avg_ids=%.2fms avg_fill=%.2fms avg_stagesync=%.2fms (%llu syncs)\n",
                    (unsigned long long) n,
                    g_tm.total_us.load()/1000.0/n, g_tm.ids_us.load()/1000.0/n,
                    g_tm.fill_us.load()/1000.0/n,
                    g_tm.stage_sync_n.load() ? g_tm.stage_sync_us.load()/1000.0/g_tm.stage_sync_n.load() : 0.0,
                    (unsigned long long) g_tm.stage_sync_n.load());
                g_tm.n.store(0); g_tm.ids_us.store(0); g_tm.fill_us.store(0);
                g_tm.total_us.store(0); g_tm.stage_sync_us.store(0); g_tm.stage_sync_n.store(0);
            }
        }

        maybe_flush_profile();
        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
        const char * profile_path, const char * hot_path) :
    pimpl(new impl(model, backends, bufts, size, profile_path, hot_path)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, ggml_backend_sched_t sched) {
    return pimpl->copy(backend, src, dst, graph, sched);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
