#include "llama-moe-cache.h"
#include "llama-moe-tier-policy.h"

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

static std::mutex g_moe_profile_mu;

static const char * moe_get_tensor_type(const char * tensor_name) {
    if (!tensor_name) return nullptr;
    const char * p = strstr(tensor_name, "ffn_");
    if (!p) return nullptr;
    p += 4;
    if (strncmp(p, "gate_up", 7) == 0) return "gate_up";
    if (strncmp(p, "gate", 4) == 0) return "gate";
    if (strncmp(p, "up", 2) == 0) return "up";
    if (strncmp(p, "down", 4) == 0) return "down";
    return nullptr;
}

namespace {

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

struct moe_cache_lru {
    static constexpr int   n_class       = 8;
    static constexpr float crf_half_life = 128.0f;

    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    int32_t empty    = 0;

    std::vector<int32_t *> slot_map;
    std::vector<int32_t>   key_of;

    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head[n_class];
    int32_t tail[n_class];
    std::vector<int8_t>  list_of;
    int32_t n_filled = 0;

    std::vector<float>   crf;
    std::vector<int64_t> last_use;
    std::vector<int64_t> clock;

    std::vector<uint32_t> seen;
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;
    std::vector<int32_t> misses;
    std::vector<uint64_t> token_zero;

    // Basalt's promotion rule: a second miss of the same expert soon after the
    // first means the routing is recycling it, so a slot is worth spending and
    // the expert must not re-enter at the bottom protection class. `last_miss`
    // holds the token + 1 of the previous miss (0 = never seen).
    std::vector<uint64_t> last_miss;
    std::vector<uint8_t>  promote;
    uint64_t              token = 0;
    // The window is what separates the two modes: Basalt promotes on a 64-token
    // horizon, Strata adapts over a longer one so a conversation's hot set survives
    // several turns before re-promotion.
    int64_t promote_window = llama_moe_tier::kPromoteWindow;

    uint32_t * counts = nullptr;

    // n_zero adds one always-zero bank row per CPU lane; the extra +1 is the
    // "empty" row, which every evicted expert points at. mul_mat_id dereferences
    // it unconditionally, so it must exist or the bank is read out of bounds.
    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots, int32_t n_zero) {
        GGML_ASSERT(n_class == (int) std::log2(1.0f/(1.0f - std::exp2(-1.0f/crf_half_life))) + 1);
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        this->empty    = n_slots + n_zero;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        for (int c = 0; c < n_class; ++c) {
            head[c] = -1;
            tail[c] = -1;
        }
        if (n_slots > 0) {
            head[0] = 0;
            tail[0] = n_slots - 1;
        }
        list_of.assign(n_slots, 0);
        n_filled = 0;
        crf.assign((size_t) n_layer*n_expert, 0.0f);
        last_use.assign((size_t) n_layer*n_expert, -1);
        clock.assign(n_layer, 0);
        seen.assign(n_expert, 0);
        if (const char * w = getenv("LLAMA_MOE_PROMOTE_WINDOW")) {
            const int64_t v = (int64_t) atoll(w);
            if (v > 0) {
                promote_window = v;
            }
        }
        const char * mode = getenv("LLAMA_MOE_MODE");
        LLAMA_LOG_INFO("%s: MoE mode %s, repeat-miss promotion window %lld tokens\n",
                       __func__, mode ? mode : "default", (long long) promote_window);
        last_miss.assign((size_t) n_layer*n_expert, 0);
        promote.assign((size_t) n_layer*n_expert, 0);
        token = 0;
    }

    void move(int32_t s, int c) {
        const int from = list_of[s];
        if (from == c && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[c];
        next[s] = -1;
        if (tail[c] >= 0) {
            next[tail[c]] = s;
        } else {
            head[c] = s;
        }
        tail[c] = s;
        list_of[s] = (int8_t) c;
    }

    int32_t make_key(int32_t il, int32_t e) const { return il*n_expert + e; }
    int32_t key_layer (int32_t key) const { return key / n_expert; }
    int32_t key_expert(int32_t key) const { return key % n_expert; }

    int class_of(int32_t key) const {
        const float v = std::max(1.0f, crf[key]);
        return std::min(n_class - 1, (int) std::log2(v));
    }

    float decayed_crf(int32_t il, int32_t key) const {
        if (last_use[key] < 0) {
            return crf[key];
        }
        return crf[key]*std::exp2(-(clock[il] - last_use[key])/crf_half_life);
    }

    void use(int32_t il, int32_t key) {
        crf[key]      = decayed_crf(il, key) + 1.0f;
        last_use[key] = clock[il];
    }

    void place(int32_t il, int32_t e, int32_t s, float init_crf = 16.0f) {
        GGML_ASSERT(s >= 0 && s < n_slots);
        if (key_of[s] < 0) {
            n_filled++;
        }
        const int32_t key = make_key(il, e);
        key_of[s]        = key;
        slot_map[il][e]  = s;
        crf[key]         = init_crf;
        last_use[key]    = 0;
        move(s, class_of(key));
    }

    bool is_selected(int32_t il, int32_t key) const {
        if (key < 0) {
            return false;
        }
        if (key_layer(key) != il) {
            return false;
        }
        return seen[key_expert(key)] == seen_gen;
    }

    int32_t candidate(int c, int32_t il) const {
        for (int32_t s = head[c]; s >= 0; s = next[s]) {
            const int32_t k = key_of[s];
            // A slot may only be recycled by the layer that owns it. Stealing
            // another layer's slot corrupts that layer: its slot map was already
            // uploaded for this ubatch and still points at the old row, which now
            // holds a different layer's expert weights.
            if (k >= 0 && key_layer(k) != il) {
                continue;
            }
            if (!is_selected(il, k)) {
                return s;
            }
        }
        return -1;
    }

    int32_t victim(int32_t il) const {
        int32_t res = -1;
        double res_score = -1.0;
        for (int c = 0; c < n_class; ++c) {
            const int32_t s = candidate(c, il);
            if (s < 0) {
                continue;
            }
            const int32_t key = key_of[s];
            if (key < 0) {
                return s;
            }
            const int64_t age   = clock[key_layer(key)] - last_use[key];
            const double  score = std::ldexp((double) age, -c);
            if (score > res_score) {
                res       = s;
                res_score = score;
            }
        }
        return res;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    template <typename F, typename P>
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int64_t n_tokens, int32_t n_used, int32_t n_zero, F && n_cpu, P && cpu_ok,
            std::vector<fill> & fills, std::vector<fill> & cpu, size_t & n_hit) {
        fills.clear();
        cpu.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        if (counts != nullptr) {
            std::lock_guard<std::mutex> lk(g_moe_profile_mu);
            uint32_t * row = counts + (size_t) il*n_expert;
            for (size_t i = 0; i < n_ids; ++i) {
                GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
                row[ids[i]]++;
                if (seen[ids[i]] != seen_gen) {
                    seen[ids[i]] = seen_gen;
                    uniq.push_back(ids[i]);
                }
            }
        } else {
            for (size_t i = 0; i < n_ids; ++i) {
                GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
                if (seen[ids[i]] != seen_gen) {
                    seen[ids[i]] = seen_gen;
                    uniq.push_back(ids[i]);
                }
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        misses.clear();
        for (int32_t e : uniq) {
            const int32_t key = make_key(il, e);
            use(il, key);
            if (slots[e] < n_slots) {
                move(slots[e], class_of(key));
                n_hit++;
            } else {
                // LLAMA_MOE_PROMOTE=0 keeps the pre-port policy for A/B measurement;
                // it must be read at run time so both arms share one binary.
                static const bool promote_enabled = [] {
                    const char * e = getenv("LLAMA_MOE_PROMOTE");
                    return !(e && atoi(e) == 0);
                }();
                if (promote_enabled &&
                    llama_moe_tier::repeat_miss(last_miss[key], token, promote_window)) {
                    promote[key] = 1;
                }
                last_miss[key] = token + 1;
                misses.push_back(e);
            }
        }

        const size_t n_c = std::min(n_cpu(misses.size()), misses.size());
        // Residency decides who runs on the CPU: an expert that already sits in
        // host RAM costs a CPU matvec, while an expert that does not must be read
        // from the model file and uploaded. Handing the resident set to the CPU is
        // what removes it from the PCIe critical path.
        if (n_zero > 0) {
            std::vector<int32_t> to_fill;
            to_fill.reserve(misses.size());
            token_zero.assign(n_tokens, 0);
            size_t n_cpu_used = 0;
            for (int32_t e : misses) {
                if (n_cpu_used >= n_c || !cpu_ok(e)) {
                    to_fill.push_back(e);
                    continue;
                }
                // Lanes are all-zero rows, so any of them is numerically correct;
                // they only have to be distinct per token, otherwise the CUDA id
                // sort sees two experts on one row and under-counts the map.
                uint64_t used = 0;
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            used |= token_zero[it];
                        }
                    }
                }
                int32_t z = 0;
                while (z < n_zero && ((used >> z) & 1ull)) {
                    z++;
                }
                if (z >= n_zero) {
                    to_fill.push_back(e);
                    continue;
                }
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            token_zero[it] |= 1ull << z;
                        }
                    }
                }
                slots[e] = n_slots + z;
                cpu.push_back({ e, n_slots + z });
                n_cpu_used++;
            }
            misses.swap(to_fill);
        }

        std::sort(misses.begin(), misses.end());
        for (int32_t e : misses) {
            const int32_t s = victim(il);
            if (s < 0) {
                // Every slot holds an expert selected this generation. Mark the
                // expert non-resident instead of leaving a stale slot: the cached
                // branch then reads the zero row and the CPU branch computes it.
                if (slots[e] < n_slots) {
                    continue;
                }
                slots[e] = empty;
                continue;
            }
            if (key_of[s] >= 0) {
                slot_map[key_layer(key_of[s])][key_expert(key_of[s])] = empty;
            } else {
                n_filled++;
            }
            const int32_t key = make_key(il, e);
            if (promote[key]) {
                // The expert missed twice inside the promotion window: give it the
                // protection a steady hit would have built up, so the next eviction
                // round does not immediately take the slot back.
                crf[key]      = std::max(crf[key], 16.0f);
                promote[key]  = 0;
            }
            key_of[s] = key;
            slots[e]  = s;
            move(s, class_of(key));
            fills.push_back({ e, s });
        }
        clock[il] += n_tokens;
        token      += (uint64_t) n_tokens;
        return true;
    }
};

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

// A layer can serve experts from the CPU only when every one of its expert
// tensors is backed by a real host buffer of the full tensor size. Lazy/disk
// tier tensors carry a placeholder buffer, and reading those rows would compute
// garbage, so they stay on the PCIe path.
static bool llama_moe_cache_layer_cpu_ready(const llama_layer & layer) {
    const auto ts = llama_moe_cache_layer_experts(layer);
    if (ts.empty()) {
        return false;
    }
    for (const ggml_tensor * w : ts) {
        if (w->buffer == nullptr || w->data == nullptr ||
            !ggml_backend_buffer_is_host(w->buffer) ||
            ggml_backend_buffer_get_size(w->buffer) != (size_t) ggml_nbytes(w) ||
            ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
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

struct llama_moe_cache::impl {
    struct device {
        ggml_backend_t backend = nullptr;
        ggml_backend_buffer_type_t buft = nullptr;
        ggml_backend_buffer_type_t buft_pinned = nullptr;
        size_t host_bytes = 0;
        double split = 0.0;

        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;

        ggml_backend_buffer_ptr buf_stage;
        ggml_backend_event_t stage_ev[2] = { nullptr, nullptr };
        size_t stage_experts = getenv("LLAMA_MOE_STAGE_EXPERTS") ? (size_t) atoi(getenv("LLAMA_MOE_STAGE_EXPERTS")) : 32;
        bool stage_outstd[2] = { false, false };
        int  stage_flip    = 0;
    };

    struct group {
        int32_t id = -1;
        std::vector<ggml_tensor *> ref;
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks;
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
    };

    struct layer {
        int32_t ig = -1;
        ggml_tensor * slot_map = nullptr;
        ggml_tensor * slot_map_cpu = nullptr;
        ggml_tensor * cpu_map  = nullptr;
        std::vector<int32_t> cpu_experts;
        // arena tensor types this layer reads (gate/up/down/gate_up); an expert is
        // CPU-eligible only when it is resident in all of them
        std::vector<std::string> cpu_types;
        // true only when every expert tensor of this layer has real host-resident
        // data: a lazy/disk-tier tensor carries a dummy buffer, and computing it on
        // the CPU would read unmapped memory and corrupt the output.
        bool cpu_ready = false;
        uint32_t cpu_acc = 0;
        std::vector<ggml_tensor *> experts;
        bool arena_has_disk = false;
        bool arena_has_comb = false;
    };

    struct binding {
        int32_t il;
        int32_t ip;
        ggml_tensor * cached;
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t cpu    = 0;
        size_t bytes  = 0;
    };

    int64_t max_batch = 16;
    int32_t n_expert_used = 0;
    int32_t n_expert = 0;
    int32_t n_zero = 0;
    float   cpu_share = 0.0f;
    uint32_t cpu_share_q = 0;
    moe_ram_arena * arena = nullptr;
    bool debug_log = false;

    std::string profile_path;
    std::string hot_path;
    std::vector<uint32_t> counts;
    std::vector<uint32_t> flushed_counts;
    uint64_t profile_tokens = 0;
    std::chrono::steady_clock::time_point last_profile_flush = std::chrono::steady_clock::now();
    uint64_t profile_flush_interval_s = 60;
    uint64_t profile_flush_calls      = 0;
    uint64_t profile_tokens_at_flush  = 0;

    stats stats_small;
    stats stats_large;
    stats stats_copy;

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings;
    std::unordered_map<const ggml_tensor *, std::pair<int32_t, bool>> layer_of;

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;
    std::vector<moe_cache_lru::fill> cpu;

    struct tap_job {
        int32_t il = 0;
        int32_t ip = 0;
        int32_t expert = 0;
        std::string ttype;
        size_t es = 0;
        std::shared_ptr<std::vector<uint8_t>> buf;
    };
    struct tap_ent {
        std::shared_ptr<std::vector<uint8_t>> buf;
        bool ready = false;
    };
    bool tap_enabled = false;
    std::mutex tap_mu;
    std::unordered_map<uint64_t, tap_ent> tap_map;
    std::deque<tap_job> tap_q;
    std::condition_variable tap_cv;
    std::atomic<bool> tap_stop{false};
    std::vector<std::thread> tap_workers;
    std::atomic<size_t> tap_hits{0};
    std::atomic<size_t> tap_enq{0};
    std::unordered_map<uint64_t, int> tap_run;

    static uint64_t tap_key(int32_t il, int32_t ip, int32_t e) {
        return ((uint64_t)(uint32_t) il << 40) | ((uint64_t)(uint32_t)(ip & 0xffff) << 24) | (uint64_t)(uint32_t)(e & 0xffffff);
    }

    void tap_loop() {
        for (;;) {
            tap_job j;
            {
                std::unique_lock<std::mutex> lk(tap_mu);
                tap_cv.wait(lk, [this] { return tap_stop.load() || !tap_q.empty(); });
                if (tap_stop.load()) {
                    return;
                }
                j = std::move(tap_q.front());
                tap_q.pop_front();
            }
            j.buf->resize(j.es);
            bool ok = arena && arena->read_expert_direct(j.il, j.ttype.c_str(), j.expert, j.buf->data(), j.es) > 0;
            std::lock_guard<std::mutex> lk(tap_mu);
            auto it = tap_map.find(tap_key(j.il, j.ip, j.expert));
            if (it == tap_map.end()) {
                continue;
            }
            if (ok) {
                it->second.ready = true;
            } else {
                tap_map.erase(it);
            }
        }
    }

    void tap_shutdown() {
        if (tap_workers.empty()) {
            return;
        }
        tap_stop.store(true);
        tap_cv.notify_all();
        for (auto & t : tap_workers) {
            t.join();
        }
        tap_workers.clear();
        tap_map.clear();
    }

    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
         float cpu_share_, const char * profile_path_, const char * hot_path_) :
            n_expert_used(model.hparams.n_expert_used_max()),
            n_expert((int32_t) model.hparams.n_expert),
            cpu_share(std::clamp(cpu_share_, 0.0f, 1.0f)),
            layers(model.layers.size()) {
        debug_log = getenv("LLAMA_MOE_DEBUG") != nullptr;

        // LLAMA_MOE_CPU_SPLIT=<n|fraction>: run this many of the routed experts per
        // token on the CPU, straight out of host RAM, in the same graph as the GPU
        // experts. Fraction values (< 1) scale n_expert_used; integers are a count.
        // Default unset -> 0 -> the CPU branch is never materialised and the graph is
        // exactly the all-GPU staged path.
        if (const char * split = getenv("LLAMA_MOE_CPU_SPLIT")) {
            const float v = (float) atof(split);
            if (v > 0.0f) {
                const float share = v < 1.0f ? v : v / (float) std::max(1, n_expert_used);
                cpu_share = std::min(1.0f, share);
                LLAMA_LOG_INFO("%s: LLAMA_MOE_CPU_SPLIT=%s -> %.0f of %d routed experts per token on CPU (share %.3f)\n",
                        __func__, split, cpu_share * n_expert_used, n_expert_used, cpu_share);
            }
        }
        {
            // registered under the loader's GGUF block_count == n_layer_all, and
            // keyed by the per-layer expert geometry so a heterogeneous model
            // never resolves to an arena built for uniform expert widths.
            const int nl_all = (int) model.hparams.n_layer_all;
            const int ne_all = (int) model.hparams.n_expert;
            const uint64_t geom = moe_arena_geom_hash(nl_all, ne_all,
                    model.hparams.n_ff_exp_arr.data(), model.hparams.n_expert_used_arr.data());
            const auto it = g_moe_arenas.find(moe_arena_key{ nl_all, ne_all, geom });
            if (it != g_moe_arenas.end()) {
                arena = it->second.get();
            }
        }
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
                device d;
                d.backend = backends[i];
                d.buft    = bufts[i];
                d.buft_pinned = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backends[i]));
                devices.push_back(std::move(d));
            }
        }
        if (devices.empty() || size == 0) {
            return;
        }

        auto dev_of = [&](int32_t il) -> int32_t {
            ggml_backend_dev_t dev = model.dev_layer(il);
            for (size_t i = 0; i < devices.size(); ++i) {
                if (ggml_backend_get_device(devices[i].backend) == dev) {
                    return (int32_t) i;
                }
            }
            return 0;
        };

        for (int32_t il = 0; il < (int32_t) model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            // A layer may only be cached when every expert byte can be fetched from
            // a source that actually holds it: the RAM arena, the mapped shard, or a
            // real host tensor. Layers the arena skipped (it was full) have a lazy
            // tensor behind them, and filling a bank row from it copies uninitialised
            // memory into the GPU weights.
            {
                bool readable = true;
                for (const ggml_tensor * t : experts) {
                    const char * ttype = moe_get_tensor_type(t->name);
                    bool ok = false;
                    if (arena != nullptr && ttype != nullptr) {
                        ok = arena->combined_buffers.count(std::make_pair(il, std::string(ttype))) > 0 ||
                             arena->disk_meta.count(std::make_pair(il, std::string(ttype))) > 0;
                    }
                    if (!ok) {
                        const ggml_backend_buffer_t b = t->buffer;
                        ok = b != nullptr && ggml_backend_buffer_get_size(b) > 0 &&
                             t->data != nullptr && ggml_backend_buffer_is_host(b) &&
                             (arena == nullptr || !arena->layer_expert_bytes(il, ttype ? ttype : ""));
                    }
                    if (!ok) {
                        readable = false;
                        break;
                    }
                }
                if (!readable) {
                    continue;
                }
            }
            const int32_t id = dev_of(il);
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) {
                return g.id == id && llama_moe_cache_same_layout(g.ref, experts);
            });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                const size_t nbytes = ggml_nbytes(t);
                it->host_bytes += nbytes;
                devices[id].host_bytes += nbytes;
            }
        }
        if (groups.empty()) {
            return;
        }

        // One always-zero bank row is enough to neutralise every expert that the
        // CPU branch computes: mul_mat_id reads it as an all-zero weight row.
        const char * cpu_branch_env = getenv("LLAMA_MOE_CPU_BRANCH");
        const bool cpu_branch = cpu_branch_env == nullptr || atoi(cpu_branch_env) != 0;
        int32_t n_cpu_layers = 0;
        for (const auto & layer : model.layers) {
            if (cpu_branch && llama_moe_cache_layer_cpu_ready(layer)) {
                n_cpu_layers++;
            }
        }
        // Every CPU-computed expert needs its own all-zero bank row: the CUDA
        // fallback path matches ids to rows one-by-one and breaks on the first hit,
        // so two experts sharing a row makes the sorted id map short and trips an
        // assert. n_expert_used distinct lanes is exactly the per-token maximum.
        n_zero = (cpu_share > 0.0f || n_cpu_layers > 0) ? std::min<int32_t>(n_expert_used, 64) : 0;
        if (n_cpu_layers > 0) {
            LLAMA_LOG_INFO("%s: %d of %zu MoE layers expose full-size host expert tensors (%d CPU lanes)\n",
                    __func__, n_cpu_layers, model.layers.size(), n_zero);
        }
        cpu_share_q = (uint32_t) std::lround(cpu_share * 65536.0f);

        auto alloc_size = [&](const group & g, int32_t slots) -> size_t {
            if (slots == 0) {
                return 0;
            }
            const size_t align = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t total = 0;
            for (const ggml_tensor * t : g.ref) {
                total += GGML_PAD((size_t) (slots + n_zero + 1) * t->nb[2], align);
            }
            return total;
        };

        // Correctness floor: a VRAM cache smaller than one token's expert working
        // set cannot serve a token consistently (observed as corrupted logits /
        // degenerate repeated-token output). Disable it rather than thrash.
        {
            size_t token_ws = 0;
            for (const group & g : groups) {
                for (const ggml_tensor * t : g.ref) {
                    token_ws += (size_t) t->nb[2];
                }
            }
            token_ws *= (size_t) std::max<int64_t>(1, model.hparams.n_expert_used_max());
            const size_t min_bytes = std::max<size_t>((size_t) 2048 * 1024 * 1024, token_ws);
            if (size < min_bytes) {
                LLAMA_LOG_WARN("%s: VRAM MoE cache budget %.2f MiB is below the %.2f MiB minimum working set; cache disabled\n",
                    __func__, size / 1048576.0, min_bytes / 1048576.0);
                return;
            }
        }

        double split_sum = 0.0;
        for (size_t id = 0; id < devices.size(); ++id) {
            devices[id].split = devices[id].host_bytes > 0 ? (double) devices[id].host_bytes : 0.0;
            split_sum += devices[id].split;
        }
        if (split_sum <= 0.0) {
            return;
        }

        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            const size_t dev_budget = (size_t) ((double) size * (d.split / split_sum));
            for (group & g : groups) {
                if (g.id != (int32_t) id) {
                    continue;
                }
                const size_t grp_budget = (size_t) ((double) dev_budget * ((double) g.host_bytes / (double) d.host_bytes));
                size_t per_slot = 0;
                for (const ggml_tensor * t : g.ref) {
                    per_slot += t->nb[2];
                }
                if (per_slot == 0) {
                    continue;
                }
                const int32_t max_slots = (int32_t) std::min<int64_t>(
                    (int64_t) g.layers.size() * g.ref[0]->ne[2],
                    (int64_t) INT32_MAX - n_zero - 1);
                int32_t lo = 0;
                int32_t hi = max_slots;
                while (lo < hi) {
                    const int32_t mid = lo + (hi - lo + 1) / 2;
                    if (alloc_size(g, mid) <= grp_budget) {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }
                const int32_t min_slots = std::min<int32_t>(n_expert_used, max_slots);
                if (lo >= min_slots) {
                    g.n_slots = lo;
                }
            }
        }

        bool any_slots = false;
        for (const group & g : groups) {
            if (g.n_slots > 0) {
                any_slots = true;
            }
        }
        if (!any_slots) {
            groups.clear();
            return;
        }

        if (!profile_path.empty() && n_expert > 0) {
            counts.assign((size_t) layers.size() * (size_t) n_expert, 0);
            flushed_counts.assign((size_t) layers.size() * (size_t) n_expert, 0);
        }

        std::vector<bool> llm_ready(model.layers.size(), false);
        for (size_t il = 0; il < model.layers.size(); ++il) {
            llm_ready[il] = llama_moe_cache_layer_cpu_ready(model.layers[il]);
        }

        size_t n_host_tensors = 0;
        for (const group & g : groups) {
            if (g.n_slots > 0) {
                n_host_tensors += g.layers.size() * (n_zero > 0 ? 3 : 2);
            }
        }
        ctx_host.reset(ggml_init({ ggml_tensor_overhead() * n_host_tensors, nullptr, true }));
        ctx_views.reset(ggml_init({ ggml_tensor_overhead() * 4, nullptr, true }));

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t align_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t id = 0; id < devices.size(); ++id) {
            size_t n_t = 0;
            for (const group & g : groups) {
                if (g.id == (int32_t) id && g.n_slots > 0) {
                    n_t += g.ref.size() * (1 + g.layers.size());
                }
            }
            if (n_t > 0) {
                devices[id].ctx.reset(ggml_init({ ggml_tensor_overhead() * n_t, nullptr, true }));
            }
        }

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            g.banks.reserve(g.ref.size());
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + n_zero + 1);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            g.lru.init((int32_t) layers.size(), (int32_t) g.ref[0]->ne[2], g.n_slots, n_zero);
            {
                int32_t n_ready = 0;
                for (int32_t il : g.layers) {
                    if (layers[il].cpu_ready) {
                        n_ready++;
                    }
                }
                LLAMA_LOG_INFO("%s: group %zu: n_slots=%d n_zero=%d cpu_ready_layers=%d/%zu\n",
                        __func__, ig, g.n_slots, n_zero, n_ready, g.layers.size());
            }
            g.lru.counts = counts.empty() ? nullptr : counts.data();

            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                l.cpu_types.clear();
                for (const ggml_tensor * w : l.experts) {
                    const char * tt = moe_get_tensor_type(w->name);
                    if (tt != nullptr) {
                        l.cpu_types.emplace_back(tt);
                    }
                }
                l.cpu_ready = llm_ready[il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], bank->ne[2], bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                    if (arena != nullptr) {
                        const char * ttype = moe_get_tensor_type(l.experts[ip]->name);
                        if (ttype != nullptr) {
                            const auto k = std::make_pair(il, std::string(ttype));
                            if (arena->disk_meta.count(k) > 0) {
                                l.arena_has_disk = true;
                            }
                            if (arena->combined_buffers.count(k) > 0) {
                                l.arena_has_comb = true;
                            }
                        }
                    }
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = { il, cpu_branch && llm_ready[il] && n_zero > 0 };
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), align_host);
                // Expert -> CPU map for this ubatch. Every expert without a bank
                // row is computed on the CPU from its host-resident weights, so
                // the split is not limited to the experts cpu_share picked.
                l.cpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.cpu_map, "moe_cache.cpu_map-%d", il);
                buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_map), align_host);
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            if (!d.ctx) {
                continue;
            }
            d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
            if (!d.buf) {
                throw std::runtime_error("failed to allocate the MoE cache device buffers");
            }
            ggml_backend_buffer_clear(d.buf.get(), 0);
            d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            size_t max_es = 0;
            for (const group & gg : groups) {
                if (gg.id == (int32_t) id && !gg.ref.empty()) {
                    for (const ggml_tensor * t : gg.ref) {
                        max_es = std::max(max_es, (size_t) t->nb[2]);
                    }
                }
            }
            if (max_es > 0) {
                ggml_backend_buffer_type_t buft_p = d.buft_pinned ? d.buft_pinned : ggml_backend_cpu_buffer_type();
                const size_t stage_bytes = 2 * d.stage_experts * max_es;
                d.buf_stage.reset(ggml_backend_buft_alloc_buffer(buft_p, stage_bytes));
                if (!d.buf_stage) {
                    LLAMA_LOG_WARN("%s: failed to allocate %.2f MiB of pinned staging\n",
                                   __func__, stage_bytes / 1024.0 / 1024.0);
                } else {
                    ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
                    d.stage_ev[0] = ggml_backend_event_new(dev);
                    d.stage_ev[1] = ggml_backend_event_new(dev);
                }
            }
        }

        buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
        if (!buf_host) {
            throw std::runtime_error("failed to allocate the MoE cache host slot maps");
        }
        ggml_backend_buffer_clear(buf_host.get(), 0xff);
        buf_host_size = ggml_backend_buffer_get_size(buf_host.get());
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            for (int32_t il : g.layers) {
                if (layers[il].slot_map != nullptr) {
                    int32_t * slots = (int32_t *) layers[il].slot_map->data;
                    std::fill(slots, slots + layers[il].slot_map->ne[1], g.lru.empty);
                    g.lru.slot_map[il] = slots;
                }
            }
        }

        seed_from_profile();

        tap_enabled = getenv("LLAMA_MOE_TAP") != nullptr && atoi(getenv("LLAMA_MOE_TAP")) != 0;
        if (tap_enabled) {
            for (int t = 0; t < 2; ++t) {
                tap_workers.emplace_back([this] { tap_loop(); });
            }
            LLAMA_LOG_INFO("%s: token-ahead prefetch enabled (2 workers, persistent-miss filter)\n", __func__);
        }

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0 || !d.buf) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts, %.0f%% of misses on CPU (ClassLRU, pinned staging=%s)\n",
                __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0,
                100.0*cpu_share, d.buf_stage ? "on" : "off");
            for (const group & g : groups) {
                if (g.id == (int32_t) id && g.n_slots > 0) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots,
                        100.0 * g.n_slots / (g.layers.size() * g.ref[0]->ne[2]));
                }
            }
        }
    }

    ~impl() {
        tap_shutdown();
        flush_profile_now();
        log_stats();
        for (device & d : devices) {
            for (int i = 0; i < 2; ++i) {
                if (d.stage_ev[i] != nullptr) {
                    ggml_backend_event_free(d.stage_ev[i]);
                    d.stage_ev[i] = nullptr;
                }
            }
        }
    }

    void seed_from_profile() {
        if (hot_path.empty() || n_expert <= 0) {
            return;
        }
        if (getenv("LLAMA_MOE_NO_PIN") != nullptr) {
            return;
        }
        int32_t nl = 0;
        int32_t ne = 0;
        std::vector<uint32_t> hot;
        uint64_t selections = 0;
        const bool ok = moe_profile_read(hot_path, nl, ne, hot, selections);
        if (!ok || ne != n_expert || nl <= 0 || nl > (int32_t) layers.size()) {
            return;
        }
        if (nl < (int32_t) layers.size()) {
            hot.resize((size_t) layers.size() * (size_t) n_expert, 0);
        }

        size_t n_seeded = 0;
        struct seed_job { int32_t il; int32_t e; int32_t slot; };
        std::vector<seed_job> seed_jobs;
        for (group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            const int32_t free_pool = std::max<int32_t>(n_expert_used * 8, g.n_slots / 5);
            const int32_t budget    = std::max<int32_t>(0, g.n_slots - std::min<int32_t>(free_pool, g.n_slots));
            if (budget == 0) {
                continue;
            }
            std::vector<uint64_t> layer_tot(g.layers.size(), 0);
            uint64_t sum_tot = 0;
            for (size_t li = 0; li < g.layers.size(); ++li) {
                const int32_t il = g.layers[li];
                uint64_t s = 0;
                for (int32_t e = 0; e < n_expert; ++e) {
                    s += hot[(size_t) il * n_expert + e];
                }
                layer_tot[li] = s;
                sum_tot += s;
            }
            if (sum_tot == 0) {
                continue;
            }
            std::vector<int32_t> take(g.layers.size(), 0);
            int32_t assigned = 0;
            for (size_t li = 0; li < g.layers.size(); ++li) {
                take[li] = (int32_t) ((uint64_t) budget * layer_tot[li] / sum_tot);
                assigned += take[li];
            }
            for (size_t li = 0; li < g.layers.size() && assigned < budget; ++li) {
                if (layer_tot[li] > 0) {
                    take[li]++;
                    assigned++;
                }
            }
            int32_t slot = 0;
            for (size_t li = 0; li < g.layers.size(); ++li) {
                const int32_t il = g.layers[li];
                if (take[li] == 0 || layers[il].slot_map == nullptr) {
                    continue;
                }
                std::vector<std::pair<uint32_t, int32_t>> rank;
                rank.reserve(n_expert);
                for (int32_t e = 0; e < n_expert; ++e) {
                    const uint32_t c = hot[(size_t) il * n_expert + e];
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
                    g.lru.place(il, e, slot, 16.0f);
                    seed_jobs.push_back({ il, e, slot });
                    n_seeded++;
                    slot++;
                }
            }
        }

        if (seed_jobs.empty()) {
            return;
        }
        std::vector<uint8_t> tmp;
        for (const auto & j : seed_jobs) {
            const layer & l = layers[j.il];
            if (l.ig < 0) continue;
            group & g = groups[l.ig];
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                ggml_tensor * bank = g.banks[ip];
                const size_t es = w->nb[2];
                const size_t bank_es = bank->nb[2];
                const size_t cpy = std::min(es, bank_es);
                const char * ttype = moe_get_tensor_type(w->name);
                // This layer's own expert size, as captured by the arena. It must
                // match the tensor's nb[2]; otherwise the arena's stride is not
                // this layer's and the upload would read the wrong bytes.
                const size_t arena_es = (arena && ttype) ? arena->layer_expert_bytes(j.il, ttype) : 0;
                const bool   geom_ok  = arena_es == 0 || arena_es == es;
                bool uploaded = false;
                if (arena && ttype && geom_ok) {
                    if (void * comb = arena->get_combined_buffer(j.il, ttype)) {
                        arena->ensure_experts_loaded(j.il, ttype, &j.e, 1);
                        ggml_backend_tensor_set(bank, (const uint8_t *) comb + (size_t) j.e * es, (size_t) j.slot * bank_es, cpy);
                        uploaded = true;
                    } else if (arena->disk_meta.count(std::make_pair(j.il, std::string(ttype))) > 0) {
                        tmp.resize(es);
                        if (arena->read_expert_direct(j.il, ttype, j.e, tmp.data(), es) > 0) {
                            ggml_backend_tensor_set(bank, tmp.data(), (size_t) j.slot * bank_es, cpy);
                            uploaded = true;
                        }
                    }
                }
                if (!uploaded) {
                    ggml_backend_tensor_set(bank, (const uint8_t *) w->data + (size_t) j.e * es, (size_t) j.slot * bank_es, cpy);
                }
            }
        }
        LLAMA_LOG_INFO("%s: seeded %zu hot experts from '%s' (ClassLRU warm start)\n",
                       __func__, n_seeded, hot_path.c_str());
    }

    void flush_profile_now() {
        if (profile_path.empty() || counts.empty()) {
            return;
        }
        std::vector<uint32_t> delta;
        uint64_t tok_delta = 0;
        {
            std::lock_guard<std::mutex> lk(g_moe_profile_mu);
            delta.resize(counts.size());
            bool any = false;
            for (size_t i = 0; i < counts.size(); ++i) {
                const uint32_t d = counts[i] - flushed_counts[i];
                delta[i] = d;
                flushed_counts[i] = counts[i];
                if (d > 0) any = true;
            }
            tok_delta = profile_tokens - profile_tokens_at_flush;
            profile_tokens_at_flush = profile_tokens;
            if (!any) {
                return;
            }
        }
        moe_profile_write(profile_path, (int32_t) layers.size(), n_expert, n_expert_used, delta, tok_delta);
    }

    void maybe_flush_profile() {
        if (profile_path.empty() || counts.empty()) {
            return;
        }
        if ((++profile_flush_calls & 255) != 0) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_profile_flush).count();
        if (elapsed >= (int64_t) profile_flush_interval_s) {
            last_profile_flush = now;
            flush_profile_now();
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_used, bool cpu_part) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens * n_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        (void) cpu_part;
        return l.slot_map;
    }

    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].cpu_map == nullptr) {
            return nullptr;
        }
        const layer & l = layers[il];
        // A layer only reaches the cache when every expert tensor is host-resident
        // (see the group build), so the CPU branch reads valid weights either from
        // the mmap view or from the RAM arena's combined buffer.
        (void) n_tokens;
        if (!l.cpu_ready) {
            return nullptr;
        }
        return l.cpu_map;
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
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] >= g.n_slots) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] < g.n_slots && slots[e + n] == slots[e] + n) {
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

    void wait_stage_half(device & dev, ggml_backend_t backend, int f) {
        if (!dev.stage_outstd[f]) {
            return;
        }
        if (dev.stage_ev[f] != nullptr) {
            ggml_backend_event_synchronize(dev.stage_ev[f]);
        } else {
            ggml_backend_synchronize(backend);
        }
        dev.stage_outstd[f] = false;
    }

    void mark_stage_half(device & dev, ggml_backend_t backend, int f) {
        if (dev.stage_ev[f] != nullptr) {
            ggml_backend_event_record(dev.stage_ev[f], backend);
        }
        dev.stage_outstd[f] = true;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, ggml_backend_sched_t sched) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il      = it->second.first;
        const bool    has_cpu = it->second.second;
        layer & l = layers[il];
        group & g = groups[l.ig];
        device & dev = devices[g.id];

        GGML_ASSERT(backend == dev.backend);

        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        // Architectures whose router selects experts with a graph shape other than
        // RESHAPE(topk(router)) (e.g. nemotron_h_moe) cannot be served by this cache.
        // Fall back to the normal copy path instead of aborting.
        if (sel->op != GGML_OP_RESHAPE || !ggml_is_contiguous(sel)) {
            static bool warned_sel = false;
            if (!warned_sel) {
                warned_sel = true;
                LLAMA_LOG_WARN("%s: expert selection is %s (not a contiguous reshape); "
                               "VRAM MoE cache disabled for this model\n", __func__, ggml_op_name(sel->op));
            }
            return false;
        }

        const int32_t n_used = (int32_t) sel->src[0]->ne[0];

        ids.resize(ggml_nelements(sel));
        ggml_backend_t sel_be = backend;
        if (sched != nullptr) {
            ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, const_cast<ggml_tensor *>(sel));
            if (b != nullptr) {
                sel_be = b;
            }
        }
        ggml_backend_tensor_get_async(sel_be, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(sel_be);

        // Ram-resident experts are computed on the CPU; the share only caps how
        // many of the misses may leave the PCIe path (1.0 / unset = residency only).
        //
        // Batch gate. A CPU lane is a win only when it replaces a *blocking* PCIe
        // fill for one or two tokens: the CPU does the same FLOPs per expert either
        // way, while a fill is a fixed byte cost. Across a prompt of hundreds of
        // tokens the CPU expert matmuls dominate by orders of magnitude (480 draws
        // per token at ~19.7 MFLOP each), so a cached-but-not-resident expert must
        // stay on the fill path there and keep prompt processing on the GPU.
        const int64_t n_tok = sel->ne[1];
        static const int64_t cpu_batch_max = getenv("LLAMA_MOE_CPU_BATCH_MAX")
            ? (int64_t) atoll(getenv("LLAMA_MOE_CPU_BATCH_MAX")) : 4;
        const bool    cpu_batch_ok = n_tok <= cpu_batch_max;
        const uint32_t share_q = (has_cpu && cpu_batch_ok) ? cpu_share_q : 0;
        auto n_cpu = [&](size_t n_miss) {
            if (!has_cpu || !cpu_batch_ok || n_zero == 0) {
                return (size_t) 0;
            }
            if (share_q == 0 || share_q >= 65535) {
                return n_miss;
            }
            return (size_t) (((uint64_t) n_miss * share_q) >> 16);
        };
        auto cpu_ok = [&](int32_t e) {
            if (!has_cpu || !cpu_batch_ok || arena == nullptr || l.cpu_types.empty()) {
                return false;
            }
            return arena->expert_resident_all(il, e, l.cpu_types);
        };

        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), sel->src[0]->ne[1], n_used, n_zero, n_cpu, cpu_ok, fills, cpu, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        int32_t n_cpu_now = 0;
        if (l.cpu_map != nullptr) {
            // Residency is the single source of truth: a bank row means the GPU
            // branch computes it, and an expert that is host-resident but has no
            // bank row is computed on the CPU by the same graph. Anything else must
            // be filled from the model file, so it can never silently drop.
            const int32_t * slots = g.lru.slot_map[il];
            int32_t * cpu_map = (int32_t *) l.cpu_map->data;
            const int32_t n_map = (int32_t) l.cpu_map->ne[1];
            for (int32_t e = 0; e < n_map; ++e) {
                const bool in_bank = e < n_expert && slots[e] < g.n_slots;
                cpu_map[e] = (!in_bank && cpu_ok(e)) ? e : -1;
            }
            for (size_t i = 0; i < ids.size(); ++i) {
                if (cpu_map[ids[i]] >= 0) {
                    n_cpu_now++;
                }
            }
        }
        {
            static std::atomic<int> diag{0};
            if (debug_log && diag.fetch_add(1) < 400) {
                LLAMA_LOG_INFO("MOEDIAG il=%d n_tok=%lld used=%d hit=%zu fill=%zu cpu_share=%zu nonres=%d n_slots=%d\n",
                        il, (long long) sel->src[0]->ne[1], n_used, n_hit, fills.size(), cpu.size(), n_cpu_now, g.n_slots);
            }
        }
        if (!counts.empty()) {
            std::lock_guard<std::mutex> lk(g_moe_profile_mu);
            profile_tokens += ids.size() / (size_t) std::max(1, n_used);
        }

        if (arena && !fills.empty()) {
            std::vector<int> expert_ids;
            expert_ids.reserve(fills.size());
            for (const auto & fill : fills) {
                expert_ids.push_back(fill.expert);
            }
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                const char * tensor_type = moe_get_tensor_type(w->name);
                if (tensor_type) {
                    arena->ensure_experts_loaded(il, tensor_type, expert_ids.data(), (int) expert_ids.size());
                }
            }
        }

        if (tap_enabled && fills.size() <= 64) {
            {
                std::lock_guard<std::mutex> lk(tap_mu);
                if (tap_run.size() > 8192) {
                    tap_run.clear();
                }
                for (const auto & f : fills) {
                    tap_run[tap_key(il, 0, f.expert)] += 1;
                }
            }
        }
        if (tap_enabled && !fills.empty() && fills.size() <= 64) {
            constexpr size_t tap_cap = 384;
            std::lock_guard<std::mutex> lk(tap_mu);
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w = l.experts[ip];
                const char * tt = moe_get_tensor_type(w->name);
                if (!tt) {
                    continue;
                }
                const size_t es = w->nb[2];
                for (const auto & f : fills) {
                    if (tap_map.size() >= tap_cap) {
                        break;
                    }
                    if (tap_run[tap_key(il, 0, f.expert)] < 2) {
                        continue;
                    }
                    const uint64_t k = tap_key(il, (int32_t) ip, f.expert);
                    if (tap_map.count(k)) {
                        continue;
                    }
                    tap_ent ent;
                    ent.buf = std::make_shared<std::vector<uint8_t>>();
                    tap_job job;
                    job.il = il;
                    job.ip = (int32_t) ip;
                    job.expert = f.expert;
                    job.ttype = tt;
                    job.es = es;
                    job.buf = ent.buf;
                    tap_map.emplace(k, std::move(ent));
                    tap_q.push_back(std::move(job));
                    tap_enq.fetch_add(1);
                    tap_cv.notify_one();
                }
            }
        }

        size_t bytes = 0;
        const size_t K = dev.stage_experts;
        uint8_t * stage_base = dev.buf_stage ? (uint8_t *) ggml_backend_buffer_get_base(dev.buf_stage.get()) : nullptr;

        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            const size_t bank_es     = bank->nb[2];
            const size_t cpy         = std::min(expert_size, bank_es);

            const char * tensor_type = moe_get_tensor_type(w->name);
            const uint8_t * src_data = (const uint8_t *) w->data;
            // Per-layer geometry gate: the arena may only serve this layer if the
            // stride it captured for this very layer equals the tensor's nb[2].
            // Anything else would copy a neighbouring expert's bytes.
            const size_t arena_es = (tensor_type && arena) ? arena->layer_expert_bytes(il, tensor_type) : 0;
            const bool   geom_ok  = arena_es == 0 || arena_es == expert_size;
            bool have_combined = false;
            bool has_disk = false;
            if (tensor_type && arena && geom_ok) {
                if (void * combined = arena->get_combined_buffer(il, tensor_type)) {
                    src_data = (const uint8_t *) combined;
                    have_combined = true;
                } else {
                    has_disk = arena->disk_meta.count(std::make_pair(il, std::string(tensor_type))) > 0;
                }
            } else if (tensor_type && arena && !geom_ok) {
                static std::mutex gm_mu;
                static std::set<std::pair<int, std::string>> gm_note;
                std::lock_guard<std::mutex> lk(gm_mu);
                if (gm_note.insert(std::make_pair(il, std::string(tensor_type))).second) {
                    LLAMA_LOG_WARN("%s: L%d %s arena stride %zu != tensor stride %zu; using mmap copies\n",
                            __func__, il, tensor_type, arena_es, expert_size);
                }
            }

            if (has_disk && stage_base != nullptr) {
                // opt-loop A/B knob: default halves the cores (12 of 24 here);
                // LLAMA_MOE_STAGE_WORKERS overrides for full-core read depth
                int n_workers_env = -1;
                if (const char * e = getenv("LLAMA_MOE_STAGE_WORKERS")) {
                    n_workers_env = atoi(e);
                }
                const int n_workers = n_workers_env > 0
                    ? (int) std::min<size_t>((size_t) n_workers_env, fills.size())
                    : (int) std::min<size_t>(K, std::max(1u, std::thread::hardware_concurrency() / 2));
                size_t i = 0;
                while (i < fills.size()) {
                    const size_t c0 = i;
                    const size_t ci = std::min(K, fills.size() - i);
                    i += ci;
                    const int f = dev.stage_flip;
                    dev.stage_flip ^= 1;
                    wait_stage_half(dev, backend, f);

                    std::atomic<size_t> next(c0);
                    std::atomic<bool> any_fail(false);
                    auto worker = [&]() {
                        for (;;) {
                            const size_t k = next.fetch_add(1);
                            if (k >= c0 + ci) {
                                return;
                            }
                            void * dstp = stage_base + (size_t) f * K * expert_size + (k - c0) * expert_size;
                            std::shared_ptr<std::vector<uint8_t>> sbuf;
                            if (tap_enabled) {
                                std::lock_guard<std::mutex> lk(tap_mu);
                                auto it = tap_map.find(tap_key(il, (int32_t) ip, fills[k].expert));
                                if (it != tap_map.end() && it->second.ready) {
                                    sbuf = it->second.buf;
                                    tap_map.erase(it);
                                }
                            }
                            if (sbuf) {
                                tap_hits.fetch_add(1);
                                memcpy(dstp, sbuf->data(), cpy);
                            } else if (arena->read_expert_direct(il, tensor_type, fills[k].expert, dstp, expert_size) == 0) {
                                any_fail.store(true);
                                return;
                            }
                        }
                    };
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
                        for (size_t k = c0; k < c0 + ci; ++k) {
                            if (fills[k].slot < 0 || fills[k].slot >= g.n_slots) continue;
                            ggml_backend_tensor_set_async(backend, bank,
                                src_data + (size_t) fills[k].expert * expert_size,
                                (size_t) fills[k].slot * bank_es, cpy);
                            bytes += cpy;
                        }
                        continue;
                    }
                    for (size_t k = c0; k < c0 + ci; ++k) {
                        if (fills[k].slot < 0 || fills[k].slot >= g.n_slots) continue;
                        ggml_backend_tensor_set_async(backend, bank,
                            stage_base + (size_t) f * K * expert_size + (k - c0) * expert_size,
                            (size_t) fills[k].slot * bank_es, cpy);
                        bytes += cpy;
                    }
                    mark_stage_half(dev, backend, f);
                }
                continue;
            }

            const bool src_is_pinned = !have_combined && dev.buft_pinned != nullptr &&
                w->buffer != nullptr && ggml_backend_buffer_get_type(w->buffer) == dev.buft_pinned;

            if (!src_is_pinned && stage_base != nullptr && !fills.empty()) {
                size_t i = 0;
                while (i < fills.size()) {
                    const size_t c0 = i;
                    const size_t ci = std::min(K, fills.size() - i);
                    i += ci;
                    const int f = dev.stage_flip;
                    dev.stage_flip ^= 1;
                    wait_stage_half(dev, backend, f);

                    uint8_t * half = stage_base + (size_t) f * K * expert_size;
                    for (size_t k = 0; k < ci; ++k) {
                        memcpy(half + k * expert_size,
                               src_data + (size_t) fills[c0 + k].expert * expert_size,
                               cpy);
                    }
                    for (size_t k = 0; k < ci; ) {
                        size_t n = 1;
                        if (cpy == expert_size && bank_es == expert_size) {
                            while (k + n < ci && fills[c0 + k + n].slot == fills[c0 + k].slot + (int32_t) n) {
                                n++;
                            }
                        }
                        if (fills[c0 + k].slot >= 0 && fills[c0 + k].slot + (int32_t) n <= g.n_slots) {
                            ggml_backend_tensor_set_async(backend, bank,
                                half + k * expert_size,
                                (size_t) fills[c0 + k].slot * bank_es,
                                n * cpy);
                            bytes += n * cpy;
                        }
                        k += n;
                    }
                    mark_stage_half(dev, backend, f);
                }
                continue;
            }

            for (size_t i = 0; i < fills.size(); ) {
                size_t n = 1;
                while (i + n < fills.size() &&
                       fills[i + n].expert == fills[i].expert + (int32_t) n &&
                       fills[i + n].slot   == fills[i].slot   + (int32_t) n) {
                    n++;
                }
                if (fills[i].slot >= 0 && fills[i].slot + (int32_t) n <= g.n_slots) {
                    ggml_backend_tensor_set_async(backend, bank,
                        src_data + (size_t) fills[i].expert * expert_size,
                        (size_t) fills[i].slot * bank_es,
                        std::min(cpy * n, bank_es * n));
                    bytes += n * cpy;
                }
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8 * n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size() + cpu.size();
        st.cpu    += cpu.size();
        st.bytes  += bytes;

        ggml_backend_tensor_set_async(backend, dst, l.slot_map->data, 0, ggml_nbytes(l.slot_map));

        maybe_flush_profile();
        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu (%zu on CPU), hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, st.cpu, 100.0 * st.hits / n, st.bytes / 1024.0 / 1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n",
                stats_copy.hits, stats_copy.bytes / 1024.0 / 1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
        float cpu_share, const char * profile_path, const char * hot_path) :
    pimpl(new impl(model, backends, bufts, size, cpu_share, profile_path, hot_path)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

llama_moe_cache_lookup llama_moe_cache::build_lookup(const llm_graph_context & g, const llama_moe_cache_layer & layer, ggml_tensor * selected_experts) const {
    const int32_t il = layer.il;
    ggml_context * ctx0 = g.ctx0;

    bool cpu_part = !layer.weight_before_ffn && g.loras->empty();
    for (ggml_tensor * w : { layer.up_exps, layer.gate_exps, layer.down_exps, layer.gate_up_exps }) {
        cpu_part = cpu_part && !(w != nullptr && g.hdmd != nullptr && g.hdmd->rot.count(w) > 0);
    }

    ggml_tensor * cpu_map  = cpu_part ? pimpl->get_cpu_map(il, selected_experts->ne[1]) : nullptr;
    ggml_tensor * slot_map = pimpl->get_slot_map(il, selected_experts->ne[1], selected_experts->ne[0], cpu_map != nullptr);
    if (slot_map == nullptr) {
        return {};
    }
    for (ggml_tensor * w : { layer.up_exps, layer.gate_exps, layer.down_exps, layer.gate_up_exps }) {
        if (w == nullptr) {
            continue;
        }
        ggml_tensor * experts = pimpl->get_experts(w);
        if (experts == nullptr) {
            return {};
        }
        ggml_tensor * b = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, experts->ne[0], w == layer.down_exps ? selected_experts->ne[0] : 1, selected_experts->ne[1]);
        if (!ggml_backend_supports_op(pimpl->backend(il), ggml_mul_mat_id(ctx0, experts, b, selected_experts))) {
            return {};
        }
    }

    ggml_tensor * ids = selected_experts;
    if (!ggml_is_contiguous(ids)) {
        ids = ggml_cont(ctx0, ids);
    }
    ids = ggml_reshape_1d(ctx0, ids, ggml_nelements(ids));

    ggml_tensor * slots = ggml_get_rows(ctx0, slot_map, ids);
    if (!ggml_backend_supports_op(pimpl->backend(il), slots)) {
        return {};
    }
    ggml_backend_sched_set_tensor_backend(g.sched, slots, pimpl->backend(il));
    g.cb(slots, "ffn_moe_slots", il);
    ggml_build_forward_expand(g.gf, slots);

    ggml_tensor * cpu_ids = nullptr;
    if (cpu_map != nullptr) {
        cpu_ids = ggml_get_rows(ctx0, cpu_map, ids);
        cpu_ids = ggml_reshape_2d(ctx0, cpu_ids, selected_experts->ne[0], selected_experts->ne[1]);
    }

    return { il, ggml_reshape_2d(ctx0, slots, selected_experts->ne[0], selected_experts->ne[1]), cpu_ids };
}

ggml_tensor * llama_moe_cache::build_experts(const llm_graph_context & g, const llama_moe_cache_lookup & lookup,
        ggml_tensor * cur, ggml_tensor * selected_experts, llama_moe_cache_build_experts_t build, void * user_data) const {
    GGML_ASSERT(lookup.slots != nullptr);

    ggml_tensor * experts = build(cur, selected_experts, lookup.slots, true, user_data);
    if (lookup.cpu_ids != nullptr) {
        ggml_build_forward_expand(g.gf, experts);
        experts = ggml_add(g.ctx0, experts, build(cur, selected_experts, lookup.cpu_ids, false, user_data));
        g.cb(experts, "ffn_moe_down_merged", lookup.il);
    }
    return experts;
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
