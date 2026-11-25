#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;
struct llm_graph_context;

struct llama_moe_cache_layer {
    int32_t       il;
    ggml_tensor * up_exps;
    ggml_tensor * gate_exps;
    ggml_tensor * down_exps;
    ggml_tensor * gate_up_exps;
    bool          weight_before_ffn;
};

struct llama_moe_cache_lookup {
    int32_t       il      = -1;
    ggml_tensor * slots   = nullptr;
    ggml_tensor * cpu_ids = nullptr;
};

using llama_moe_cache_build_experts_t = ggml_tensor * (*)(ggml_tensor * cur, ggml_tensor * ids, ggml_tensor * mm_ids, bool cached, void * user_data);

class llama_moe_cache {
public:
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
                    float cpu_share = 0.0f, const char * profile_path = nullptr, const char * hot_path = nullptr);
    ~llama_moe_cache();

    ggml_backend_t backend(int32_t il) const;
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    llama_moe_cache_lookup build_lookup(const llm_graph_context & g, const llama_moe_cache_layer & layer, ggml_tensor * selected_experts) const;
    ggml_tensor * build_experts(const llm_graph_context & g, const llama_moe_cache_lookup & lookup,
            ggml_tensor * cur, ggml_tensor * selected_experts, llama_moe_cache_build_experts_t build, void * user_data) const;

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph, ggml_backend_sched_t sched = nullptr);
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
