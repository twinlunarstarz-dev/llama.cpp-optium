from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    s = p.read_text()
    if old not in s:
        raise SystemExit(f'anchor not found in {path}: {old[:120]!r}')
    p.write_text(s.replace(old, new, 1))


# Remove the fixed dense/expert cache percentages. Residency now uses the whole
# measured post-reservation weight window and globally evicts the coldest completed
# entry until the current allocation passes both the ownership and live-memory guards.
replace_once(
    'ggml/src/ggml-backend.cpp',
    '''static bool ggml_backend_sched_make_resident_space(\n        ggml_backend_sched_t sched, int backend_id, size_t request, bool expert_tier) {\n    const size_t window = sched->weight_window_limit[backend_id];\n    // Keep dense/recurrent weights and compact expert slabs in independent tiers. A\n    // scan-resistant dense tier is deliberately not evicted by a one-use cyclic\n    // stream: once admitted, its stable subset produces hits on every later token.\n    // Expert slabs retain LFU/LRU replacement because their active set is dynamic.\n    const bool partitioned = window >= (size_t) 1024 * 1024 * 1024;\n    // Reserve half of the window for the currently executing split. DeepSeek V4\n    // has individual sequential allocations above 2 GiB on a roughly 4 GiB\n    // CUDA0 window, so filling most of the window with cache makes forward\n    // progress impossible even though cache admission itself succeeded.\n    const size_t tier_limit = !partitioned ? window :\n        (expert_tier ? window / 10 : (window * 4) / 10);\n    while (true) {\n        size_t tier_owned = 0;\n        for (const auto & entry : *sched->residents) {\n            const auto & resident = entry.second;\n            if (resident.backend_id == backend_id && resident.expert_tier == expert_tier) {\n                tier_owned += resident.allocation_size;\n            }\n        }\n        if (request <= tier_limit && tier_owned <= tier_limit - request) {\n            bool unknown = false;\n            bool live_rejected = false;\n            return ggml_backend_sched_weight_window_admit(sched, backend_id, request, &unknown, &live_rejected);\n        }\n        if (partitioned && !expert_tier) {\n            return false;\n        }\n        auto victim = sched->residents->end();\n        for (auto it = sched->residents->begin(); it != sched->residents->end(); ++it) {\n            if (it->second.backend_id == backend_id && it->second.expert_tier == expert_tier && !it->second.executing &&\n                    (victim == sched->residents->end() || it->second.frequency < victim->second.frequency ||\n                     (it->second.frequency == victim->second.frequency &&\n                      it->second.completed_use < victim->second.completed_use))) {\n                victim = it;\n            }\n        }\n        if (victim == sched->residents->end()) {\n            return false;\n        }\n        ggml_backend_sched_evict_resident(sched, victim);\n    }\n}\n''',
    '''static bool ggml_backend_sched_make_resident_space(\n        ggml_backend_sched_t sched, int backend_id, size_t request) {\n    const size_t window = sched->weight_window_limit[backend_id];\n    if (!sched->weight_window_configured[backend_id] || !sched->weight_window_memory_valid[backend_id] ||\n            request == 0 || request > window) {\n        return false;\n    }\n    while (true) {\n        bool unknown = false;\n        bool live_rejected = false;\n        if (ggml_backend_sched_weight_window_admit(\n                sched, backend_id, request, &unknown, &live_rejected)) {\n            return true;\n        }\n        if (unknown) {\n            return false;\n        }\n\n        // Dense weights and expert slabs share one global budget. Frequency first\n        // makes repeated dense weights scan-resistant, while completed-use breaks\n        // ties in LRU order for the changing MoE active set. Executing entries are\n        // part of the current working set and are never eviction candidates.\n        auto victim = sched->residents->end();\n        for (auto it = sched->residents->begin(); it != sched->residents->end(); ++it) {\n            if (it->second.backend_id != backend_id || it->second.executing) {\n                continue;\n            }\n            if (victim == sched->residents->end() || it->second.frequency < victim->second.frequency ||\n                    (it->second.frequency == victim->second.frequency &&\n                     it->second.completed_use < victim->second.completed_use)) {\n                victim = it;\n            }\n        }\n        if (victim == sched->residents->end()) {\n            return false;\n        }\n        ggml_backend_sched_evict_resident(sched, victim);\n    }\n}\n''')
replace_once(
    'ggml/src/ggml-backend.cpp',
    '''            const bool resident_admitted = cache_eligible && alloc_size > 0 &&\n                ggml_backend_sched_make_resident_space(sched, split_backend_id, alloc_size, expert_tier);\n''',
    '''            const bool resident_admitted = cache_eligible && alloc_size > 0 &&\n                ggml_backend_sched_make_resident_space(sched, split_backend_id, alloc_size);\n''')

# Coalesce compact expert misses when both source expert ids and destination slots
# are adjacent. This converts many small storage/H2D operations into one contiguous
# transfer without changing the compact slot layout.
replace_once(
    'ggml/src/ggml-backend.cpp',
    '''                        for (size_t compact_id = 0; compact_id < compact_experts.size(); ++compact_id) {\n                            if (!compact_misses[compact_id]) {\n                                continue;\n                            }\n                            const size_t src_offset = (size_t) compact_experts[compact_id] * expert_size;\n                            const size_t dst_offset = (size_t) compact_slots[compact_id] * expert_size;\n                            // Every compact slot contains a complete expert, so there is no unused\n                            // inter-slot padding to initialize. Extending this copy according to the\n                            // destination slot can read past a final source expert and overwrite the\n                            // following resident slot when source and destination orders differ.\n                            ggml_backend_sched_weight_upload_chunked(sched, split_backend, split_backend_id,\n                                input_cpy, (const uint8_t *) input->data + src_offset, dst_offset,\n                                expert_size, split->input_transient[input_id]);\n                        }\n''',
    '''                        for (size_t compact_id = 0; compact_id < compact_experts.size();) {\n                            if (!compact_misses[compact_id]) {\n                                ++compact_id;\n                                continue;\n                            }\n                            size_t last = compact_id;\n                            while (last + 1 < compact_experts.size() && compact_misses[last + 1] &&\n                                    compact_experts[last + 1] == compact_experts[last] + 1 &&\n                                    compact_slots[last + 1] == compact_slots[last] + 1) {\n                                ++last;\n                            }\n                            const size_t src_offset = (size_t) compact_experts[compact_id] * expert_size;\n                            const size_t dst_offset = (size_t) compact_slots[compact_id] * expert_size;\n                            const size_t copy_size = (last - compact_id + 1) * expert_size;\n                            ggml_backend_sched_weight_upload_chunked(sched, split_backend, split_backend_id,\n                                input_cpy, (const uint8_t *) input->data + src_offset, dst_offset,\n                                copy_size, split->input_transient[input_id]);\n                            compact_id = last + 1;\n                        }\n''')

# Conservative hybrid overflow: before pass 4 propagates backend assignments to
# sources, move a routed-MoE op to another GPU with sufficient sequential budget,
# or to CPU as a final fallback, when its worst-case active expert set cannot fit
# the selected GPU's current weight window. This is only active in forced sequential
# offload and leaves normal scheduling unchanged.
p = Path('ggml/src/ggml-backend.cpp')
s = p.read_text()
anchor = '    // pass 4: assign backends to remaining src from dst and view_src\n'
insert = r'''    if (sched->force_weight_offload) {
        const int cpu_backend_id = sched->n_backends - 1;
        for (int i = 0; i < graph->n_nodes; ++i) {
            struct ggml_tensor * node = graph->nodes[i];
            if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] == NULL || node->src[2] == NULL) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id < 0 || *node_backend_id == cpu_backend_id) {
                continue;
            }
            const ggml_tensor * weights = node->src[0];
            const ggml_tensor * ids = node->src[2];
            if (weights->buffer == NULL || !ggml_backend_buffer_is_host(weights->buffer) ||
                    weights->ne[2] <= 1 || weights->nb[2] == 0 || ids->ne[0] <= 0 || ids->ne[1] <= 0) {
                continue;
            }
            const uint64_t n_ids = (uint64_t) ids->ne[0] > UINT64_MAX / (uint64_t) ids->ne[1] ?
                UINT64_MAX : (uint64_t) ids->ne[0] * (uint64_t) ids->ne[1];
            const uint64_t n_active = std::min<uint64_t>((uint64_t) weights->ne[2], n_ids);
            const uint64_t estimate = n_active > UINT64_MAX / weights->nb[2] ?
                UINT64_MAX : n_active * weights->nb[2];
            auto backend_limit = [&](int backend_id) -> size_t {
                return sched->weight_window_configured[backend_id] ?
                    sched->weight_window_limit[backend_id] : sched->max_weight_bytes_per_split[backend_id];
            };
            const size_t selected_limit = backend_limit(*node_backend_id);
            if (selected_limit == 0 || estimate <= selected_limit) {
                continue;
            }

            int overflow_backend = -1;
            for (int candidate = 0; candidate < cpu_backend_id; ++candidate) {
                if (candidate == *node_backend_id || !ggml_backend_supports_op(sched->backends[candidate], node)) {
                    continue;
                }
                const size_t limit = backend_limit(candidate);
                if (limit > 0 && estimate <= limit) {
                    overflow_backend = candidate;
                    break;
                }
            }
            if (overflow_backend < 0 && ggml_backend_supports_op(sched->backends[cpu_backend_id], node)) {
                overflow_backend = cpu_backend_id;
            }
            if (overflow_backend >= 0) {
                GGML_LOG_DEBUG("sequential MoE overflow: %s estimated active weights=%" PRIu64
                    " bytes exceed backend %s limit=%zu; routing to %s\n",
                    node->name, estimate, ggml_backend_name(sched->backends[*node_backend_id]), selected_limit,
                    ggml_backend_name(sched->backends[overflow_backend]));
                *node_backend_id = overflow_backend;
                SET_CAUSE(node, "3.moe-overflow");
            }
        }
    }

'''
if anchor not in s:
    raise SystemExit('scheduler pass4 anchor not found')
s = s.replace(anchor, insert + anchor, 1)
p.write_text(s)

# Avoid cast-qual warnings introduced by the pinned staging path.
p = Path('ggml/src/ggml-backend.cpp')
s = p.read_text()
s = s.replace('const uint8_t ** data) {', 'uint8_t ** data) {', 2)
s = s.replace('*data = (const uint8_t *) ggml_backend_buffer_get_base(staging.buffers[slot]) + task.data_offset;',
              '*data = (uint8_t *) ggml_backend_buffer_get_base(staging.buffers[slot]) + task.data_offset;', 1)
s = s.replace('*data = (const uint8_t *) base + data_offset;', '*data = (uint8_t *) base + data_offset;', 1)
s = s.replace('const uint8_t * staged_data = NULL;', 'uint8_t * staged_data = NULL;', 1)
s = s.replace('staged_data = (const uint8_t *) ggml_backend_buffer_get_base(staging.buffers[slot]);\n                    memcpy((void *) staged_data, src_bytes + copied, chunk);',
              'staged_data = (uint8_t *) ggml_backend_buffer_get_base(staging.buffers[slot]);\n                    memcpy(staged_data, src_bytes + copied, chunk);', 1)
p.write_text(s)

p = Path('src/llama-context.cpp')
s = p.read_text()
s = s.replace('(void *) &model);', 'const_cast<llama_model *>(&model));')
p.write_text(s)

# Sequential-only resource fallback: if context reservation fails, retry with a
# smaller physical microbatch rather than giving up immediately. The successful
# value is reflected back into common_params so server diagnostics are truthful.
replace_once(
    'common/common.cpp',
    '''    llama_context * lctx = llama_init_from_model(model, cparams);\n    if (lctx == NULL) {\n        COM_ERR("failed to create context with model '%s'\\n", params.model.path.c_str());\n        return;\n    }\n''',
    '''    llama_context * lctx = nullptr;\n    while (true) {\n        lctx = llama_init_from_model(model, cparams);\n        if (lctx != nullptr || !params.sequential_load || cparams.n_ubatch <= 32) {\n            break;\n        }\n        const uint32_t next_ubatch = std::max<uint32_t>(32, cparams.n_ubatch / 2);\n        if (next_ubatch >= cparams.n_ubatch) {\n            break;\n        }\n        COM_WRN("sequential context allocation failed with ubatch=%u; retrying with ubatch=%u\\n",\n            cparams.n_ubatch, next_ubatch);\n        cparams.n_ubatch = next_ubatch;\n        params.n_ubatch = (int32_t) next_ubatch;\n    }\n    if (lctx == NULL) {\n        COM_ERR("failed to create context with model '%s'\\n", params.model.path.c_str());\n        return;\n    }\n''')

# Repair the unrelated full-test build regression exposed by the validation run:
# test-chat directly includes server-common.h, which directly includes mtmd.h.
replace_once(
    'tests/CMakeLists.txt',
    '''    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)\n''',
    '''    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)\n''')
