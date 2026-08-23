from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    s = p.read_text()
    if old not in s:
        raise SystemExit(f"anchor not found in {path}: {old[:120]!r}")
    p.write_text(s.replace(old, new, 1))


# -----------------------------------------------------------------------------
# 1. Register only the retained mmap span with the accelerator host-registration
#    hook.  Sequential/O_DIRECT mode is deliberately excluded: its defining
#    property is that the whole model need not be resident in RAM.
# -----------------------------------------------------------------------------
replace_once(
    "src/llama-mmap.h",
    """    void unmap_fragment(size_t first, size_t last);\n\n    static const bool SUPPORTED;\n\nprivate:\n    struct impl;\n    std::unique_ptr<impl> pimpl;\n};\n""",
    """    void unmap_fragment(size_t first, size_t last);\n\n    // Register the retained mmap pages as accelerator-accessible host memory.\n    // The registration is released before the mapping is destroyed.\n    // Returns the registered byte count, or 0 when registration is unsupported.\n    size_t register_host(size_t first, size_t last, bool (*reg_fn)(void *, size_t), void (*unreg_fn)(void *));\n\n    static const bool SUPPORTED;\n\nprivate:\n    struct impl;\n    std::unique_ptr<impl> pimpl;\n\n    void * host_reg_addr = nullptr;\n    void (*host_unreg_fn)(void *) = nullptr;\n};\n""")

replace_once(
    "src/llama-mmap.cpp",
    """llama_mmap::llama_mmap(struct llama_file * file, size_t prefetch, bool numa) : pimpl(std::make_unique<impl>(file, prefetch, numa)) {}\nllama_mmap::~llama_mmap() = default;\n\nsize_t llama_mmap::size() const { return pimpl->size; }\nvoid * llama_mmap::addr() const { return pimpl->addr; }\n\nvoid llama_mmap::unmap_fragment(size_t first, size_t last) { pimpl->unmap_fragment(first, last); }\n\n#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)\n""",
    """llama_mmap::llama_mmap(struct llama_file * file, size_t prefetch, bool numa) : pimpl(std::make_unique<impl>(file, prefetch, numa)) {}\n\nllama_mmap::~llama_mmap() {\n    // The backend registration must be released while the virtual mapping still\n    // exists. pimpl is destroyed after this destructor body returns.\n    if (host_reg_addr != nullptr && host_unreg_fn != nullptr) {\n        host_unreg_fn(host_reg_addr);\n    }\n}\n\nsize_t llama_mmap::size() const { return pimpl->size; }\nvoid * llama_mmap::addr() const { return pimpl->addr; }\n\nvoid llama_mmap::unmap_fragment(size_t first, size_t last) { pimpl->unmap_fragment(first, last); }\n\nsize_t llama_mmap::register_host(\n        size_t first, size_t last, bool (*reg_fn)(void *, size_t), void (*unreg_fn)(void *)) {\n#ifdef _POSIX_MAPPED_FILES\n    if (host_reg_addr != nullptr || reg_fn == nullptr || unreg_fn == nullptr || last <= first || first >= pimpl->size) {\n        return 0;\n    }\n\n    last = std::min(last, pimpl->size);\n    const size_t page_size = (size_t) sysconf(_SC_PAGESIZE);\n    if (page_size == 0 || (page_size & (page_size - 1)) != 0) {\n        return 0;\n    }\n    first &= ~(page_size - 1);\n    last = (last + page_size - 1) & ~(page_size - 1);\n\n    void * reg_addr = (uint8_t *) pimpl->addr + first;\n    if (!reg_fn(reg_addr, last - first)) {\n        return 0;\n    }\n\n    host_reg_addr = reg_addr;\n    host_unreg_fn = unreg_fn;\n    return last - first;\n#else\n    GGML_UNUSED(first);\n    GGML_UNUSED(last);\n    GGML_UNUSED(reg_fn);\n    GGML_UNUSED(unreg_fn);\n    return 0;\n#endif\n}\n\n#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)\n""")

p = Path("src/llama-model-loader.cpp")
s = p.read_text()
old = """        // unmap offloaded tensors and metadata\n        if (use_mmap) {\n            for (uint32_t idx = 0; idx < mappings.size(); idx++) {\n                const auto & mmap_used = mmaps_used.at(idx);\n                auto & mapping = mappings.at(idx);\n                if (sequential_load) {\n                    // sequential mode: keep tensor data mapped for on-demand disk paging;\n                    // only unmap the metadata header (before first tensor)\n                    mapping->unmap_fragment(0, mmap_used.first);\n                } else {\n                    mapping->unmap_fragment(0, mmap_used.first);\n                    if (mmap_used.second != 0) {\n                        mapping->unmap_fragment(mmap_used.second, mapping->size());\n                    }\n                }\n            }\n        }\n"""
new = """        // unmap offloaded tensors and metadata\n        if (use_mmap) {\n            bool (*reg_fn)(void *, size_t) = nullptr;\n            void (*unreg_fn)(void *) = nullptr;\n            const char * register_host_env = getenv(\"GGML_CUDA_REGISTER_HOST\");\n            const bool register_host = !sequential_load && register_host_env != nullptr && atoi(register_host_env) != 0;\n\n            if (register_host) {\n                // Resolve the generic host-registration hook from the first backend\n                // that exposes it. CUDA's implementation uses cudaHostRegister.\n                for (size_t i = 0; i < ggml_backend_dev_count() && reg_fn == nullptr; ++i) {\n                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_dev_get(i));\n                    reg_fn = (bool (*)(void *, size_t)) ggml_backend_reg_get_proc_address(\n                        reg, \"ggml_backend_register_host_buffer\");\n                    unreg_fn = (void (*)(void *)) ggml_backend_reg_get_proc_address(\n                        reg, \"ggml_backend_unregister_host_buffer\");\n                    if (reg_fn == nullptr || unreg_fn == nullptr) {\n                        reg_fn = nullptr;\n                        unreg_fn = nullptr;\n                    }\n                }\n            }\n\n            for (uint32_t idx = 0; idx < mappings.size(); idx++) {\n                const auto & mmap_used = mmaps_used.at(idx);\n                auto & mapping = mappings.at(idx);\n                if (sequential_load) {\n                    // sequential mode: keep tensor data mapped for on-demand disk paging;\n                    // only unmap the metadata header (before first tensor). Never pin\n                    // this full span: sequential mode must work when the model exceeds RAM.\n                    mapping->unmap_fragment(0, mmap_used.first);\n                } else {\n                    mapping->unmap_fragment(0, mmap_used.first);\n                    if (mmap_used.second != 0) {\n                        mapping->unmap_fragment(mmap_used.second, mapping->size());\n                    }\n                    if (mmap_used.second > mmap_used.first && reg_fn != nullptr && unreg_fn != nullptr) {\n                        const size_t n_registered = mapping->register_host(\n                            mmap_used.first, mmap_used.second, reg_fn, unreg_fn);\n                        if (n_registered > 0) {\n                            LLAMA_LOG_INFO(\"%s: registered %.2f MiB of mmap-backed CPU weights for direct H2D DMA\\n\",\n                                __func__, n_registered / 1024.0 / 1024.0);\n                        }\n                    }\n                }\n            }\n        }\n"""
if old not in s:
    raise SystemExit("model-loader final mmap cleanup anchor not found")
p.write_text(s.replace(old, new, 1))


# -----------------------------------------------------------------------------
# 2. Dense-routing MoE prefetch.  Reuse optium's manually-owned transient
#    buffers and secondary per-device backend instead of importing the older
#    fork's standalone slot allocator.  This gives the same lifetime guarantee
#    while composing with the newer residency/window accounting.
# -----------------------------------------------------------------------------
p = Path("ggml/src/ggml-backend.cpp")
s = p.read_text()

s = s.replace(
    """    bool input_transient[GGML_SCHED_MAX_SPLIT_INPUTS];\n    ggml_backend_buffer_t transient_buffers[GGML_SCHED_MAX_SPLIT_INPUTS];\n""",
    """    bool input_transient[GGML_SCHED_MAX_SPLIT_INPUTS];\n    bool input_full_moe_prefetch[GGML_SCHED_MAX_SPLIT_INPUTS];\n    ggml_backend_buffer_t transient_buffers[GGML_SCHED_MAX_SPLIT_INPUTS];\n""",
    1)

s = s.replace(
    """    bool op_offload;\n    bool force_weight_offload;\n    bool async_weight_prefetch;\n""",
    """    bool op_offload;\n    bool force_weight_offload;\n    bool async_weight_prefetch;\n    bool prefetch_full_moe;\n""",
    1)

needle = """static bool ggml_backend_sched_input_is_transient(\n        ggml_backend_sched_t sched, const struct ggml_tensor * input, int backend_id) {\n    return sched->force_weight_offload && sched->n_copies == 1 &&\n        input->view_src == NULL && input->buffer != NULL && input->data != NULL &&\n        ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&\n        ggml_backend_buffer_is_host(input->buffer) && !ggml_backend_buft_is_host(sched->bufts[backend_id]);\n}\n"""
insert = needle + """\nstatic bool ggml_backend_sched_input_is_full_moe_prefetch(\n        ggml_backend_sched_t sched, const struct ggml_tensor * node,\n        const struct ggml_tensor * input, int backend_id) {\n    if (!sched->prefetch_full_moe || sched->force_weight_offload || sched->n_copies != 1 ||\n            node == NULL || node->op != GGML_OP_MUL_MAT_ID || node->src[0] != input || node->src[2] == NULL ||\n            input == NULL || input->view_src != NULL || input->buffer == NULL || input->data == NULL ||\n            ggml_backend_buffer_get_usage(input->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||\n            !ggml_backend_buffer_is_host(input->buffer) || ggml_backend_buft_is_host(sched->bufts[backend_id]) ||\n            input->ne[2] <= 1) {\n        return false;\n    }\n\n    // At large prefill batches the router selects nearly every expert.  Waiting\n    // for the IDs and then issuing sparse copies only serializes PCIe behind the\n    // router.  The reference optimization used this conservative density test:\n    // at least twice as many routed IDs as experts.\n    const struct ggml_tensor * ids = node->src[2];\n    if (ids->ne[0] <= 0 || ids->ne[1] <= 0) {\n        return false;\n    }\n    const uint64_t n0 = (uint64_t) ids->ne[0];\n    const uint64_t n1 = (uint64_t) ids->ne[1];\n    const uint64_t n_ids = n0 > UINT64_MAX / n1 ? UINT64_MAX : n0 * n1;\n    const uint64_t n_expert = (uint64_t) input->ne[2];\n    return n_expert <= UINT64_MAX / 2 && n_ids >= 2 * n_expert;\n}\n"""
if needle not in s:
    raise SystemExit("transient helper anchor not found")
s = s.replace(needle, insert, 1)

# Every split reset must clear the new flag as well.
s = s.replace(
    "memset(split->input_transient, 0, sizeof(split->input_transient));\n",
    "memset(split->input_transient, 0, sizeof(split->input_transient));\n        memset(split->input_full_moe_prefetch, 0, sizeof(split->input_full_moe_prefetch));\n")

# Both transient decisions in graph splitting have node/src/backend in scope.
old_transient = "const bool transient = ggml_backend_sched_input_is_transient(sched, src, cur_backend_id);"
new_transient = "const bool transient = ggml_backend_sched_input_is_transient(sched, src, cur_backend_id) ||\n                                ggml_backend_sched_input_is_full_moe_prefetch(sched, node, src, cur_backend_id);"
count = s.count(old_transient)
if count != 2:
    raise SystemExit(f"expected 2 transient decision anchors, found {count}")
s = s.replace(old_transient, new_transient)

old_enroll = """                        split->input_prefetched[n_inputs] = false;\n                        split->input_transient[n_inputs] = transient;\n                        split->transient_buffers[n_inputs] = NULL;\n"""
new_enroll = """                        split->input_prefetched[n_inputs] = false;\n                        split->input_transient[n_inputs] = transient;\n                        split->input_full_moe_prefetch[n_inputs] =\n                            ggml_backend_sched_input_is_full_moe_prefetch(sched, node, src, cur_backend_id);\n                        split->transient_buffers[n_inputs] = NULL;\n"""
if old_enroll not in s:
    raise SystemExit("split enrollment anchor not found")
s = s.replace(old_enroll, new_enroll, 1)

start = s.find("static bool ggml_backend_sched_prefetch_resident_transient_input(")
end = s.find("\nstatic void ggml_backend_sched_prefetch_split_inputs", start)
if start < 0 or end < 0:
    raise SystemExit("prefetch transient function boundaries not found")
new_func = r'''static bool ggml_backend_sched_prefetch_resident_transient_input(
        ggml_backend_sched_t sched,
        struct ggml_backend_sched_split * split,
        int input_id,
        ggml_backend_t prefetch_backend) {
    const int backend_id = split->backend_id;
    if (!split->input_transient[input_id] || split->input_prefetched[input_id] ||
            split->transient_buffers[input_id] != NULL) {
        return false;
    }

    struct ggml_tensor * input = split->inputs[input_id];
    struct ggml_tensor * input_cpy = tensor_copy(input, backend_id, sched->cur_copy);
    const bool full_moe_prefetch = split->input_full_moe_prefetch[input_id];
    const bool is_moe = input_cpy != NULL && ggml_backend_sched_split_input_is_moe(split, input_id, input_cpy);
    if (input == NULL || input_cpy == NULL || input->data == NULL || input->view_src != NULL ||
            input_cpy->buffer != NULL || input_cpy->data != NULL ||
            (is_moe && !full_moe_prefetch) ||
            input->buffer == NULL || ggml_backend_buffer_get_usage(input->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            !ggml_backend_buffer_is_host(input->buffer) || ggml_backend_buft_is_host(sched->bufts[backend_id])) {
        return false;
    }

    // Full-layer MoE prefetch is intentionally ephemeral.  Caching the entire
    // expert tensor would compete with the compact hot-expert cache used by
    // sequential mode and can consume several GiB per layer.
    const bool cache_eligible = sched->residency_enabled[backend_id] && !full_moe_prefetch;
    const std::vector<int32_t> empty_experts;
    const auto resident_key = ggml_backend_sched_resident_key_make(input, backend_id, empty_experts);
    if (cache_eligible && sched->residents->find(resident_key) != sched->residents->end()) {
        return false;
    }

    const size_t alloc_size = ggml_backend_buft_get_alloc_size(sched->bufts[backend_id], input_cpy);
    if (alloc_size == 0) {
        return false;
    }
    const size_t split_limit = sched->max_weight_bytes_per_split[backend_id];
    size_t split_bytes = 0;
    for (int i = 0; i < split->n_inputs; ++i) {
        if (SIZE_MAX - split_bytes < split->transient_sizes[i]) {
            return false;
        }
        split_bytes += split->transient_sizes[i];
    }
    if (split_limit > 0 && (alloc_size > split_limit || split_bytes > split_limit - alloc_size)) {
        return false;
    }

    bool unknown_memory = false;
    bool live_guard_rejected = false;
    // Never evict from the early-prefetch path. Eviction synchronizes the
    // compute stream and would turn the intended overlap back into serialization.
    if (!ggml_backend_sched_weight_window_admit(
            sched, backend_id, alloc_size, &unknown_memory, &live_guard_rejected)) {
        return false;
    }

    auto & metrics = sched->transient_metrics.backends[backend_id];
    const int64_t allocation_start_us = ggml_time_us();
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(sched->bufts[backend_id], alloc_size);
    ggml_backend_sched_counter_add(sched, &metrics.allocation_time_us,
        ggml_backend_sched_elapsed_us(allocation_start_us));
    if (buffer == NULL) {
        return false;
    }
    if (ggml_backend_tensor_alloc(buffer, input_cpy, ggml_backend_buffer_get_base(buffer)) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        input_cpy->buffer = NULL;
        input_cpy->data = NULL;
        return false;
    }

    split->transient_buffers[input_id] = buffer;
    split->transient_sizes[input_id] = alloc_size;
    split->input_resident[input_id] = cache_eligible;
    split->input_resident_hit[input_id] = false;

    ggml_backend_sched_counter_add(sched, &metrics.allocation_requested_bytes, alloc_size);
    ggml_backend_sched_counter_add(sched, &metrics.allocation_admitted_bytes, alloc_size);
    ggml_backend_sched_counter_add(sched, &metrics.allocation_count, 1);

    if (cache_eligible) {
        ggml_backend_sched_resident resident{};
        resident.source = input;
        resident.source_buffer = input->buffer;
        resident.source_data = input->data;
        resident.logical_size = ggml_nbytes(input);
        resident.backend_id = backend_id;
        resident.copy = input_cpy;
        resident.buffer = buffer;
        resident.allocation_size = alloc_size;
        resident.completed_use = ++sched->residency_use_clock;
        resident.frequency = 1;
        resident.executing = true;
        resident.expert_tier = false;
        sched->residents->emplace(resident_key, resident);
        metrics.current_resident_bytes += alloc_size;
        metrics.current_resident_records++;
        ggml_backend_sched_counter_add(sched, &metrics.residency_miss_count, 1);
        ggml_backend_sched_resident_metrics_update(sched, backend_id);
    } else {
        ggml_backend_sched_ledger_enter(sched, backend_id, alloc_size);
    }

    // For mmap-backed Fable prefetch, submit directly from the registered mmap
    // pages. Sequential storage-backed weights retain the pinned staging path.
    ggml_backend_sched_weight_upload_chunked(sched, prefetch_backend, backend_id,
        input_cpy, input->data, 0, ggml_nbytes(input_cpy), !full_moe_prefetch);
    ggml_backend_sched_counter_add(sched, &metrics.upload_count, 1);
    ggml_backend_sched_counter_add(sched, &metrics.uploaded_backend_bytes, alloc_size);
    if (cache_eligible) {
        ggml_backend_sched_counter_add(sched, &metrics.residency_upload_count, 1);
    }
    if (!sched->transient_sources_seen->insert(input).second) {
        ggml_backend_sched_counter_add(sched, &metrics.shared_reload_count, 1);
    }

    split->input_prefetched[input_id] = true;
    return true;
}
'''
s = s[:start] + new_func + s[end:]

# Enable the already-existing secondary prefetch backend when the Fable MoE
# optimization is requested, even for ordinary (non-sequential) offload mode.
old_ctor = """    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);\n    sched->op_offload = op_offload;\n\n    ggml_backend_sched_reset(sched);\n"""
new_ctor = """    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);\n    sched->op_offload = op_offload;\n\n    const char * prefetch_moe_env = getenv(\"GGML_SCHED_PREFETCH_EXPERTS\");\n    sched->prefetch_full_moe = op_offload && prefetch_moe_env != NULL && atoi(prefetch_moe_env) > 0;\n    if (sched->prefetch_full_moe) {\n        // Standard offloaded-MoE models do not otherwise enable the sequential\n        // prefetch machinery. Reuse it rather than maintaining a second stream stack.\n        sched->async_weight_prefetch = true;\n    }\n\n    ggml_backend_sched_reset(sched);\n"""
if old_ctor not in s:
    raise SystemExit("scheduler constructor anchor not found")
s = s.replace(old_ctor, new_ctor, 1)

p.write_text(s)

# Remove this one-shot integration helper from the resulting source commit.
Path(__file__).unlink()
