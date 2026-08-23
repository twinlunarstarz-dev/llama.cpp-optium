from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    s = p.read_text()
    if old not in s:
        raise SystemExit(f'anchor not found in {path}: {old[:180]!r}')
    p.write_text(s.replace(old, new, 1))


def replace_between(path, start_marker, end_marker, replacement):
    p = Path(path)
    s = p.read_text()
    start = s.find(start_marker)
    if start < 0:
        raise SystemExit(f'start marker not found in {path}: {start_marker[:160]!r}')
    end = s.find(end_marker, start)
    if end < 0:
        raise SystemExit(f'end marker not found in {path}: {end_marker[:160]!r}')
    p.write_text(s[:start] + replacement + s[end:])


# -----------------------------------------------------------------------------
# 1. Adaptive sequential transfer chunking and prefetch depth.
# -----------------------------------------------------------------------------
replace_once(
    'ggml/src/ggml-backend.cpp',
    '''static constexpr size_t GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_SIZE = (size_t) 64 * 1024 * 1024;\nstatic constexpr size_t GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_THRESHOLD = (size_t) 1024 * 1024 * 1024;\n''',
    r'''static constexpr size_t GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_DEFAULT = (size_t) 64 * 1024 * 1024;
static constexpr size_t GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_MIN     = (size_t) 16 * 1024 * 1024;
static constexpr size_t GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_MAX     = (size_t) 256 * 1024 * 1024;
static constexpr size_t GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_THRESHOLD = (size_t) 1024 * 1024 * 1024;

static size_t ggml_backend_sched_transfer_chunk_size(
        ggml_backend_sched_t sched, int backend_id, size_t total_size) {
    if (total_size == 0) {
        return GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_MIN;
    }

    // Explicit override is useful for storage/controller-specific tuning. Keep a
    // bounded range so four pinned ring slots cannot accidentally consume many GiB.
    const char * env = getenv("GGML_SEQUENTIAL_CHUNK_MB");
    if (env != NULL && env[0] != '\0') {
        char * end = NULL;
        const unsigned long long mb = strtoull(env, &end, 10);
        if (end != env && mb > 0) {
            const size_t requested = mb > SIZE_MAX/(1024ull*1024ull) ? SIZE_MAX : (size_t) mb*1024ull*1024ull;
            return std::min(total_size, std::max(GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_MIN,
                std::min(requested, GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_MAX)));
        }
    }

    // Automatic mode is workload- and memory-aware. Small transfers use smaller
    // chunks to reduce first-byte latency; very large sequential weights use wider
    // chunks to amortize O_DIRECT and PCIe submission overhead when live VRAM allows.
    size_t chunk = GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_DEFAULT;
    if (total_size < (size_t) 256*1024*1024) {
        chunk = (size_t) 32*1024*1024;
    } else if (total_size >= (size_t) 8*1024*1024*1024ull) {
        chunk = GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_MAX;
    } else if (total_size >= (size_t) 2*1024*1024*1024ull) {
        chunk = (size_t) 128*1024*1024;
    }

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(sched->backends[backend_id]), &free_bytes, &total_bytes);
    if (free_bytes > 0 && total_bytes > 0 && free_bytes <= total_bytes) {
        if (free_bytes < (size_t) 2*1024*1024*1024ull) {
            chunk = std::min(chunk, (size_t) 32*1024*1024);
        } else if (free_bytes < (size_t) 4*1024*1024*1024ull) {
            chunk = std::min(chunk, GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_DEFAULT);
        }
    }

    return std::min(total_size, std::max(chunk, GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_MIN));
}

static int ggml_backend_sched_transfer_prefetch_depth(
        ggml_backend_sched_t sched, int backend_id, size_t chunk_size, size_t total_size) {
    const int max_depth = GGML_BACKEND_SCHED_STAGING_SLOTS - 1;
    const char * env = getenv("GGML_SEQUENTIAL_PREFETCH_DEPTH");
    if (env != NULL && env[0] != '\0') {
        char * end = NULL;
        const long requested = strtol(env, &end, 10);
        if (end != env && requested > 0) {
            return std::max(1, std::min(max_depth, (int) requested));
        }
    }

    int depth = chunk_size >= GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_MAX ? 1 :
        (chunk_size >= (size_t) 128*1024*1024 ? 2 : max_depth);

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(ggml_backend_get_device(sched->backends[backend_id]), &free_bytes, &total_bytes);
    if (free_bytes > 0 && total_bytes > 0 && free_bytes <= total_bytes) {
        if (free_bytes < (size_t) 2*1024*1024*1024ull) {
            depth = 1;
        } else if (free_bytes < (size_t) 4*1024*1024*1024ull) {
            depth = std::min(depth, 2);
        }
    }

    if (chunk_size > 0) {
        const size_t n_chunks = total_size/chunk_size + (total_size % chunk_size != 0);
        depth = std::min(depth, (int) std::min<size_t>(n_chunks, max_depth));
    }
    return std::max(1, depth);
}
''')

replace_once(
    'ggml/src/ggml-backend.cpp',
    '''    while (copied < size) {\n        const size_t chunk_size = instrument || storage_read ? GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_SIZE :\n            (size > GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_THRESHOLD ? GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_SIZE : size);\n        const size_t chunk = std::min(chunk_size, size - copied);\n''',
    '''    const size_t tuned_chunk_size = instrument || storage_read ?\n        ggml_backend_sched_transfer_chunk_size(sched, backend_id, size) :\n        (size > GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_THRESHOLD ? GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_DEFAULT : size);\n    while (copied < size) {\n        const size_t chunk = std::min(tuned_chunk_size, size - copied);\n''')

new_storage_prefetch = r'''static void ggml_backend_sched_prefetch_storage_inputs(ggml_backend_sched_t sched, int start_split_id) {
    if (!sched->async_weight_prefetch ||
            (sched->weight_read_callback == NULL && sched->weight_read_padded_callback == NULL) ||
            start_split_id < 0 || start_split_id >= sched->n_splits) {
        return;
    }

    for (int split_id = start_split_id; split_id < sched->n_splits; ++split_id) {
        auto * split = &sched->splits[split_id];
        const int backend_id = split->backend_id;
        for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
            ggml_tensor * input = split->inputs[input_id];
            ggml_tensor * input_cpy = tensor_copy(input, backend_id, 0);
            if (!split->input_transient[input_id] ||
                    ggml_backend_sched_split_input_is_moe(split, input_id, input_cpy) ||
                    input == NULL || input->data == NULL || input_cpy == NULL) {
                continue;
            }

            const size_t total = ggml_nbytes(input_cpy);
            const size_t chunk_size = ggml_backend_sched_transfer_chunk_size(sched, backend_id, total);
            const int max_active = ggml_backend_sched_transfer_prefetch_depth(
                sched, backend_id, chunk_size, total);
            if (ggml_backend_sched_storage_prefetch_active_count(sched, backend_id) >= max_active) {
                continue;
            }

            for (size_t copied = 0; copied < total; copied += chunk_size) {
                if (ggml_backend_sched_storage_prefetch_active_count(sched, backend_id) >= max_active) {
                    break;
                }
                const size_t size = std::min(chunk_size, total - copied);
                const uint8_t * logical_src = (const uint8_t *) input->data + copied;
                if (ggml_backend_sched_storage_task_exists(sched, backend_id, logical_src, size)) {
                    continue;
                }
                const int slot = ggml_backend_sched_staging_acquire(sched, backend_id, size, true);
                if (slot < 0) {
                    break;
                }
                auto & task = sched->storage_prefetch[backend_id][slot];
                task.logical_src = logical_src;
                task.size = size;
                task.data_offset = 0;
                task.split_id = split_id;
                task.input_id = input_id;
                task.active = true;
                task.success = false;
                task.worker = std::thread([sched, backend_id, slot]() {
                    auto & staging = sched->staging[backend_id];
                    auto & task = sched->storage_prefetch[backend_id][slot];
                    if (staging.pending[slot]) {
                        ggml_backend_event_synchronize(staging.events[slot]);
                    }
                    void * base = ggml_backend_buffer_get_base(staging.buffers[slot]);
                    size_t data_offset = 0;
                    if (sched->weight_read_padded_callback != NULL) {
                        task.success = sched->weight_read_padded_callback(
                            sched->weight_read_callback_user_data, task.logical_src,
                            base, staging.capacities[slot], task.size, &data_offset);
                    } else {
                        task.success = sched->weight_read_callback(
                            sched->weight_read_callback_user_data, task.logical_src, base, task.size);
                    }
                    task.data_offset = data_offset;
                });
            }
        }
    }
}

'''
replace_between(
    'ggml/src/ggml-backend.cpp',
    'static void ggml_backend_sched_prefetch_storage_inputs(ggml_backend_sched_t sched, int start_split_id) {',
    '// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend\n',
    new_storage_prefetch)


# -----------------------------------------------------------------------------
# 2. Actual N+1 H2D overlap for sequential transient weights.
#    Only early-admit a resident when it fits without eviction; otherwise the
#    ordinary JIT path remains authoritative and may evict safely after N completes.
# -----------------------------------------------------------------------------
new_split_prefetch = r'''static bool ggml_backend_sched_prefetch_resident_transient_input(
        ggml_backend_sched_t sched,
        struct ggml_backend_sched_split * split,
        int input_id,
        ggml_backend_t prefetch_backend) {
    const int backend_id = split->backend_id;
    if (!split->input_transient[input_id] || split->input_prefetched[input_id] ||
            split->transient_buffers[input_id] != NULL || !sched->residency_enabled[backend_id]) {
        return false;
    }

    struct ggml_tensor * input = split->inputs[input_id];
    struct ggml_tensor * input_cpy = tensor_copy(input, backend_id, sched->cur_copy);
    if (input == NULL || input_cpy == NULL || input->data == NULL || input->view_src != NULL ||
            input_cpy->buffer != NULL || input_cpy->data != NULL ||
            ggml_backend_sched_split_input_is_moe(split, input_id, input_cpy) ||
            input->buffer == NULL || ggml_backend_buffer_get_usage(input->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            !ggml_backend_buffer_is_host(input->buffer) || ggml_backend_buft_is_host(sched->bufts[backend_id])) {
        return false;
    }

    const std::vector<int32_t> empty_experts;
    const auto resident_key = ggml_backend_sched_resident_key_make(input, backend_id, empty_experts);
    auto found = sched->residents->find(resident_key);
    if (found != sched->residents->end()) {
        // A valid cache hit needs no H2D. A stale/mismatched record is deliberately
        // left to the normal split path, which owns synchronization and eviction.
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
    // Do not call make_resident_space() here: eviction synchronizes the compute
    // backend and would destroy N/N+1 overlap. Early H2D is strictly opportunistic.
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

    split->transient_buffers[input_id] = buffer;
    split->transient_sizes[input_id] = alloc_size;
    split->input_resident[input_id] = true;
    split->input_resident_hit[input_id] = false;

    metrics.current_resident_bytes += alloc_size;
    metrics.current_resident_records++;
    ggml_backend_sched_counter_add(sched, &metrics.allocation_requested_bytes, alloc_size);
    ggml_backend_sched_counter_add(sched, &metrics.allocation_admitted_bytes, alloc_size);
    ggml_backend_sched_counter_add(sched, &metrics.allocation_count, 1);
    ggml_backend_sched_counter_add(sched, &metrics.residency_miss_count, 1);
    ggml_backend_sched_resident_metrics_update(sched, backend_id);

    // If O_DIRECT prefetch already filled pinned slots, this consumes those bytes
    // immediately. Otherwise the read happens on the scheduler thread while N is
    // executing on the GPU, then H2D is queued on the secondary same-device stream.
    ggml_backend_sched_weight_upload_chunked(sched, prefetch_backend, backend_id,
        input_cpy, input->data, 0, ggml_nbytes(input_cpy), true);
    ggml_backend_sched_counter_add(sched, &metrics.upload_count, 1);
    ggml_backend_sched_counter_add(sched, &metrics.uploaded_backend_bytes, alloc_size);
    ggml_backend_sched_counter_add(sched, &metrics.residency_upload_count, 1);
    if (!sched->transient_sources_seen->insert(input).second) {
        ggml_backend_sched_counter_add(sched, &metrics.shared_reload_count, 1);
    }

    split->input_prefetched[input_id] = true;
    return true;
}

static void ggml_backend_sched_prefetch_split_inputs(ggml_backend_sched_t sched, int split_id) {
    if (!sched->async_weight_prefetch || split_id <= 0 || split_id >= sched->n_splits) {
        return;
    }

    struct ggml_backend_sched_split * split = &sched->splits[split_id];
    const int split_backend_id = split->backend_id;
    if (split->has_prefetched_inputs) {
        return;
    }

    ggml_backend_sched_init_prefetch_backend(sched, split_backend_id);

    ggml_backend_t prefetch_backend = sched->prefetch_backends[split_backend_id];
    ggml_backend_event_t prefetch_event = sched->prefetch_events[split_backend_id][split_id & 1];
    if (prefetch_backend == NULL || prefetch_event == NULL) {
        return;
    }

    bool prefetched = false;
    for (int input_id = 0; input_id < split->n_inputs; input_id++) {
        if (split->input_transient[input_id]) {
            prefetched = ggml_backend_sched_prefetch_resident_transient_input(
                sched, split, input_id, prefetch_backend) || prefetched;
            continue;
        }

        if (!ggml_backend_sched_split_input_can_prefetch(sched, split, input_id)) {
            continue;
        }

        struct ggml_tensor * input = split->inputs[input_id];
        if (input->view_src != NULL || tensor_backend_id(input) != sched->n_backends - 1) {
            continue;
        }

        struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
        if (input->data == NULL || input_cpy == NULL || input_cpy->buffer == NULL || ggml_backend_buffer_is_host(input_cpy->buffer)) {
            continue;
        }

        ggml_backend_tensor_set_async(prefetch_backend, input_cpy, input->data, 0, ggml_nbytes(input_cpy));
        split->input_prefetched[input_id] = true;
        prefetched = true;
    }

    if (prefetched) {
        ggml_backend_event_record(prefetch_event, prefetch_backend);
        split->has_prefetched_inputs = true;
    }
}

'''
replace_between(
    'ggml/src/ggml-backend.cpp',
    'static void ggml_backend_sched_prefetch_split_inputs(ggml_backend_sched_t sched, int split_id) {',
    'static int ggml_backend_sched_storage_prefetch_active_count(ggml_backend_sched_t sched, int backend_id) {',
    new_split_prefetch)

# Early-prefetched sequential inputs are already allocated, attached, and resident.
replace_once(
    'ggml/src/ggml-backend.cpp',
    '''            struct ggml_tensor * input_cpy = tensor_copy(split->inputs[input_id], split_backend_id, 0);\n            GGML_ASSERT(input_cpy != NULL && (input_cpy->flags & GGML_TENSOR_FLAG_NO_ALLOC));\n            const struct ggml_tensor * source = split->inputs[input_id];\n            const bool cache_eligible = sched->residency_enabled[split_backend_id];\n''',
    '''            struct ggml_tensor * input_cpy = tensor_copy(split->inputs[input_id], split_backend_id, 0);\n            GGML_ASSERT(input_cpy != NULL && (input_cpy->flags & GGML_TENSOR_FLAG_NO_ALLOC));\n            if (split->input_prefetched[input_id] && split->transient_buffers[input_id] != NULL) {\n                GGML_ASSERT(input_cpy->buffer == split->transient_buffers[input_id] && input_cpy->data != NULL);\n                GGML_ASSERT(SIZE_MAX - split_transient_bytes >= split->transient_sizes[input_id]);\n                split_transient_bytes += split->transient_sizes[input_id];\n                split_has_transients = true;\n                continue;\n            }\n            const struct ggml_tensor * source = split->inputs[input_id];\n            const bool cache_eligible = sched->residency_enabled[split_backend_id];\n''')

# Launch N+1 prefetch after N is submitted but before waiting for N completion.
replace_once(
    'ggml/src/ggml-backend.cpp',
    '''\n        ggml_backend_sched_prefetch_split_inputs(sched, split_id + 1);\n\n        ggml_backend_sched_release_transients(sched, split,\n''',
    '''\n        ggml_backend_sched_release_transients(sched, split,\n''')
replace_once(
    'ggml/src/ggml-backend.cpp',
    '''\n        // record the event of this copy\n''',
    '''\n        // With N executing asynchronously, allocate/admit N+1 only when spare\n        // residency space is immediately available, then queue H2D on the\n        // secondary backend stream. Different-GPU transitions are valid too.\n        ggml_backend_sched_prefetch_split_inputs(sched, split_id + 1);\n\n        // record the event of this copy\n''')


# -----------------------------------------------------------------------------
# 3. Decouple sequential weight placement from KV placement.
# -----------------------------------------------------------------------------
replace_once(
    'src/llama-model.h',
    '''    ggml_backend_dev_t dev_layer(int il) const;\n    ggml_backend_dev_t dev_output() const;\n''',
    '''    ggml_backend_dev_t dev_layer(int il) const;\n    ggml_backend_dev_t dev_kv_layer(int il) const;\n    ggml_backend_dev_t dev_output() const;\n''')

replace_once(
    'src/llama-model.cpp',
    '''    layer_dev dev_input = {};\n    layer_dev dev_output = {};\n    std::vector<layer_dev> dev_layer;\n''',
    '''    layer_dev dev_input = {};\n    layer_dev dev_output = {};\n    std::vector<layer_dev> dev_layer;\n    std::vector<layer_dev> dev_kv_layer;\n''')

replace_once(
    'src/llama-model.cpp',
    '''ggml_backend_dev_t llama_model::dev_layer(int il) const {\n    return pimpl->dev_layer.at(il).dev;\n}\n\nggml_backend_dev_t llama_model::dev_output() const {\n''',
    '''ggml_backend_dev_t llama_model::dev_layer(int il) const {\n    return pimpl->dev_layer.at(il).dev;\n}\n\nggml_backend_dev_t llama_model::dev_kv_layer(int il) const {\n    return pimpl->dev_kv_layer.at(il).dev;\n}\n\nggml_backend_dev_t llama_model::dev_output() const {\n''')

replace_once(
    'src/llama-model.cpp',
    '''    auto get_layer_buft_list = [&](int il) -> llama_model::impl::layer_dev {\n        const bool is_swa = il < n_layer_all && hparams.is_swa(il);\n        // sequential load: ALL layers start on CPU (mmap-direct), GPU window managed at runtime\n        if (pimpl->sequential_load) {\n''',
    '''    auto get_layer_buft_list = [&](int il, bool sequential_weight) -> llama_model::impl::layer_dev {\n        const bool is_swa = il < n_layer_all && hparams.is_swa(il);\n        // Sequential weights remain CPU/storage-backed, but KV placement uses the\n        // ordinary layer/tensor-split mapping so --kv-offload remains effective.\n        if (pimpl->sequential_load && sequential_weight) {\n''')

replace_once(
    'src/llama-model.cpp',
    '''    // assign the repeating layers to the devices according to the splits\n    pimpl->dev_layer.resize(n_layer_all);\n    for (int il = 0; il < n_layer_all; ++il) {\n        pimpl->dev_layer[il] = get_layer_buft_list(il);\n    }\n\n    // assign the output layer\n    pimpl->dev_output = get_layer_buft_list(n_layer_all);\n''',
    '''    // Assign persistent weights and KV independently. In ordinary loading the\n    // two maps are identical; sequential loading keeps only weights host-backed.\n    pimpl->dev_layer.resize(n_layer_all);\n    pimpl->dev_kv_layer.resize(n_layer_all);\n    for (int il = 0; il < n_layer_all; ++il) {\n        pimpl->dev_layer[il] = get_layer_buft_list(il, true);\n        pimpl->dev_kv_layer[il] = pimpl->sequential_load ?\n            get_layer_buft_list(il, false) : pimpl->dev_layer[il];\n    }\n\n    // assign the output layer\n    pimpl->dev_output = get_layer_buft_list(n_layer_all, true);\n''')

replace_once(
    'src/llama-kv-cache.cpp',
    '''            auto * dev = model.dev_layer(il);\n            buft = ggml_backend_dev_buffer_type(dev);\n''',
    '''            auto * dev = model.dev_kv_layer(il);\n            buft = ggml_backend_dev_buffer_type(dev);\n''')

replace_once(
    'src/llama-kv-cache-dsv4.cpp',
    '''            auto * dev = model.dev_layer(il);\n            buft = ggml_backend_dev_buffer_type(dev);\n''',
    '''            auto * dev = model.dev_kv_layer(il);\n            buft = ggml_backend_dev_buffer_type(dev);\n''')

replace_once(
    'src/llama-context.cpp',
    '''            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,\n            // but is still wrong for cases like --no-kv-offload.\n            ggml_backend_dev_t device_layer = model.dev_layer(node.il);\n''',
    '''            // Sequential weights are storage-backed, while fused attention/state\n            // kernels execute with the runtime layer/KV placement. Use that placement\n            // for capability probing so sequential mode does not disable CUDA fusion\n            // merely because persistent weights live on CPU.\n            ggml_backend_dev_t device_layer = model.is_sequential() ?\n                model.dev_kv_layer(node.il) : model.dev_layer(node.il);\n''')


# -----------------------------------------------------------------------------
# 4. IQ4_NL CUDA FlashAttention fallback: admit the storage type generally but
#    never select a direct vector specialization. TILE/MMA paths already request
#    F16 staging and convert.cu provides IQ4_NL -> F16 conversion.
# -----------------------------------------------------------------------------
replace_once(
    'ggml/src/ggml-cuda/fattn.cu',
    '''        case GGML_TYPE_Q4_0:\n        case GGML_TYPE_Q8_0:\n        case GGML_TYPE_BF16:\n            return true;\n''',
    '''        case GGML_TYPE_Q4_0:\n        case GGML_TYPE_Q8_0:\n        case GGML_TYPE_BF16:\n        case GGML_TYPE_IQ4_NL:\n            return true;\n''')

replace_once(
    'ggml/src/ggml-cuda/fattn.cu',
    '''    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;\n''',
    '''    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 &&\n        K->ne[1] % FATTN_KQ_STRIDE == 0 && K->type != GGML_TYPE_IQ4_NL && V->type != GGML_TYPE_IQ4_NL;\n''')


# -----------------------------------------------------------------------------
# 5. Remove one-shot integration patchers after this transaction succeeds.
# -----------------------------------------------------------------------------
Path('integration/apply_pending.py').unlink(missing_ok=True)
Path('integration/apply_final.py').unlink(missing_ok=True)
