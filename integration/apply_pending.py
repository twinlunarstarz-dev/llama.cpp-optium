from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    s = p.read_text()
    if old not in s:
        raise SystemExit(f'anchor not found in {path}: {old[:100]!r}')
    p.write_text(s.replace(old, new, 1))


# Public scheduler API: keep the exact reader for compatibility and add a padded
# reader so O_DIRECT can target aligned bytes inside a reusable pinned allocation.
replace_once(
    'ggml/include/ggml-backend.h',
    '''    // Optional source reader for storage-backed sequential weights. The callback\n    // copies exactly size bytes from the logical tensor address into dst and\n    // returns true when it handled the address. Unhandled addresses use memcpy.\n    typedef bool (*ggml_backend_sched_weight_read_callback)(\n            void * user_data, const void * logical_src, void * dst, size_t size);\n    GGML_API void                 ggml_backend_sched_set_weight_read_callback(\n            ggml_backend_sched_t sched, ggml_backend_sched_weight_read_callback callback, void * user_data);\n''',
    '''    // Optional source readers for storage-backed sequential weights. The exact\n    // callback copies size bytes to dst. The padded callback may place those bytes\n    // at data_offset inside a larger destination allocation, which lets aligned\n    // direct I/O read into pinned host memory without an intermediate host copy.\n    typedef bool (*ggml_backend_sched_weight_read_callback)(\n            void * user_data, const void * logical_src, void * dst, size_t size);\n    typedef bool (*ggml_backend_sched_weight_read_padded_callback)(\n            void * user_data, const void * logical_src, void * dst, size_t capacity,\n            size_t size, size_t * data_offset);\n    GGML_API void                 ggml_backend_sched_set_weight_read_callback(\n            ggml_backend_sched_t sched, ggml_backend_sched_weight_read_callback callback, void * user_data);\n    GGML_API void                 ggml_backend_sched_set_weight_read_padded_callback(\n            ggml_backend_sched_t sched, ggml_backend_sched_weight_read_padded_callback callback, void * user_data);\n''')

# Model exposes whether its sequential logical pointers are storage identities
# rather than directly readable mmap addresses.
replace_once(
    'src/llama-model.h',
    '''    bool is_sequential() const;\n    bool read_sequential_weight(const void * logical_src, void * dst, size_t size) const;\n''',
    '''    bool is_sequential() const;\n    bool is_sequential_direct_io() const;\n    bool read_sequential_weight(const void * logical_src, void * dst, size_t size) const;\n''')
replace_once(
    'src/llama-model.cpp',
    '''bool llama_model::is_sequential() const {\n    return pimpl->sequential_load;\n}\n\n\nbool llama_model::read_sequential_weight''',
    '''bool llama_model::is_sequential() const {\n    return pimpl->sequential_load;\n}\n\nbool llama_model::is_sequential_direct_io() const {\n    return pimpl->sequential_load && !pimpl->direct_io_regions.empty();\n}\n\nbool llama_model::read_sequential_weight''')

# Scheduler storage/pinned state: four reusable slots, one storage task per slot,
# and a proper C++ object lifetime for std::thread members.
replace_once(
    'ggml/src/ggml-backend.cpp',
    '''using ggml_backend_sched_resident_map = std::unordered_map<\n    ggml_backend_sched_resident_key, ggml_backend_sched_resident, ggml_backend_sched_resident_key_hash>;\n\nstruct ggml_backend_sched_staging {\n    ggml_backend_buffer_t buffers[2];\n    ggml_backend_event_t events[2];\n    size_t capacities[2];\n    bool pending[2];\n    int next;\n};\n\nstruct ggml_backend_sched_storage_prefetch {\n    std::thread worker;\n    std::vector<uint8_t> data;\n    const uint8_t * logical_src;\n    size_t size;\n    size_t consumed;\n    bool active;\n    bool success;\n};\n''',
    '''using ggml_backend_sched_resident_map = std::unordered_map<\n    ggml_backend_sched_resident_key, ggml_backend_sched_resident, ggml_backend_sched_resident_key_hash>;\n\nstatic constexpr int GGML_BACKEND_SCHED_STAGING_SLOTS = 4;\nstatic constexpr size_t GGML_BACKEND_SCHED_STORAGE_PADDING = (size_t) 2 * 1024 * 1024;\n\nstruct ggml_backend_sched_staging {\n    ggml_backend_buffer_t buffers[GGML_BACKEND_SCHED_STAGING_SLOTS];\n    ggml_backend_event_t events[GGML_BACKEND_SCHED_STAGING_SLOTS];\n    size_t capacities[GGML_BACKEND_SCHED_STAGING_SLOTS];\n    bool pending[GGML_BACKEND_SCHED_STAGING_SLOTS];\n    bool reserved[GGML_BACKEND_SCHED_STAGING_SLOTS];\n    int next;\n};\n\nstruct ggml_backend_sched_storage_prefetch {\n    std::thread worker;\n    const uint8_t * logical_src;\n    size_t size;\n    size_t data_offset;\n    int split_id;\n    int input_id;\n    bool active;\n    bool success;\n};\n''')
replace_once(
    'ggml/src/ggml-backend.cpp',
    '''    ggml_backend_sched_weight_read_callback weight_read_callback;\n    void * weight_read_callback_user_data;\n''',
    '''    ggml_backend_sched_weight_read_callback weight_read_callback;\n    ggml_backend_sched_weight_read_padded_callback weight_read_padded_callback;\n    void * weight_read_callback_user_data;\n''')
replace_once(
    'ggml/src/ggml-backend.cpp',
    '''    struct ggml_backend_sched_staging staging[GGML_SCHED_MAX_BACKENDS];\n    struct ggml_backend_sched_storage_prefetch storage_prefetch[GGML_SCHED_MAX_BACKENDS];\n''',
    '''    struct ggml_backend_sched_staging staging[GGML_SCHED_MAX_BACKENDS];\n    struct ggml_backend_sched_storage_prefetch\n        storage_prefetch[GGML_SCHED_MAX_BACKENDS][GGML_BACKEND_SCHED_STAGING_SLOTS];\n''')

# Replace vector-backed one-task prefetch + two-slot staging with direct reads into
# reserved pinned slots. A prefetch worker may wait on the slot's previous H2D event;
# that wait happens off the scheduler thread and therefore overlaps GPU compute.
p = Path('ggml/src/ggml-backend.cpp')
s = p.read_text()
start = s.index('static void ggml_backend_sched_storage_prefetch_finish(')
end = s.index('static void ggml_backend_sched_weight_upload_chunked(', start)
new_helpers = r'''static void ggml_backend_sched_storage_prefetch_finish(ggml_backend_sched_storage_prefetch & prefetch) {
    if (prefetch.worker.joinable()) {
        prefetch.worker.join();
    }
}

static void ggml_backend_sched_storage_prefetch_release(
        ggml_backend_sched_t sched, int backend_id, int slot) {
    auto & task = sched->storage_prefetch[backend_id][slot];
    if (!task.active && !task.worker.joinable()) {
        return;
    }
    ggml_backend_sched_storage_prefetch_finish(task);
    // A storage worker waits for the previous H2D event before touching the slot.
    // Once joined, that old transfer is complete even when the read itself failed.
    sched->staging[backend_id].pending[slot] = false;
    sched->staging[backend_id].reserved[slot] = false;
    task.active = false;
    task.success = false;
    task.logical_src = NULL;
    task.size = 0;
    task.data_offset = 0;
    task.split_id = -1;
    task.input_id = -1;
}

static size_t ggml_backend_sched_staging_required_capacity(ggml_backend_sched_t sched, size_t size) {
    if (sched->weight_read_padded_callback == NULL) {
        return size;
    }
    return size <= SIZE_MAX - GGML_BACKEND_SCHED_STORAGE_PADDING ?
        size + GGML_BACKEND_SCHED_STORAGE_PADDING : SIZE_MAX;
}

static void ggml_backend_sched_staging_metrics_update(ggml_backend_sched_t sched, int backend_id) {
    size_t bytes = 0;
    for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {
        bytes += sched->staging[backend_id].capacities[slot];
    }
    sched->transient_metrics.backends[backend_id].staging_buffer_bytes = bytes;
}

static bool ggml_backend_sched_staging_prepare(
        ggml_backend_sched_t sched, int backend_id, int slot, size_t size, bool wait_pending) {
    auto & staging = sched->staging[backend_id];
    ggml_backend_t backend = sched->backends[backend_id];
    if (staging.reserved[slot] || backend->iface.event_record == NULL || staging.events[slot] == NULL) {
        return false;
    }
    const size_t required = ggml_backend_sched_staging_required_capacity(sched, size);
    if (required == SIZE_MAX) {
        return false;
    }
    if (staging.pending[slot]) {
        // Prefetch workers can reserve an already-large-enough slot and wait for its
        // previous H2D event themselves. Reallocation, however, must happen here only
        // after the prior transfer is complete.
        if (!wait_pending && staging.capacities[slot] >= required) {
            return true;
        }
        if (!wait_pending) {
            return false;
        }
        ggml_backend_event_synchronize(staging.events[slot]);
        staging.pending[slot] = false;
    }
    if (staging.capacities[slot] >= required) {
        return true;
    }
    ggml_backend_buffer_free(staging.buffers[slot]);
    staging.buffers[slot] = NULL;
    staging.capacities[slot] = 0;
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
    if (host_buft == NULL) {
        return false;
    }
    staging.buffers[slot] = ggml_backend_buft_alloc_buffer(host_buft, required);
    if (staging.buffers[slot] == NULL || !ggml_backend_buffer_is_host(staging.buffers[slot])) {
        ggml_backend_buffer_free(staging.buffers[slot]);
        staging.buffers[slot] = NULL;
        return false;
    }
    staging.capacities[slot] = ggml_backend_buffer_get_size(staging.buffers[slot]);
    ggml_backend_sched_staging_metrics_update(sched, backend_id);
    return staging.capacities[slot] >= required;
}

static int ggml_backend_sched_staging_acquire(
        ggml_backend_sched_t sched, int backend_id, size_t size, bool for_prefetch) {
    auto & staging = sched->staging[backend_id];
    for (int pass = 0; pass < (for_prefetch ? 1 : 2); ++pass) {
        for (int n = 0; n < GGML_BACKEND_SCHED_STAGING_SLOTS; ++n) {
            const int slot = (staging.next + n) % GGML_BACKEND_SCHED_STAGING_SLOTS;
            if (staging.reserved[slot]) {
                continue;
            }
            if (pass == 0 && staging.pending[slot] && !for_prefetch) {
                continue;
            }
            if (!ggml_backend_sched_staging_prepare(sched, backend_id, slot, size, !for_prefetch)) {
                continue;
            }
            staging.reserved[slot] = true;
            staging.next = (slot + 1) % GGML_BACKEND_SCHED_STAGING_SLOTS;
            return slot;
        }
    }
    return -1;
}

static bool ggml_backend_sched_storage_task_exists(
        ggml_backend_sched_t sched, int backend_id, const uint8_t * logical_src, size_t size) {
    for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {
        const auto & task = sched->storage_prefetch[backend_id][slot];
        if (task.active && task.logical_src == logical_src && task.size == size) {
            return true;
        }
    }
    return false;
}

static int ggml_backend_sched_storage_prefetch_consume(
        ggml_backend_sched_t sched, int backend_id, const uint8_t * logical_src, size_t size,
        const uint8_t ** data) {
    for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {
        auto & task = sched->storage_prefetch[backend_id][slot];
        if (!task.active || task.logical_src != logical_src || task.size != size) {
            continue;
        }
        ggml_backend_sched_storage_prefetch_finish(task);
        auto & staging = sched->staging[backend_id];
        // The worker waits on the old upload before reading, so the slot is no
        // longer pending at this point. Keep it reserved until the new H2D is queued.
        staging.pending[slot] = false;
        if (!task.success || task.data_offset > staging.capacities[slot] ||
                size > staging.capacities[slot] - task.data_offset) {
            staging.reserved[slot] = false;
            task.active = false;
            task.success = false;
            return -1;
        }
        *data = (const uint8_t *) ggml_backend_buffer_get_base(staging.buffers[slot]) + task.data_offset;
        task.active = false;
        task.success = false;
        return slot;
    }
    return -1;
}

static bool ggml_backend_sched_read_storage_into_slot(
        ggml_backend_sched_t sched, int backend_id, int slot,
        const uint8_t * logical_src, size_t size, const uint8_t ** data) {
    auto & staging = sched->staging[backend_id];
    void * base = ggml_backend_buffer_get_base(staging.buffers[slot]);
    size_t data_offset = 0;
    bool ok = false;
    if (sched->weight_read_padded_callback != NULL) {
        ok = sched->weight_read_padded_callback(sched->weight_read_callback_user_data,
            logical_src, base, staging.capacities[slot], size, &data_offset);
    } else if (sched->weight_read_callback != NULL) {
        ok = sched->weight_read_callback(sched->weight_read_callback_user_data,
            logical_src, base, size);
    }
    if (!ok || data_offset > staging.capacities[slot] || size > staging.capacities[slot] - data_offset) {
        return false;
    }
    *data = (const uint8_t *) base + data_offset;
    return true;
}

'''
s = s[:start] + new_helpers + s[end:]
p.write_text(s)

# Replace the upload function body through its next helper. This version consumes
# prefetched pinned slots directly and otherwise reads synchronously into one free
# pinned slot. No storage-backed path falls back to dereferencing a logical pointer.
p = Path('ggml/src/ggml-backend.cpp')
s = p.read_text()
start = s.index('static void ggml_backend_sched_weight_upload_chunked(')
end = s.index('static size_t ggml_backend_sched_weight_window_safety_reserve(', start)
new_upload = r'''static void ggml_backend_sched_weight_upload_chunked(
        ggml_backend_sched_t sched,
        ggml_backend_t backend,
        int backend_id,
        struct ggml_tensor * dst,
        const void * src,
        size_t offset,
        size_t size,
        bool instrument) {
    const uint8_t * src_bytes = (const uint8_t *) src;
    size_t copied = 0;
    const int64_t upload_start_us = instrument ? ggml_time_us() : 0;
    const ggml_backend_sched_fault_sample faults_before = instrument ? ggml_backend_sched_faults() : ggml_backend_sched_fault_sample{};
    const bool storage_read = sched->weight_read_callback != NULL || sched->weight_read_padded_callback != NULL;
    if (instrument && !storage_read) {
        ggml_backend_sched_readahead(sched, backend_id, src, size);
    }
    while (copied < size) {
        const size_t chunk_size = instrument || storage_read ? GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_SIZE :
            (size > GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_THRESHOLD ? GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_SIZE : size);
        const size_t chunk = std::min(chunk_size, size - copied);
        const uint8_t * staged_data = NULL;
        int slot = storage_read ? ggml_backend_sched_storage_prefetch_consume(
            sched, backend_id, src_bytes + copied, chunk, &staged_data) : -1;
        if (slot < 0 && (instrument || storage_read)) {
            slot = ggml_backend_sched_staging_acquire(sched, backend_id, chunk, false);
            if (slot >= 0) {
                auto & staging = sched->staging[backend_id];
                if (storage_read) {
                    if (!ggml_backend_sched_read_storage_into_slot(
                            sched, backend_id, slot, src_bytes + copied, chunk, &staged_data)) {
                        staging.reserved[slot] = false;
                        GGML_ASSERT(false && "storage-backed weight callback rejected its logical source");
                    }
                } else {
                    staged_data = (const uint8_t *) ggml_backend_buffer_get_base(staging.buffers[slot]);
                    memcpy((void *) staged_data, src_bytes + copied, chunk);
                }
            }
        }
        if (slot >= 0) {
            auto & staging = sched->staging[backend_id];
            ggml_backend_tensor_set_async(backend, dst, staged_data, offset + copied, chunk);
            ggml_backend_event_record(staging.events[slot], backend);
            staging.pending[slot] = true;
            staging.reserved[slot] = false;
            if (instrument) {
                auto & metrics = sched->transient_metrics.backends[backend_id];
                ggml_backend_sched_counter_add(sched, &metrics.staged_upload_chunk_count, 1);
                ggml_backend_sched_counter_add(sched, &metrics.staged_upload_bytes, chunk);
            }
        } else {
            GGML_ASSERT(!storage_read && "storage-backed weights require a scheduler staging slot");
            ggml_backend_tensor_set_async(backend, dst, src_bytes + copied, offset + copied, chunk);
        }
        if (instrument) {
            auto & metrics = sched->transient_metrics.backends[backend_id];
            ggml_backend_sched_counter_add(sched, &metrics.upload_chunk_count, 1);
            ggml_backend_sched_counter_add(sched, &metrics.uploaded_logical_bytes, chunk);
            metrics.max_upload_chunk_bytes = std::max(metrics.max_upload_chunk_bytes, chunk);
        }
        copied += chunk;
    }
    if (instrument) {
        const ggml_backend_sched_fault_sample faults_after = ggml_backend_sched_faults();
        auto & metrics = sched->transient_metrics.backends[backend_id];
        ggml_backend_sched_counter_add(sched, &metrics.mmap_minor_faults,
            faults_after.minor >= faults_before.minor ? faults_after.minor - faults_before.minor : 0);
        ggml_backend_sched_counter_add(sched, &metrics.mmap_major_faults,
            faults_after.major >= faults_before.major ? faults_after.major - faults_before.major : 0);
        ggml_backend_sched_counter_add(sched,
            &metrics.upload_submission_time_us,
            ggml_backend_sched_elapsed_us(upload_start_us));
    }
}

'''
s = s[:start] + new_upload + s[end:]
p.write_text(s)

# Replace the one-split/vector storage prefetch function with a bounded three-task
# lookahead per backend. Each task reads one scheduler upload chunk directly into
# its reserved pinned slot.
p = Path('ggml/src/ggml-backend.cpp')
s = p.read_text()
start = s.index('static void ggml_backend_sched_prefetch_storage_input(')
end = s.index('// assigns backends to ops and splits the graph into subgraphs', start)
new_prefetch = r'''static int ggml_backend_sched_storage_prefetch_active_count(ggml_backend_sched_t sched, int backend_id) {
    int count = 0;
    for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {
        count += sched->storage_prefetch[backend_id][slot].active ? 1 : 0;
    }
    return count;
}

static void ggml_backend_sched_prefetch_storage_inputs(ggml_backend_sched_t sched, int start_split_id) {
    if (!sched->async_weight_prefetch ||
            (sched->weight_read_callback == NULL && sched->weight_read_padded_callback == NULL) ||
            start_split_id < 0 || start_split_id >= sched->n_splits) {
        return;
    }

    const int max_active = GGML_BACKEND_SCHED_STAGING_SLOTS - 1;
    for (int split_id = start_split_id; split_id < sched->n_splits; ++split_id) {
        auto * split = &sched->splits[split_id];
        const int backend_id = split->backend_id;
        if (ggml_backend_sched_storage_prefetch_active_count(sched, backend_id) >= max_active) {
            continue;
        }
        for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
            ggml_tensor * input = split->inputs[input_id];
            ggml_tensor * input_cpy = tensor_copy(input, backend_id, 0);
            if (!split->input_transient[input_id] ||
                    ggml_backend_sched_split_input_is_moe(split, input_id, input_cpy) ||
                    input == NULL || input->data == NULL || input_cpy == NULL) {
                continue;
            }
            const size_t total = ggml_nbytes(input_cpy);
            for (size_t copied = 0; copied < total; copied += GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_SIZE) {
                if (ggml_backend_sched_storage_prefetch_active_count(sched, backend_id) >= max_active) {
                    break;
                }
                const size_t size = std::min(GGML_BACKEND_SCHED_WEIGHT_UPLOAD_CHUNK_SIZE, total - copied);
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
s = s[:start] + new_prefetch + s[end:]
s = s.replace('ggml_backend_sched_prefetch_storage_input(sched, 0);',
              'ggml_backend_sched_prefetch_storage_inputs(sched, 0);', 1)
s = s.replace('ggml_backend_sched_prefetch_storage_input(sched, split_id + 1);',
              'ggml_backend_sched_prefetch_storage_inputs(sched, split_id + 1);', 1)
# General prefetch eligibility should be disabled for either storage callback.
s = s.replace('if (!sched->async_weight_prefetch || sched->weight_read_callback != NULL ||\n',
              'if (!sched->async_weight_prefetch || sched->weight_read_callback != NULL || sched->weight_read_padded_callback != NULL ||\n', 1)
# A split completion event covers all earlier H2D submissions on the backend stream.
s = s.replace('''            sched->staging[split_backend_id].pending[0] = false;\n            sched->staging[split_backend_id].pending[1] = false;\n''',
'''            for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {\n                if (!sched->staging[split_backend_id].reserved[slot]) {\n                    sched->staging[split_backend_id].pending[slot] = false;\n                }\n            }\n''', 1)
p.write_text(s)

# Properly construct/destruct the scheduler and expand all staging init/free/reset loops.
p = Path('ggml/src/ggml-backend.cpp')
s = p.read_text()
s = s.replace('''    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));\n''',
              '''    struct ggml_backend_sched * sched = new ggml_backend_sched{};\n''', 1)
s = s.replace('''        if (ggml_backend_dev_type(backends[b]->device) != GGML_BACKEND_DEVICE_TYPE_CPU) {\n            sched->staging[b].events[0] = ggml_backend_event_new(backends[b]->device);\n            sched->staging[b].events[1] = ggml_backend_event_new(backends[b]->device);\n        }\n''',
'''        if (ggml_backend_dev_type(backends[b]->device) != GGML_BACKEND_DEVICE_TYPE_CPU) {\n            for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {\n                sched->staging[b].events[slot] = ggml_backend_event_new(backends[b]->device);\n            }\n        }\n''', 1)
s = s.replace('''    for (int b = 0; b < sched->n_backends; b++) {\n        ggml_backend_sched_storage_prefetch_finish(sched->storage_prefetch[b]);\n        for (int c = 0; c < sched->n_copies; c++) {\n''',
'''    for (int b = 0; b < sched->n_backends; b++) {\n        for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {\n            ggml_backend_sched_storage_prefetch_release(sched, b, slot);\n        }\n        for (int c = 0; c < sched->n_copies; c++) {\n''', 1)
s = s.replace('''        for (int c = 0; c < 2; ++c) {\n            if (sched->staging[b].pending[c]) {\n                ggml_backend_event_synchronize(sched->staging[b].events[c]);\n            }\n            ggml_backend_buffer_free(sched->staging[b].buffers[c]);\n            ggml_backend_event_free(sched->staging[b].events[c]);\n        }\n''',
'''        for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {\n            if (sched->staging[b].pending[slot]) {\n                ggml_backend_event_synchronize(sched->staging[b].events[slot]);\n            }\n            ggml_backend_buffer_free(sched->staging[b].buffers[slot]);\n            ggml_backend_event_free(sched->staging[b].events[slot]);\n        }\n''', 1)
s = s.replace('''    delete sched->transient_sources_seen;\n    delete sched->residents;\n    free(sched);\n''',
'''    delete sched->transient_sources_seen;\n    delete sched->residents;\n    delete sched;\n''', 1)
s = s.replace('''    for (int b = 0; b < sched->n_backends; ++b) {\n        ggml_backend_sched_storage_prefetch_finish(sched->storage_prefetch[b]);\n        sched->storage_prefetch[b].active = false;\n    }\n''',
'''    for (int b = 0; b < sched->n_backends; ++b) {\n        for (int slot = 0; slot < GGML_BACKEND_SCHED_STAGING_SLOTS; ++slot) {\n            ggml_backend_sched_storage_prefetch_release(sched, b, slot);\n        }\n    }\n''', 1)
p.write_text(s)

# Residency is intentionally compatible with async storage/H2D prefetch: resident
# hits skip uploads, misses use the same bounded staging ring.
replace_once(
    'ggml/src/ggml-backend.cpp',
    '''    GGML_ASSERT(backend_id >= 0);\n    GGML_ASSERT(!sched->async_weight_prefetch);\n    if (!enabled) {\n''',
    '''    GGML_ASSERT(backend_id >= 0);\n    if (!enabled) {\n''')
replace_once(
    'ggml/src/ggml-backend.cpp',
    '''void ggml_backend_sched_set_weight_read_callback(\n        ggml_backend_sched_t sched, ggml_backend_sched_weight_read_callback callback, void * user_data) {\n    GGML_ASSERT(sched);\n    sched->weight_read_callback = callback;\n    sched->weight_read_callback_user_data = user_data;\n}\n''',
    '''void ggml_backend_sched_set_weight_read_callback(\n        ggml_backend_sched_t sched, ggml_backend_sched_weight_read_callback callback, void * user_data) {\n    GGML_ASSERT(sched);\n    sched->weight_read_callback = callback;\n    sched->weight_read_callback_user_data = user_data;\n}\n\nvoid ggml_backend_sched_set_weight_read_padded_callback(\n        ggml_backend_sched_t sched, ggml_backend_sched_weight_read_padded_callback callback, void * user_data) {\n    GGML_ASSERT(sched);\n    sched->weight_read_padded_callback = callback;\n    sched->weight_read_callback_user_data = user_data;\n}\n''')

# Wire direct-I/O model readers into every sequential scheduler instance, preserve
# residents across generation graph rebuilds, and keep mmap mode on the ordinary
# directly-addressable host path.
p = Path('src/llama-context.cpp')
s = p.read_text()
needle = '''        ggml_backend_sched_set_force_weight_offload(sched.get(), true);\n        ggml_backend_sched_set_async_weight_prefetch(sched.get(), true);\n'''
insert = '''        ggml_backend_sched_set_force_weight_offload(sched.get(), true);\n        ggml_backend_sched_set_async_weight_prefetch(sched.get(), true);\n        ggml_backend_sched_set_persistent_weight_residency(sched.get(), true);\n        if (model.is_sequential_direct_io()) {\n            ggml_backend_sched_set_weight_read_callback(sched.get(),\n                [](void * user_data, const void * logical_src, void * dst, size_t size) {\n                    return static_cast<const llama_model *>(user_data)->read_sequential_weight(logical_src, dst, size);\n                }, (void *) &model);\n            ggml_backend_sched_set_weight_read_padded_callback(sched.get(),\n                [](void * user_data, const void * logical_src, void * dst, size_t capacity, size_t size, size_t * data_offset) {\n                    return static_cast<const llama_model *>(user_data)->read_sequential_weight_padded(\n                        logical_src, dst, capacity, size, data_offset);\n                }, (void *) &model);\n        }\n'''
count = s.count(needle)
if count != 2:
    raise SystemExit(f'expected two sequential scheduler setup blocks, found {count}')
s = s.replace(needle, insert)
p.write_text(s)
