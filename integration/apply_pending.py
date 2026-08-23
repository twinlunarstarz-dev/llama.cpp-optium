from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    s = p.read_text()
    if old not in s:
        raise SystemExit(f'anchor not found in {path}: {old[:140]!r}')
    p.write_text(s.replace(old, new, 1))


# Grow a resident compact expert slab geometrically. The old payload stays marked
# executing while we admit/allocate the replacement, which prevents the global
# residency policy from evicting it before the device-to-device copy completes.
p = Path('ggml/src/ggml-backend.cpp')
s = p.read_text()
anchor = '''static bool ggml_backend_sched_ledger_valid(ggml_backend_sched_t sched, int backend_id) {\n'''
helper = r'''static bool ggml_backend_sched_grow_expert_slab(
        ggml_backend_sched_t sched,
        int backend_id,
        struct ggml_tensor * layout,
        ggml_backend_sched_resident_map::iterator slab_it,
        size_t active_slots,
        size_t max_slots) {
    auto & slab = slab_it->second;
    if (slab.executing || slab.experts.empty() || active_slots <= slab.experts.size() || max_slots == 0) {
        return active_slots <= slab.experts.size();
    }

    const size_t old_slots = slab.experts.size();
    size_t target_slots = std::max(active_slots, old_slots <= SIZE_MAX / 2 ? old_slots * 2 : max_slots);
    target_slots = std::min(target_slots, max_slots);
    if (target_slots <= old_slots || target_slots > (size_t) INT64_MAX) {
        return false;
    }

    const int64_t saved_ne2 = layout->ne[2];
    layout->ne[2] = (int64_t) target_slots;
    const size_t new_alloc_size = ggml_backend_buft_get_alloc_size(sched->bufts[backend_id], layout);
    layout->ne[2] = saved_ne2;
    if (new_alloc_size == 0 || new_alloc_size <= slab.allocation_size) {
        return false;
    }

    slab.executing = true;
    if (!ggml_backend_sched_make_resident_space(sched, backend_id, new_alloc_size)) {
        slab.executing = false;
        return false;
    }

    ggml_backend_buffer_t new_buffer = ggml_backend_buft_alloc_buffer(sched->bufts[backend_id], new_alloc_size);
    if (new_buffer == NULL) {
        slab.executing = false;
        return false;
    }

    // Copy only the initialized prefix corresponding to the old compact slots.
    // Both temporary descriptors have exactly the same layout, so the backend can
    // use its native D2D path. They do not own either backend buffer.
    layout->ne[2] = (int64_t) old_slots;
    struct ggml_tensor * old_view = ggml_dup_tensor_layout(sched->ctx, layout);
    struct ggml_tensor * new_view = ggml_dup_tensor_layout(sched->ctx, layout);
    layout->ne[2] = saved_ne2;
    old_view->flags = (enum ggml_tensor_flag) (old_view->flags | GGML_TENSOR_FLAG_NO_ALLOC);
    new_view->flags = (enum ggml_tensor_flag) (new_view->flags | GGML_TENSOR_FLAG_NO_ALLOC);

    const enum ggml_status old_ec = ggml_backend_tensor_alloc(
        slab.buffer, old_view, ggml_backend_buffer_get_base(slab.buffer));
    const enum ggml_status new_ec = ggml_backend_tensor_alloc(
        new_buffer, new_view, ggml_backend_buffer_get_base(new_buffer));
    if (old_ec != GGML_STATUS_SUCCESS || new_ec != GGML_STATUS_SUCCESS) {
        old_view->buffer = NULL;
        old_view->data = NULL;
        new_view->buffer = NULL;
        new_view->data = NULL;
        ggml_backend_buffer_free(new_buffer);
        slab.executing = false;
        return false;
    }

    ggml_backend_t backend = sched->backends[backend_id];
    ggml_backend_tensor_copy_async(backend, backend, old_view, new_view);
    ggml_backend_synchronize(backend);

    old_view->buffer = NULL;
    old_view->data = NULL;
    new_view->buffer = NULL;
    new_view->data = NULL;

    auto & row = sched->transient_metrics.backends[backend_id];
    GGML_ASSERT(row.current_resident_bytes >= slab.allocation_size);
    row.current_resident_bytes -= slab.allocation_size;
    row.current_resident_bytes += new_alloc_size;

    ggml_backend_buffer_t old_buffer = slab.buffer;
    slab.buffer = new_buffer;
    slab.allocation_size = new_alloc_size;
    slab.copy = NULL;
    slab.experts.resize(target_slots, -1);
    slab.expert_slots.resize(target_slots, -1);
    slab.expert_frequency.resize(target_slots, 0);
    slab.expert_completed_use.resize(target_slots, 0);
    slab.executing = false;
    slab.completed_use = ++sched->residency_use_clock;
    ggml_backend_buffer_free(old_buffer);
    ggml_backend_sched_resident_metrics_update(sched, backend_id);

    GGML_LOG_DEBUG("sequential compact slab grow: backend=%s old_slots=%zu new_slots=%zu old_bytes=%zu new_bytes=%zu\n",
        ggml_backend_name(backend), old_slots, target_slots,
        old_view ? ggml_nbytes(old_view) : 0, new_alloc_size);
    return true;
}

'''
if anchor not in s:
    raise SystemExit('resident ledger anchor not found')
s = s.replace(anchor, helper + anchor, 1)
p.write_text(s)

# Replace destructive undersized-slab eviction with D2D-preserving growth. If the
# temporary old+new allocation cannot be admitted, keep the previous safe fallback:
# evict the old slab and reconstitute a right-sized slab from storage.
replace_once(
    'ggml/src/ggml-backend.cpp',
    '''                        // A resident slab's shape is fixed by its initial allocation. A later graph\n                        // (notably a multi-token prompt after single-token warmup) can route to more\n                        // unique experts than that slab has slots. Replace the completed undersized\n                        // slab and allocate a larger compact slab instead of falling back to the full\n                        // expert tensor, which may not fit the sequential weight window.\n                        if (slab_it != sched->residents->end() && experts->size() > slab_it->second.experts.size()) {\n                            if (slab_it->second.executing) {\n                                valid = false;\n                            } else {\n                                ggml_backend_sched_evict_resident(sched, slab_it);\n                            }\n                        }\n''',
    '''                        // Grow compact capacity while preserving all currently resident expert\n                        // payloads on-device. If temporary old+new residency is impossible, fall\n                        // back to evict+reload rather than failing the graph.\n                        if (slab_it != sched->residents->end() && experts->size() > slab_it->second.experts.size()) {\n                            if (slab_it->second.executing) {\n                                valid = false;\n                            } else if (!ggml_backend_sched_grow_expert_slab(\n                                    sched, split_backend_id, input_cpy, slab_it, experts->size(), (size_t) source->ne[2])) {\n                                ggml_backend_sched_evict_resident(sched, slab_it);\n                            }\n                        }\n''')

# Empty slots created by geometric growth are free capacity, not evictions. Prefer
# them before LFU/LRU replacement and only count a real expert eviction when a
# populated slot is overwritten.
replace_once(
    'ggml/src/ggml-backend.cpp',
    '''                                size_t victim = slab.experts.size();\n                                for (size_t slot = 0; slot < slab.experts.size(); ++slot) {\n                                    if (!reserved[slot] && (victim == slab.experts.size() ||\n                                            slab.expert_frequency[slot] < slab.expert_frequency[victim] ||\n                                            (slab.expert_frequency[slot] == slab.expert_frequency[victim] &&\n                                             slab.expert_completed_use[slot] < slab.expert_completed_use[victim]))) {\n                                        victim = slot;\n                                    }\n                                }\n                                if (victim == slab.experts.size()) {\n                                    valid = false;\n                                    break;\n                                }\n                                ggml_backend_sched_counter_add(sched, &metrics.compact_expert_eviction_count, 1);\n                                slab.experts[victim] = (*experts)[i];\n''',
    '''                                size_t victim = slab.experts.size();\n                                for (size_t slot = 0; slot < slab.experts.size(); ++slot) {\n                                    if (reserved[slot]) {\n                                        continue;\n                                    }\n                                    if (slab.experts[slot] < 0) {\n                                        victim = slot;\n                                        break;\n                                    }\n                                    if (victim == slab.experts.size() ||\n                                            slab.expert_frequency[slot] < slab.expert_frequency[victim] ||\n                                            (slab.expert_frequency[slot] == slab.expert_frequency[victim] &&\n                                             slab.expert_completed_use[slot] < slab.expert_completed_use[victim])) {\n                                        victim = slot;\n                                    }\n                                }\n                                if (victim == slab.experts.size()) {\n                                    valid = false;\n                                    break;\n                                }\n                                if (slab.experts[victim] >= 0) {\n                                    ggml_backend_sched_counter_add(sched, &metrics.compact_expert_eviction_count, 1);\n                                }\n                                slab.experts[victim] = (*experts)[i];\n''')

# A resident slab with spare geometric capacity can have more slots than the
# current active expert set. Report only populated experts when draining/evicting.
replace_once(
    'ggml/src/ggml-backend.cpp',
    '''    ggml_backend_sched_counter_add(sched, &row.residency_eviction_count, 1);\n    ggml_backend_sched_counter_add(sched, &row.compact_expert_eviction_count, resident.experts.size());\n''',
    '''    ggml_backend_sched_counter_add(sched, &row.residency_eviction_count, 1);\n    ggml_backend_sched_counter_add(sched, &row.compact_expert_eviction_count,\n        std::count_if(resident.experts.begin(), resident.experts.end(), [](int32_t expert) { return expert >= 0; }));\n''')

# Remove the one-shot phase2 patcher helper after its changes have landed.
Path('integration/fix_phase2_patcher.py').unlink(missing_ok=True)
