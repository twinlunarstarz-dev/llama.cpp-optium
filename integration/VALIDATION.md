# Sequential port validation

- driver: v11-final
- configure rc: 0
- build rc: 2
- focused tests rc: 99

## Build
```text
[  0%] Building CXX object vendor/hash/CMakeFiles/vendor-hash.dir/hash.cpp.o
[  1%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml.c.o
[  1%] Building C object vendor/hash/CMakeFiles/vendor-hash.dir/xxhash/xxhash.c.o
[  1%] Building CXX object vendor/hash/CMakeFiles/vendor-hash.dir/sha1/sha1.c.o
[  1%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml.cpp.o
[  1%] Building C object vendor/hash/CMakeFiles/vendor-hash.dir/sha256/sha256.c.o
[  2%] Linking CXX static library libvendor-hash.a
[  2%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml-alloc.c.o
[  2%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-backend.cpp.o
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:838:10: error: ‘unordered_set’ in namespace ‘std’ does not name a template type
  838 |     std::unordered_set<const struct ggml_tensor *> * transient_sources_seen;
      |          ^~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:23:1: note: ‘std::unordered_set’ is defined in header ‘<unordered_set>’; did you forget to ‘#include <unordered_set>’?
   22 | #include <algorithm>
  +++ |+#include <unordered_set>
   23 | #include <vector>
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:841:10: error: ‘unordered_map’ in namespace ‘std’ does not name a template type
  841 |     std::unordered_map<const struct ggml_tensor *, ggml_backend_sched_resident> * residents;
      |          ^~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:23:1: note: ‘std::unordered_map’ is defined in header ‘<unordered_map>’; did you forget to ‘#include <unordered_map>’?
   22 | #include <algorithm>
  +++ |+#include <unordered_map>
   23 | #include <vector>
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: In function ‘void ggml_backend_sched_split_inputs_grow(ggml_backend_sched_split*)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:871:19: error: incompatible types in assignment of ‘ggml_tensor**’ to ‘ggml_tensor* [30]’
  871 |     split->inputs = pnew;
      |     ~~~~~~~~~~~~~~^~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: In function ‘void ggml_backend_sched_weight_upload_chunked(ggml_backend_sched_t, ggml_backend_t, int, ggml_tensor*, const void*, size_t, size_t, bool)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:911:13: error: ‘ggml_backend_sched_counter_add’ was not declared in this scope; did you mean ‘ggml_backend_sched_split_graph’?
  911 |             ggml_backend_sched_counter_add(sched, &metrics.upload_chunk_count, 1);
      |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |             ggml_backend_sched_split_graph
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: At global scope:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:977:9: error: ‘std::unordered_map’ has not been declared
  977 |         std::unordered_map<const struct ggml_tensor *, ggml_backend_sched_resident>::iterator it) {
      |         ^~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:977:27: error: expected ‘,’ or ‘...’ before ‘<’ token
  977 |         std::unordered_map<const struct ggml_tensor *, ggml_backend_sched_resident>::iterator it) {
      |                           ^
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: In function ‘void ggml_backend_sched_evict_resident(ggml_backend_sched_t, int)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:978:5: error: ‘ggml_backend_sched_resident’ was not declared in this scope; did you mean ‘ggml_backend_sched_reset’?
  978 |     ggml_backend_sched_resident resident = it->second;
      |     ^~~~~~~~~~~~~~~~~~~~~~~~~~~
      |     ggml_backend_sched_reset
In file included from /home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/../include/ggml-backend.h:3,
                 from /home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:11:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:979:18: error: ‘resident’ was not declared in this scope
  979 |     GGML_ASSERT(!resident.executing);
      |                  ^~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/../include/ggml.h:288:30: note: in definition of macro ‘GGML_ASSERT’
  288 | #define GGML_ASSERT(x) if (!(x)) GGML_ABORT("GGML_ASSERT(%s) failed", #x)
      |                              ^
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:980:46: error: ‘resident’ was not declared in this scope
  980 |     ggml_backend_synchronize(sched->backends[resident.backend_id]);
      |                                              ^~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:989:12: error: ‘struct ggml_backend_sched’ has no member named ‘residents’
  989 |     sched->residents->erase(it);
      |            ^~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:989:29: error: ‘it’ was not declared in this scope; did you mean ‘int’?
  989 |     sched->residents->erase(it);
      |                             ^~
      |                             int
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:991:5: error: ‘ggml_backend_sched_counter_add’ was not declared in this scope; did you mean ‘ggml_backend_sched_split_graph’?
  991 |     ggml_backend_sched_counter_add(sched, &row.residency_eviction_count, 1);
      |     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |     ggml_backend_sched_split_graph
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: In function ‘void ggml_backend_sched_drain_residents(ggml_backend_sched_t)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:996:19: error: ‘struct ggml_backend_sched’ has no member named ‘residents’
  996 |     while (sched->residents != NULL && !sched->residents->empty()) {
      |                   ^~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:996:48: error: ‘struct ggml_backend_sched’ has no member named ‘residents’
  996 |     while (sched->residents != NULL && !sched->residents->empty()) {
      |                                                ^~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:997:26: error: ‘struct ggml_backend_sched’ has no member named ‘residents’
  997 |         auto it = sched->residents->begin();
      |                          ^~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1001:9: error: ‘ggml_backend_sched_counter_add’ was not declared in this scope; did you mean ‘ggml_backend_sched_split_graph’?
 1001 |         ggml_backend_sched_counter_add(sched, &sched->transient_metrics.backends[backend_id].residency_drain_count, 1);
      |         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |         ggml_backend_sched_split_graph
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: In function ‘bool ggml_backend_sched_make_resident_space(ggml_backend_sched_t, int, size_t)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1012:30: error: ‘struct ggml_backend_sched’ has no member named ‘residents’
 1012 |         auto victim = sched->residents->end();
      |                              ^~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1013:31: error: ‘struct ggml_backend_sched’ has no member named ‘residents’
 1013 |         for (auto it = sched->residents->begin(); it != sched->residents->end(); ++it) {
      |                               ^~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1013:64: error: ‘struct ggml_backend_sched’ has no member named ‘residents’
 1013 |         for (auto it = sched->residents->begin(); it != sched->residents->end(); ++it) {
      |                                                                ^~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1015:39: error: ‘struct ggml_backend_sched’ has no member named ‘residents’
 1015 |                     (victim == sched->residents->end() || it->second.completed_use < victim->second.completed_use)) {
      |                                       ^~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1019:30: error: ‘struct ggml_backend_sched’ has no member named ‘residents’
 1019 |         if (victim == sched->residents->end()) {
      |                              ^~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: In function ‘void ggml_backend_sched_ledger_assert(ggml_backend_sched_t, int, bool)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1046:9: error: ‘ggml_backend_sched_counter_add’ was not declared in this scope; did you mean ‘ggml_backend_sched_ledger_valid’?
 1046 |         ggml_backend_sched_counter_add(sched, &sched->transient_metrics.ledger_mismatch_count, 1);
      |         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |         ggml_backend_sched_ledger_valid
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: In function ‘void ggml_backend_sched_ledger_enter(ggml_backend_sched_t, int, size_t)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1056:9: error: ‘ggml_backend_sched_counter_add’ was not declared in this scope; did you mean ‘ggml_backend_sched_ledger_valid’?
 1056 |         ggml_backend_sched_counter_add(sched, &sched->transient_metrics.ledger_mismatch_count, 1);
      |         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |         ggml_backend_sched_ledger_valid
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: In function ‘void ggml_backend_sched_release_transients(ggml_backend_sched_t, ggml_backend_sched_split*, bool, ggml_backend_sched_transient_drain_reason, bool)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1093:39: error: ‘struct ggml_backend_sched_split’ has no member named ‘transient_buffers’
 1093 |         has_live = has_live || split->transient_buffers[i] != NULL;
      |                                       ^~~~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1104:13: error: ‘ggml_backend_sched_counter_add’ was not declared in this scope; did you mean ‘ggml_backend_sched_ledger_valid’?
 1104 |             ggml_backend_sched_counter_add(sched, &row.compute_completion_wait_count, 1);
      |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |             ggml_backend_sched_ledger_valid
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1109:20: error: ‘struct ggml_backend_sched_split’ has no member named ‘input_resident’
 1109 |         if (split->input_resident[i]) {
      |                    ^~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1112:47: error: ‘struct ggml_backend_sched_split’ has no member named ‘transient_buffers’
 1112 |         ggml_backend_buffer_t buffer = split->transient_buffers[i];
      |                                               ^~~~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1124:16: error: ‘struct ggml_backend_sched_split’ has no member named ‘transient_buffers’
 1124 |         split->transient_buffers[i] = NULL;
      |                ^~~~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1125:74: error: ‘struct ggml_backend_sched_split’ has no member named ‘transient_sizes’
 1125 |         ggml_backend_sched_ledger_leave(sched, split->backend_id, split->transient_sizes[i]);
      |                                                                          ^~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1126:16: error: ‘struct ggml_backend_sched_split’ has no member named ‘transient_sizes’
 1126 |         split->transient_sizes[i] = 0;
      |                ^~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1129:5: error: ‘ggml_backend_sched_counter_add’ was not declared in this scope; did you mean ‘ggml_backend_sched_ledger_valid’?
 1129 |     ggml_backend_sched_counter_add(sched, &row.drain_count[reason], 1);
      |     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |     ggml_backend_sched_ledger_valid
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: In function ‘void ggml_backend_sched_drain_transients(ggml_backend_sched_t, ggml_backend_sched_transient_drain_reason)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1139:9: error: ‘ggml_backend_sched_counter_add’ was not declared in this scope; did you mean ‘ggml_backend_sched_ledger_valid’?
 1139 |         ggml_backend_sched_counter_add(sched, &sched->transient_metrics.ledger_mismatch_count, 1);
      |         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |         ggml_backend_sched_ledger_valid
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1142:27: error: ‘pnew’ was not declared in this scope
 1142 |     sched->graph_inputs = pnew;
      |                           ^~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:1143:36: error: ‘new_cap’ was not declared in this scope
 1143 |     sched->graph_inputs_capacity = new_cap;
      |                                    ^~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: In function ‘ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2060:13: error: ‘execution_instrumented’ was not declared in this scope
 2060 |         if (execution_instrumented) {
      |             ^~~~~~~~~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2061:13: error: ‘ggml_backend_sched_counter_add’ was not declared in this scope; did you mean ‘ggml_backend_sched_ledger_valid’?
 2061 |             ggml_backend_sched_counter_add(sched, &metrics.splits_seen_count, 1);
      |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |             ggml_backend_sched_ledger_valid
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2064:25: error: ‘struct ggml_backend_sched_split’ has no member named ‘input_transient’
 2064 |             if (!split->input_transient[input_id]) {
      |                         ^~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2068:66: error: ‘GGML_TENSOR_FLAG_NO_ALLOC’ was not declared in this scope; did you mean ‘GGML_TENSOR_FLAG_PARAM’?
 2068 |             GGML_ASSERT(input_cpy != NULL && (input_cpy->flags & GGML_TENSOR_FLAG_NO_ALLOC));
      |                                                                  ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/../include/ggml.h:288:30: note: in definition of macro ‘GGML_ASSERT’
  288 | #define GGML_ASSERT(x) if (!(x)) GGML_ABORT("GGML_ASSERT(%s) failed", #x)
      |                              ^
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2072:37: error: ‘struct ggml_backend_sched’ has no member named ‘residents’
 2072 |                 auto found = sched->residents->find(source);
      |                                     ^~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2073:37: error: ‘struct ggml_backend_sched’ has no member named ‘residents’
 2073 |                 if (found != sched->residents->end()) {
      |                                     ^~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2079:32: error: ‘struct ggml_backend_sched_split’ has no member named ‘input_resident’
 2079 |                         split->input_resident[input_id] = true;
      |                                ^~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2080:32: error: ‘struct ggml_backend_sched_split’ has no member named ‘input_resident_hit’
 2080 |                         split->input_resident_hit[input_id] = true;
      |                                ^~~~~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2081:25: error: ‘ggml_backend_sched_counter_add’ was not declared in this scope; did you mean ‘ggml_backend_sched_ledger_valid’?
 2081 |                         ggml_backend_sched_counter_add(sched, &metrics.residency_hit_count, 1);
      |                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                         ggml_backend_sched_ledger_valid
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2089:13: error: ‘ggml_backend_sched_counter_add’ was not declared in this scope; did you mean ‘ggml_backend_sched_ledger_valid’?
 2089 |             ggml_backend_sched_counter_add(sched, &metrics.allocation_requested_bytes, alloc_size);
      |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |             ggml_backend_sched_ledger_valid
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2136:20: error: ‘struct ggml_backend_sched_split’ has no member named ‘transient_buffers’
 2136 |             split->transient_buffers[input_id] = buffer;
      |                    ^~~~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2137:20: error: ‘struct ggml_backend_sched_split’ has no member named ‘transient_sizes’
 2137 |             split->transient_sizes[input_id] = alloc_size;
      |                    ^~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2139:17: error: ‘ggml_backend_sched_resident’ was not declared in this scope; did you mean ‘ggml_backend_sched_reset’?
 2139 |                 ggml_backend_sched_resident resident{};
      |                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                 ggml_backend_sched_reset
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2140:17: error: ‘resident’ was not declared in this scope
 2140 |                 resident.source = source;
      |                 ^~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2149:24: error: ‘struct ggml_backend_sched’ has no member named ‘residents’
 2149 |                 sched->residents->emplace(source, resident);
      |                        ^~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2152:24: error: ‘struct ggml_backend_sched_split’ has no member named ‘input_resident’
 2152 |                 split->input_resident[input_id] = true;
      |                        ^~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2167:13: error: ‘ggml_backend_sched_counter_add’ was not declared in this scope; did you mean ‘ggml_backend_sched_ledger_valid’?
 2167 |             ggml_backend_sched_counter_add(sched, &metrics.transient_split_count, 1);
      |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |             ggml_backend_sched_ledger_valid
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2170:51: error: ‘struct ggml_backend_sched_split’ has no member named ‘transient_buffers’
 2170 |             ggml_backend_buffer_t buffer = split->transient_buffers[input_id];
      |                                                   ^~~~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2179:17: error: ‘ggml_backend_sched_counter_add’ was not declared in this scope; did you mean ‘ggml_backend_sched_ledger_valid’?
 2179 |                 ggml_backend_sched_counter_add(sched, &sched->transient_metrics.graph_compute_failure_count, 1);
      |                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                 ggml_backend_sched_ledger_valid
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: In lambda function:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2270:68: error: ‘struct ggml_backend_sched_split’ has no member named ‘input_transient’
 2270 |                             expert_size_copy + padding_end, split->input_transient[input_id]);
      |                                                                    ^~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: In function ‘ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2296:32: error: ‘struct ggml_backend_sched_split’ has no member named ‘input_transient’
 2296 |                     if (split->input_transient[input_id]) {
      |                                ^~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2297:25: error: ‘ggml_backend_sched_counter_add’ was not declared in this scope; did you mean ‘ggml_backend_sched_ledger_valid’?
 2297 |                         ggml_backend_sched_counter_add(sched, &metrics.upload_count, 1);
      |                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                         ggml_backend_sched_ledger_valid
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2298:103: error: ‘struct ggml_backend_sched_split’ has no member named ‘transient_sizes’
 2298 |                         ggml_backend_sched_counter_add(sched, &metrics.uploaded_backend_bytes, split->transient_sizes[input_id]);
      |                                                                                                       ^~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2299:37: error: ‘struct ggml_backend_sched’ has no member named ‘transient_sources_seen’
 2299 |                         if (!sched->transient_sources_seen->insert(input).second) {
      |                                     ^~~~~~~~~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2302:36: error: ‘struct ggml_backend_sched_split’ has no member named ‘input_resident’
 2302 |                         if (split->input_resident[input_id]) {
      |                                    ^~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2313:87: error: ‘struct ggml_backend_sched_split’ has no member named ‘input_transient’
 2313 |                             input_cpy, input->data, 0, ggml_nbytes(input_cpy), split->input_transient[input_id]);
      |                                                                                       ^~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2314:36: error: ‘struct ggml_backend_sched_split’ has no member named ‘input_transient’
 2314 |                         if (split->input_transient[input_id]) {
      |                                    ^~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2315:29: error: ‘ggml_backend_sched_counter_add’ was not declared in this scope; did you mean ‘ggml_backend_sched_ledger_valid’?
 2315 |                             ggml_backend_sched_counter_add(sched, &metrics.upload_count, 1);
      |                             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                             ggml_backend_sched_ledger_valid
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2316:107: error: ‘struct ggml_backend_sched_split’ has no member named ‘transient_sizes’
 2316 |                             ggml_backend_sched_counter_add(sched, &metrics.uploaded_backend_bytes, split->transient_sizes[input_id]);
      |                                                                                                           ^~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2317:41: error: ‘struct ggml_backend_sched’ has no member named ‘transient_sources_seen’
 2317 |                             if (!sched->transient_sources_seen->insert(input).second) {
      |                                         ^~~~~~~~~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2320:40: error: ‘struct ggml_backend_sched_split’ has no member named ‘input_resident’
 2320 |                             if (split->input_resident[input_id]) {
      |                                        ^~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: In function ‘ggml_backend_sched* ggml_backend_sched_new(ggml_backend**, ggml_backend_buffer_type**, int, size_t, bool, bool)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2417:12: error: ‘struct ggml_backend_sched’ has no member named ‘transient_sources_seen’
 2417 |     sched->transient_sources_seen = new std::unordered_set<const struct ggml_tensor *>();
      |            ^~~~~~~~~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2417:46: error: ‘unordered_set’ in namespace ‘std’ does not name a template type
 2417 |     sched->transient_sources_seen = new std::unordered_set<const struct ggml_tensor *>();
      |                                              ^~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2417:41: note: ‘std::unordered_set’ is defined in header ‘<unordered_set>’; did you forget to ‘#include <unordered_set>’?
 2417 |     sched->transient_sources_seen = new std::unordered_set<const struct ggml_tensor *>();
      |                                         ^~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2417:60: error: expected primary-expression before ‘const’
 2417 |     sched->transient_sources_seen = new std::unordered_set<const struct ggml_tensor *>();
      |                                                            ^~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2418:12: error: ‘struct ggml_backend_sched’ has no member named ‘residents’
 2418 |     sched->residents = new std::unordered_map<const struct ggml_tensor *, ggml_backend_sched_resident>();
      |            ^~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2418:33: error: ‘unordered_map’ in namespace ‘std’ does not name a template type
 2418 |     sched->residents = new std::unordered_map<const struct ggml_tensor *, ggml_backend_sched_resident>();
      |                                 ^~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2418:28: note: ‘std::unordered_map’ is defined in header ‘<unordered_map>’; did you forget to ‘#include <unordered_map>’?
 2418 |     sched->residents = new std::unordered_map<const struct ggml_tensor *, ggml_backend_sched_resident>();
      |                            ^~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2418:47: error: expected primary-expression before ‘const’
 2418 |     sched->residents = new std::unordered_map<const struct ggml_tensor *, ggml_backend_sched_resident>();
      |                                               ^~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: At global scope:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2619:6: warning: no previous declaration for ‘void ggml_backend_sched_set_weight_residency(ggml_backend_sched_t, ggml_backend_t, bool)’ [-Wmissing-declarations]
 2619 | void ggml_backend_sched_set_weight_residency(
      |      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: In function ‘void ggml_backend_sched_set_weight_residency(ggml_backend_sched_t, ggml_backend_t, bool)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2626:31: error: ‘struct ggml_backend_sched’ has no member named ‘residents’
 2626 |         for (auto it = sched->residents->begin(); it != sched->residents->end();) {
      |                               ^~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2626:64: error: ‘struct ggml_backend_sched’ has no member named ‘residents’
 2626 |         for (auto it = sched->residents->begin(); it != sched->residents->end();) {
      |                                                                ^~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2634:13: error: ‘ggml_backend_sched_counter_add’ was not declared in this scope; did you mean ‘ggml_backend_sched_ledger_valid’?
 2634 |             ggml_backend_sched_counter_add(sched,
      |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |             ggml_backend_sched_ledger_valid
gmake[2]: *** [ggml/src/CMakeFiles/ggml-base.dir/build.make:121: ggml/src/CMakeFiles/ggml-base.dir/ggml-backend.cpp.o] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:2487: ggml/src/CMakeFiles/ggml-base.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
[  2%] Built target vendor-hash
gmake: *** [Makefile:146: all] Error 2
```
