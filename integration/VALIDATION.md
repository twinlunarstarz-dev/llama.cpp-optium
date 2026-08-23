# Sequential scheduler repair validation

- driver: v12
- scheduler source: current fork master
- CUDA source: current upstream integration
- configure rc: 0
- build rc: 2
- focused tests rc: 99

## Build
```text
[  1%] Building CXX object vendor/hash/CMakeFiles/vendor-hash.dir/hash.cpp.o
[  1%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml.c.o
[  1%] Building C object vendor/hash/CMakeFiles/vendor-hash.dir/xxhash/xxhash.c.o
[  1%] Building CXX object vendor/hash/CMakeFiles/vendor-hash.dir/sha1/sha1.c.o
[  1%] Building C object vendor/hash/CMakeFiles/vendor-hash.dir/sha256/sha256.c.o
[  1%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml.cpp.o
[  2%] Linking CXX static library libvendor-hash.a
[  2%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml-alloc.c.o
[  2%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-backend.cpp.o
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: In function ‘void ggml_backend_sched_split_graph(ggml_backend_sched_t, ggml_cgraph*)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2415:100: error: ‘GGML_TENSOR_FLAG_NO_ALLOC’ was not declared in this scope; did you mean ‘GGML_TENSOR_FLAG_PARAM’?
 2415 |                                 tensor_copy->flags = (enum ggml_tensor_flag) (tensor_copy->flags | GGML_TENSOR_FLAG_NO_ALLOC);
      |                                                                                                    ^~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                                    GGML_TENSOR_FLAG_PARAM
In file included from /home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/../include/ggml-backend.h:3,
                 from /home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:11:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp: In function ‘ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2705:66: error: ‘GGML_TENSOR_FLAG_NO_ALLOC’ was not declared in this scope; did you mean ‘GGML_TENSOR_FLAG_PARAM’?
 2705 |             GGML_ASSERT(input_cpy != NULL && (input_cpy->flags & GGML_TENSOR_FLAG_NO_ALLOC));
      |                                                                  ^~~~~~~~~~~~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/../include/ggml.h:288:30: note: in definition of macro ‘GGML_ASSERT’
  288 | #define GGML_ASSERT(x) if (!(x)) GGML_ABORT("GGML_ASSERT(%s) failed", #x)
      |                              ^
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend.cpp:2826:86: error: ‘GGML_TENSOR_FLAG_NO_ALLOC’ was not declared in this scope; did you mean ‘GGML_TENSOR_FLAG_PARAM’?
 2826 |                         ids_copy->flags = (enum ggml_tensor_flag) (ids_copy->flags | GGML_TENSOR_FLAG_NO_ALLOC);
      |                                                                                      ^~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                      GGML_TENSOR_FLAG_PARAM
gmake[2]: *** [ggml/src/CMakeFiles/ggml-base.dir/build.make:121: ggml/src/CMakeFiles/ggml-base.dir/ggml-backend.cpp.o] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:2487: ggml/src/CMakeFiles/ggml-base.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
[  2%] Built target vendor-hash
gmake: *** [Makefile:146: all] Error 2
```
