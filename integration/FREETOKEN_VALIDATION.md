# FreeToken sequential v1 validation

- configure rc: 0
- build rc: 2

```text
[  1%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml.c.o
[  1%] Building CXX object vendor/hash/CMakeFiles/vendor-hash.dir/hash.cpp.o
[  1%] Building C object vendor/hash/CMakeFiles/vendor-hash.dir/xxhash/xxhash.c.o
[  1%] Building CXX object vendor/hash/CMakeFiles/vendor-hash.dir/sha1/sha1.c.o
[  1%] Building C object vendor/hash/CMakeFiles/vendor-hash.dir/sha256/sha256.c.o
[  2%] Linking CXX static library libvendor-hash.a
[  2%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml.cpp.o
[  2%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml-alloc.c.o
[  2%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-backend.cpp.o
[  2%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-backend-meta.cpp.o
[  3%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-opt.cpp.o
[  3%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-threading.cpp.o
[  3%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml-quants.c.o
[  3%] Built target vendor-hash
[  3%] Building CXX object vendor/cpp-httplib/CMakeFiles/cpp-httplib.dir/httplib.cpp.o
[  3%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/gguf.cpp.o
[  3%] Linking CXX shared library ../../bin/libggml-base.so
[  3%] Built target ggml-base
[  3%] Building CXX object common/CMakeFiles/llama-common-base.dir/build-info.cpp.o
[  3%] Linking CXX static library libllama-common-base.a
[  3%] Built target llama-common-base
[  4%] Building CXX object tools/mtmd/CMakeFiles/llama-llava-cli.dir/deprecation-warning.cpp.o
[  4%] Linking CXX executable ../../bin/llama-llava-cli
[  4%] Built target llama-llava-cli
[  4%] Building CXX object tools/mtmd/CMakeFiles/llama-gemma3-cli.dir/deprecation-warning.cpp.o
[  4%] Linking CXX executable ../../bin/llama-gemma3-cli
[  4%] Built target llama-gemma3-cli
[  4%] Building CXX object tools/mtmd/CMakeFiles/llama-minicpmv-cli.dir/deprecation-warning.cpp.o
[  4%] Linking CXX executable ../../bin/llama-minicpmv-cli
[  4%] Built target llama-minicpmv-cli
[  4%] Building CXX object tools/mtmd/CMakeFiles/llama-qwen2vl-cli.dir/deprecation-warning.cpp.o
[  4%] Linking CXX executable ../../bin/llama-qwen2vl-cli
[  4%] Built target llama-qwen2vl-cli
[  5%] Building C object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu.c.o
[  5%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu.cpp.o
[  5%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/repack.cpp.o
[  5%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/hbm.cpp.o
[  5%] Building C object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/quants.c.o
[  5%] Linking CXX static library libcpp-httplib.a
[  6%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/traits.cpp.o
[  6%] Built target cpp-httplib
[  6%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/amx/amx.cpp.o
[  6%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/amx/mmq.cpp.o
[  6%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/binary-ops.cpp.o
[  6%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/unary-ops.cpp.o
[  7%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/vec.cpp.o
[  7%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ops.cpp.o
[  7%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/llamafile/sgemm.cpp.o
[  7%] Building C object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/x86/quants.c.o
[  7%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/x86/repack.cpp.o
[  8%] Linking CXX shared library ../../bin/libggml-cpu.so
[  8%] Built target ggml-cpu
[  8%] Building CXX object ggml/src/CMakeFiles/ggml.dir/ggml-backend-dl.cpp.o
[  8%] Building CXX object ggml/src/CMakeFiles/ggml.dir/ggml-backend-reg.cpp.o
[  8%] Linking CXX shared library ../../bin/libggml.so
[  8%] Built target ggml
[  8%] Building CXX object examples/gguf-hash/CMakeFiles/llama-gguf-hash.dir/gguf-hash.cpp.o
[  8%] Building CXX object src/CMakeFiles/llama.dir/llama.cpp.o
[  8%] Linking CXX executable ../../bin/llama-gguf-hash
[  8%] Built target llama-gguf-hash
[  9%] Building CXX object examples/gguf/CMakeFiles/llama-gguf.dir/gguf.cpp.o
[  9%] Linking CXX executable ../../bin/llama-gguf
[  9%] Built target llama-gguf
[  9%] Building CXX object src/CMakeFiles/llama.dir/llama-adapter.cpp.o
[  9%] Building CXX object src/CMakeFiles/llama.dir/llama-arch.cpp.o
[ 10%] Building CXX object src/CMakeFiles/llama.dir/llama-batch.cpp.o
[ 10%] Building CXX object src/CMakeFiles/llama.dir/llama-chat.cpp.o
[ 10%] Building CXX object src/CMakeFiles/llama.dir/llama-context.cpp.o
[ 10%] Building CXX object src/CMakeFiles/llama.dir/llama-cparams.cpp.o
[ 10%] Building CXX object src/CMakeFiles/llama.dir/llama-grammar.cpp.o
/home/runner/work/llama.cpp-optium/llama.cpp-optium/src/llama-context.cpp: In member function ‘void llama_context::sched_reserve()’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/src/llama-context.cpp:678:76: error: invalid conversion from ‘size_t’ {aka ‘long unsigned int’} to ‘ggml_backend_t’ {aka ‘ggml_backend*’} [-fpermissive]
  678 |             ggml_backend_sched_set_max_weight_bytes_per_split(sched.get(), sequential_weight_budget);
      |                                                                            ^~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                            |
      |                                                                            size_t {aka long unsigned int}
/home/runner/work/llama.cpp-optium/llama.cpp-optium/src/llama-context.cpp:678:62: error: too few arguments to function ‘void ggml_backend_sched_set_max_weight_bytes_per_split(ggml_backend_sched_t, ggml_backend_t, size_t)’
  678 |             ggml_backend_sched_set_max_weight_bytes_per_split(sched.get(), sequential_weight_budget);
      |             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In file included from /home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/../include/ggml-cpu.h:4,
                 from /home/runner/work/llama.cpp-optium/llama.cpp-optium/src/../include/llama.h:5,
                 from /home/runner/work/llama.cpp-optium/llama.cpp-optium/src/llama-context.h:3,
                 from /home/runner/work/llama.cpp-optium/llama.cpp-optium/src/llama-context.cpp:1:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/../include/ggml-backend.h:384:35: note: declared here
  384 |     GGML_API void                 ggml_backend_sched_set_max_weight_bytes_per_split(
      |                                   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/src/llama-context.cpp:720:88: error: invalid conversion from ‘size_t’ {aka ‘long unsigned int’} to ‘ggml_backend_t’ {aka ‘ggml_backend*’} [-fpermissive]
  720 |                         ggml_backend_sched_set_max_weight_bytes_per_split(sched.get(), sequential_weight_budget);
      |                                                                                        ^~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                                        |
      |                                                                                        size_t {aka long unsigned int}
/home/runner/work/llama.cpp-optium/llama.cpp-optium/src/llama-context.cpp:720:74: error: too few arguments to function ‘void ggml_backend_sched_set_max_weight_bytes_per_split(ggml_backend_sched_t, ggml_backend_t, size_t)’
  720 |                         ggml_backend_sched_set_max_weight_bytes_per_split(sched.get(), sequential_weight_budget);
      |                         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/../include/ggml-backend.h:384:35: note: declared here
  384 |     GGML_API void                 ggml_backend_sched_set_max_weight_bytes_per_split(
      |                                   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
gmake[2]: *** [src/CMakeFiles/llama.dir/build.make:149: src/CMakeFiles/llama.dir/llama-context.cpp.o] Error 1
gmake[2]: *** Waiting for unfinished jobs....
gmake[1]: *** [CMakeFiles/Makefile2:2502: src/CMakeFiles/llama.dir/all] Error 2
gmake: *** [Makefile:146: all] Error 2
```
