# FreeToken sequential continuation validation

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
[  2%] Linking CXX static library libvendor-hash.a
[  2%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml.cpp.o
[  2%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml-alloc.c.o
[  2%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-backend.cpp.o
[  2%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-backend-meta.cpp.o
[  2%] Built target vendor-hash
[  2%] Building CXX object vendor/cpp-httplib/CMakeFiles/cpp-httplib.dir/httplib.cpp.o
[  3%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-opt.cpp.o
[  3%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-threading.cpp.o
[  3%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml-quants.c.o
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
[  5%] Linking CXX static library libcpp-httplib.a
[  5%] Built target cpp-httplib
[  5%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/hbm.cpp.o
[  5%] Building C object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/quants.c.o
[  6%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/traits.cpp.o
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
[ 11%] Building CXX object src/CMakeFiles/llama.dir/llama-graph.cpp.o
[ 11%] Building CXX object src/CMakeFiles/llama.dir/llama-hparams.cpp.o
[ 11%] Building CXX object src/CMakeFiles/llama.dir/llama-impl.cpp.o
[ 11%] Building CXX object src/CMakeFiles/llama.dir/llama-io.cpp.o
[ 11%] Building CXX object src/CMakeFiles/llama.dir/llama-kv-cache.cpp.o
[ 12%] Building CXX object src/CMakeFiles/llama.dir/llama-kv-cache-iswa.cpp.o
[ 12%] Building CXX object src/CMakeFiles/llama.dir/llama-kv-cache-dsa.cpp.o
[ 12%] Building CXX object src/CMakeFiles/llama.dir/llama-kv-cache-dsa-iswa.cpp.o
[ 12%] Building CXX object src/CMakeFiles/llama.dir/llama-kv-cache-msa.cpp.o
[ 12%] Building CXX object src/CMakeFiles/llama.dir/llama-kv-cache-dsv4.cpp.o
[ 13%] Building CXX object src/CMakeFiles/llama.dir/llama-memory.cpp.o
[ 13%] Building CXX object src/CMakeFiles/llama.dir/llama-memory-hybrid.cpp.o
[ 13%] Building CXX object src/CMakeFiles/llama.dir/llama-memory-hybrid-iswa.cpp.o
[ 13%] Building CXX object src/CMakeFiles/llama.dir/llama-memory-recurrent.cpp.o
[ 13%] Building CXX object src/CMakeFiles/llama.dir/llama-mmap.cpp.o
[ 14%] Building CXX object src/CMakeFiles/llama.dir/llama-model-loader.cpp.o
[ 14%] Building CXX object src/CMakeFiles/llama.dir/llama-model-saver.cpp.o
[ 14%] Building CXX object src/CMakeFiles/llama.dir/llama-model.cpp.o
[ 14%] Building CXX object src/CMakeFiles/llama.dir/llama-quant.cpp.o
[ 14%] Building CXX object src/CMakeFiles/llama.dir/llama-sampler.cpp.o
[ 15%] Building CXX object src/CMakeFiles/llama.dir/llama-vocab.cpp.o
[ 15%] Building CXX object src/CMakeFiles/llama.dir/unicode-data.cpp.o
[ 15%] Building CXX object src/CMakeFiles/llama.dir/unicode.cpp.o
[ 15%] Building CXX object src/CMakeFiles/llama.dir/models/afmoe.cpp.o
[ 15%] Building CXX object src/CMakeFiles/llama.dir/models/apertus.cpp.o
[ 16%] Building CXX object src/CMakeFiles/llama.dir/models/arcee.cpp.o
[ 16%] Building CXX object src/CMakeFiles/llama.dir/models/arctic.cpp.o
[ 16%] Building CXX object src/CMakeFiles/llama.dir/models/arwkv7.cpp.o
[ 16%] Building CXX object src/CMakeFiles/llama.dir/models/baichuan.cpp.o
[ 16%] Building CXX object src/CMakeFiles/llama.dir/models/bailingmoe.cpp.o
[ 16%] Building CXX object src/CMakeFiles/llama.dir/models/bailingmoe2.cpp.o
[ 17%] Building CXX object src/CMakeFiles/llama.dir/models/bailingmoe3.cpp.o
/home/runner/work/llama.cpp-optium/llama.cpp-optium/src/models/bailingmoe3.cpp: In member function ‘virtual void llama_model_bailingmoe3::load_arch_tensors(llama_model_loader&)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/src/models/bailingmoe3.cpp:73:13: error: ‘struct llama_model_loader’ has no member named ‘load_mtp’
   73 |     if (!ml.load_mtp) {
      |             ^~~~~~~~
gmake[2]: *** [src/CMakeFiles/llama.dir/build.make:625: src/CMakeFiles/llama.dir/models/bailingmoe3.cpp.o] Error 1
gmake[2]: *** Waiting for unfinished jobs....
gmake[1]: *** [CMakeFiles/Makefile2:2502: src/CMakeFiles/llama.dir/all] Error 2
gmake: *** [Makefile:146: all] Error 2
```
