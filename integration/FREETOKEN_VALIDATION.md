# FreeToken sequential continuation validation

- configure rc: 0
- build rc: 2
- focused tests rc: 99

## Build
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
[ 17%] Building CXX object src/CMakeFiles/llama.dir/models/bert.cpp.o
[ 17%] Building CXX object src/CMakeFiles/llama.dir/models/bitnet.cpp.o
[ 17%] Building CXX object src/CMakeFiles/llama.dir/models/bloom.cpp.o
[ 17%] Building CXX object src/CMakeFiles/llama.dir/models/chameleon.cpp.o
[ 18%] Building CXX object src/CMakeFiles/llama.dir/models/chatglm.cpp.o
[ 18%] Building CXX object src/CMakeFiles/llama.dir/models/clip.cpp.o
[ 18%] Building CXX object src/CMakeFiles/llama.dir/models/codeshell.cpp.o
[ 18%] Building CXX object src/CMakeFiles/llama.dir/models/cogvlm.cpp.o
[ 18%] Building CXX object src/CMakeFiles/llama.dir/models/cohere2.cpp.o
[ 19%] Building CXX object src/CMakeFiles/llama.dir/models/cohere2moe.cpp.o
[ 19%] Building CXX object src/CMakeFiles/llama.dir/models/command-r.cpp.o
[ 19%] Building CXX object src/CMakeFiles/llama.dir/models/dbrx.cpp.o
[ 19%] Building CXX object src/CMakeFiles/llama.dir/models/deci.cpp.o
[ 19%] Building CXX object src/CMakeFiles/llama.dir/models/deepseek.cpp.o
[ 20%] Building CXX object src/CMakeFiles/llama.dir/models/deepseek2.cpp.o
[ 20%] Building CXX object src/CMakeFiles/llama.dir/models/deepseek2ocr.cpp.o
[ 20%] Building CXX object src/CMakeFiles/llama.dir/models/deepseek32.cpp.o
[ 20%] Building CXX object src/CMakeFiles/llama.dir/models/deepseek4.cpp.o
[ 20%] Building CXX object src/CMakeFiles/llama.dir/models/delta-net-base.cpp.o
[ 21%] Building CXX object src/CMakeFiles/llama.dir/models/dflash.cpp.o
[ 21%] Building CXX object src/CMakeFiles/llama.dir/models/dots1.cpp.o
[ 21%] Building CXX object src/CMakeFiles/llama.dir/models/dots3note.cpp.o
[ 21%] Building CXX object src/CMakeFiles/llama.dir/models/dream.cpp.o
[ 21%] Building CXX object src/CMakeFiles/llama.dir/models/eagle3.cpp.o
[ 22%] Building CXX object src/CMakeFiles/llama.dir/models/ernie4-5-moe.cpp.o
[ 22%] Building CXX object src/CMakeFiles/llama.dir/models/ernie4-5.cpp.o
[ 22%] Building CXX object src/CMakeFiles/llama.dir/models/eurobert.cpp.o
[ 22%] Building CXX object src/CMakeFiles/llama.dir/models/exaone-moe.cpp.o
[ 22%] Building CXX object src/CMakeFiles/llama.dir/models/exaone.cpp.o
[ 23%] Building CXX object src/CMakeFiles/llama.dir/models/exaone4.cpp.o
[ 23%] Building CXX object src/CMakeFiles/llama.dir/models/falcon-h1.cpp.o
[ 23%] Building CXX object src/CMakeFiles/llama.dir/models/falcon.cpp.o
[ 23%] Building CXX object src/CMakeFiles/llama.dir/models/gemma-embedding.cpp.o
[ 23%] Building CXX object src/CMakeFiles/llama.dir/models/gemma.cpp.o
[ 23%] Building CXX object src/CMakeFiles/llama.dir/models/gemma2.cpp.o
[ 24%] Building CXX object src/CMakeFiles/llama.dir/models/gemma3.cpp.o
[ 24%] Building CXX object src/CMakeFiles/llama.dir/models/gemma3n.cpp.o
[ 24%] Building CXX object src/CMakeFiles/llama.dir/models/gemma4-assistant.cpp.o
[ 24%] Building CXX object src/CMakeFiles/llama.dir/models/gemma4.cpp.o
[ 24%] Building CXX object src/CMakeFiles/llama.dir/models/glm-dsa.cpp.o
[ 25%] Building CXX object src/CMakeFiles/llama.dir/models/glm4-moe.cpp.o
[ 25%] Building CXX object src/CMakeFiles/llama.dir/models/glm4.cpp.o
[ 25%] Building CXX object src/CMakeFiles/llama.dir/models/gpt2.cpp.o
[ 25%] Building CXX object src/CMakeFiles/llama.dir/models/gptneox.cpp.o
[ 25%] Building CXX object src/CMakeFiles/llama.dir/models/granite-hybrid.cpp.o
[ 26%] Building CXX object src/CMakeFiles/llama.dir/models/granite-moe.cpp.o
[ 26%] Building CXX object src/CMakeFiles/llama.dir/models/granite-swa.cpp.o
[ 26%] Building CXX object src/CMakeFiles/llama.dir/models/granite-switch.cpp.o
[ 26%] Building CXX object src/CMakeFiles/llama.dir/models/granite.cpp.o
[ 26%] Building CXX object src/CMakeFiles/llama.dir/models/grok.cpp.o
[ 27%] Building CXX object src/CMakeFiles/llama.dir/models/grovemoe.cpp.o
[ 27%] Building CXX object src/CMakeFiles/llama.dir/models/hunyuan-dense.cpp.o
[ 27%] Building CXX object src/CMakeFiles/llama.dir/models/hunyuan-moe.cpp.o
[ 27%] Building CXX object src/CMakeFiles/llama.dir/models/hunyuan-vl.cpp.o
[ 27%] Building CXX object src/CMakeFiles/llama.dir/models/hy-v3.cpp.o
[ 28%] Building CXX object src/CMakeFiles/llama.dir/models/internlm2.cpp.o
[ 28%] Building CXX object src/CMakeFiles/llama.dir/models/jais.cpp.o
[ 28%] Building CXX object src/CMakeFiles/llama.dir/models/jais2.cpp.o
[ 28%] Building CXX object src/CMakeFiles/llama.dir/models/jamba.cpp.o
[ 28%] Building CXX object src/CMakeFiles/llama.dir/models/jina-bert-v2.cpp.o
[ 29%] Building CXX object src/CMakeFiles/llama.dir/models/jina-bert-v3.cpp.o
[ 29%] Building CXX object src/CMakeFiles/llama.dir/models/kimi-k3.cpp.o
[ 29%] Building CXX object src/CMakeFiles/llama.dir/models/kimi-linear.cpp.o
[ 29%] Building CXX object src/CMakeFiles/llama.dir/models/laguna.cpp.o
[ 29%] Building CXX object src/CMakeFiles/llama.dir/models/lfm2.cpp.o
[ 29%] Building CXX object src/CMakeFiles/llama.dir/models/lfm2moe.cpp.o
[ 30%] Building CXX object src/CMakeFiles/llama.dir/models/llada-moe.cpp.o
[ 30%] Building CXX object src/CMakeFiles/llama.dir/models/llada.cpp.o
[ 30%] Building CXX object src/CMakeFiles/llama.dir/models/llama-embed.cpp.o
[ 30%] Building CXX object src/CMakeFiles/llama.dir/models/llama.cpp.o
[ 30%] Building CXX object src/CMakeFiles/llama.dir/models/llama4.cpp.o
[ 31%] Building CXX object src/CMakeFiles/llama.dir/models/maincoder.cpp.o
[ 31%] Building CXX object src/CMakeFiles/llama.dir/models/mamba-base.cpp.o
[ 31%] Building CXX object src/CMakeFiles/llama.dir/models/mamba.cpp.o
[ 31%] Building CXX object src/CMakeFiles/llama.dir/models/mamba2.cpp.o
[ 31%] Building CXX object src/CMakeFiles/llama.dir/models/mellum.cpp.o
[ 32%] Building CXX object src/CMakeFiles/llama.dir/models/mimo2.cpp.o
[ 32%] Building CXX object src/CMakeFiles/llama.dir/models/minicpm.cpp.o
[ 32%] Building CXX object src/CMakeFiles/llama.dir/models/minicpm3.cpp.o
[ 32%] Building CXX object src/CMakeFiles/llama.dir/models/minimax-01.cpp.o
[ 32%] Building CXX object src/CMakeFiles/llama.dir/models/minimax-m2.cpp.o
[ 33%] Building CXX object src/CMakeFiles/llama.dir/models/minimax-m3.cpp.o
[ 33%] Building CXX object src/CMakeFiles/llama.dir/models/mistral3.cpp.o
[ 33%] Building CXX object src/CMakeFiles/llama.dir/models/mistral4.cpp.o
[ 33%] Building CXX object src/CMakeFiles/llama.dir/models/modern-bert.cpp.o
[ 33%] Building CXX object src/CMakeFiles/llama.dir/models/mpt.cpp.o
[ 34%] Building CXX object src/CMakeFiles/llama.dir/models/muse-glimmer.cpp.o
[ 34%] Building CXX object src/CMakeFiles/llama.dir/models/nanbeige.cpp.o
[ 34%] Building CXX object src/CMakeFiles/llama.dir/models/nemotron-h-moe.cpp.o
[ 34%] Building CXX object src/CMakeFiles/llama.dir/models/nemotron-h.cpp.o
[ 34%] Building CXX object src/CMakeFiles/llama.dir/models/nemotron.cpp.o
[ 35%] Building CXX object src/CMakeFiles/llama.dir/models/neo-bert.cpp.o
[ 35%] Building CXX object src/CMakeFiles/llama.dir/models/nomic-bert-moe.cpp.o
[ 35%] Building CXX object src/CMakeFiles/llama.dir/models/nomic-bert.cpp.o
[ 35%] Building CXX object src/CMakeFiles/llama.dir/models/olmo.cpp.o
[ 35%] Building CXX object src/CMakeFiles/llama.dir/models/olmo2.cpp.o
[ 36%] Building CXX object src/CMakeFiles/llama.dir/models/olmoe.cpp.o
[ 36%] Building CXX object src/CMakeFiles/llama.dir/models/openai-moe.cpp.o
[ 36%] Building CXX object src/CMakeFiles/llama.dir/models/openelm.cpp.o
[ 36%] Building CXX object src/CMakeFiles/llama.dir/models/orion.cpp.o
[ 36%] Building CXX object src/CMakeFiles/llama.dir/models/paddleocr.cpp.o
[ 36%] Building CXX object src/CMakeFiles/llama.dir/models/pangu-embed.cpp.o
[ 37%] Building CXX object src/CMakeFiles/llama.dir/models/phi2.cpp.o
[ 37%] Building CXX object src/CMakeFiles/llama.dir/models/phi3.cpp.o
[ 37%] Building CXX object src/CMakeFiles/llama.dir/models/phimoe.cpp.o
[ 37%] Building CXX object src/CMakeFiles/llama.dir/models/plamo.cpp.o
[ 37%] Building CXX object src/CMakeFiles/llama.dir/models/plamo2.cpp.o
[ 38%] Building CXX object src/CMakeFiles/llama.dir/models/plamo3.cpp.o
[ 38%] Building CXX object src/CMakeFiles/llama.dir/models/plm.cpp.o
[ 38%] Building CXX object src/CMakeFiles/llama.dir/models/pockettts.cpp.o
[ 38%] Building CXX object src/CMakeFiles/llama.dir/models/qwen.cpp.o
[ 38%] Building CXX object src/CMakeFiles/llama.dir/models/qwen2.cpp.o
[ 39%] Building CXX object src/CMakeFiles/llama.dir/models/qwen2moe.cpp.o
[ 39%] Building CXX object src/CMakeFiles/llama.dir/models/qwen2vl.cpp.o
[ 39%] Building CXX object src/CMakeFiles/llama.dir/models/qwen3.cpp.o
[ 39%] Building CXX object src/CMakeFiles/llama.dir/models/qwen35.cpp.o
[ 39%] Building CXX object src/CMakeFiles/llama.dir/models/qwen35moe.cpp.o
[ 40%] Building CXX object src/CMakeFiles/llama.dir/models/qwen3moe.cpp.o
[ 40%] Building CXX object src/CMakeFiles/llama.dir/models/qwen3next.cpp.o
[ 40%] Building CXX object src/CMakeFiles/llama.dir/models/qwen3tts.cpp.o
[ 40%] Building CXX object src/CMakeFiles/llama.dir/models/qwen3vl.cpp.o
[ 40%] Building CXX object src/CMakeFiles/llama.dir/models/qwen3vlmoe.cpp.o
[ 41%] Building CXX object src/CMakeFiles/llama.dir/models/refact.cpp.o
[ 41%] Building CXX object src/CMakeFiles/llama.dir/models/rnd1.cpp.o
[ 41%] Building CXX object src/CMakeFiles/llama.dir/models/rwkv6-base.cpp.o
[ 41%] Building CXX object src/CMakeFiles/llama.dir/models/rwkv6.cpp.o
[ 41%] Building CXX object src/CMakeFiles/llama.dir/models/rwkv6qwen2.cpp.o
[ 42%] Building CXX object src/CMakeFiles/llama.dir/models/rwkv7-base.cpp.o
[ 42%] Building CXX object src/CMakeFiles/llama.dir/models/rwkv7.cpp.o
[ 42%] Building CXX object src/CMakeFiles/llama.dir/models/seed-oss.cpp.o
[ 42%] Building CXX object src/CMakeFiles/llama.dir/models/smallthinker.cpp.o
[ 42%] Building CXX object src/CMakeFiles/llama.dir/models/smollm3.cpp.o
[ 43%] Building CXX object src/CMakeFiles/llama.dir/models/stablelm.cpp.o
[ 43%] Building CXX object src/CMakeFiles/llama.dir/models/starcoder.cpp.o
[ 43%] Building CXX object src/CMakeFiles/llama.dir/models/starcoder2.cpp.o
[ 43%] Building CXX object src/CMakeFiles/llama.dir/models/step35.cpp.o
[ 43%] Building CXX object src/CMakeFiles/llama.dir/models/t5.cpp.o
[ 43%] Building CXX object src/CMakeFiles/llama.dir/models/t5encoder.cpp.o
[ 44%] Building CXX object src/CMakeFiles/llama.dir/models/talkie.cpp.o
[ 44%] Building CXX object src/CMakeFiles/llama.dir/models/wavtokenizer-dec.cpp.o
[ 44%] Building CXX object src/CMakeFiles/llama.dir/models/xverse.cpp.o
[ 44%] Linking CXX shared library ../bin/libllama.so
[ 44%] Built target llama
[ 44%] Building CXX object common/CMakeFiles/llama-common.dir/arg.cpp.o
[ 44%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/mtmd.cpp.o
[ 44%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/mtmd-audio.cpp.o
[ 44%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/mtmd-image.cpp.o
[ 45%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/mtmd-helper.cpp.o
[ 45%] Building CXX object common/CMakeFiles/llama-common.dir/chat-auto-parser-generator.cpp.o
[ 46%] Building CXX object common/CMakeFiles/llama-common.dir/chat-auto-parser-helpers.cpp.o
[ 46%] Building CXX object common/CMakeFiles/llama-common.dir/chat-diff-analyzer.cpp.o
[ 46%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/mtmd-helper-gen.cpp.o
[ 46%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/clip.cpp.o
[ 46%] Building CXX object common/CMakeFiles/llama-common.dir/chat-peg-parser.cpp.o
[ 46%] Building CXX object common/CMakeFiles/llama-common.dir/chat.cpp.o
[ 46%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/cogvlm.cpp.o
[ 46%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/conformer.cpp.o
[ 47%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/dots3note.cpp.o
[ 47%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/dotsocr.cpp.o
[ 47%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/exaone4_5.cpp.o
[ 47%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/gemma4a.cpp.o
[ 47%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/gemma4v.cpp.o
[ 48%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/gemma4ua.cpp.o
[ 48%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/gemma4uv.cpp.o
[ 48%] Building CXX object common/CMakeFiles/llama-common.dir/common.cpp.o
[ 48%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/glm4v.cpp.o
[ 48%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/granite-speech.cpp.o
[ 48%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/granite4-vision.cpp.o
[ 48%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/hunyuanvl.cpp.o
[ 49%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/internvl.cpp.o
[ 49%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/kimivl.cpp.o
[ 49%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/kimik25.cpp.o
[ 50%] Building CXX object common/CMakeFiles/llama-common.dir/console.cpp.o
[ 50%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/nemotron-v2-vl.cpp.o
[ 50%] Building CXX object common/CMakeFiles/llama-common.dir/debug.cpp.o
[ 50%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/muse-glimmer.cpp.o
[ 51%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/llama4.cpp.o
[ 51%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/llava.cpp.o
[ 51%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/minicpmv.cpp.o
[ 51%] Building CXX object common/CMakeFiles/llama-common.dir/download.cpp.o
[ 51%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/paddleocr.cpp.o
[ 51%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/pixtral.cpp.o
[ 52%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/qwen2vl.cpp.o
[ 52%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/minimax-m3.cpp.o
[ 52%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/qwen3vl.cpp.o
[ 52%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/mimovl.cpp.o
[ 52%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/qwen3a.cpp.o
[ 53%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/mimo-audio.cpp.o
[ 53%] Building CXX object common/CMakeFiles/llama-common.dir/fit.cpp.o
/home/runner/work/llama.cpp-optium/llama.cpp-optium/common/fit.cpp: In function ‘std::vector<llama_device_memory_data> common_get_device_memory_data_impl(const char*, const llama_model_params*, const llama_context_params*, std::vector<ggml_backend_device*>&, uint32_t&, uint32_t&, uint32_t&, ggml_log_level)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/common/fit.cpp:57:18: error: ‘struct llama_model_params’ has no member named ‘use_mmap’
   57 |     mparams_copy.use_mmap        = false;
      |                  ^~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/common/fit.cpp:58:18: error: ‘struct llama_model_params’ has no member named ‘use_mlock’
   58 |     mparams_copy.use_mlock       = false;
      |                  ^~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/common/fit.cpp:142:18: error: ‘const struct llama_model_params’ has no member named ‘load_mtp’; did you mean ‘load_mode’?
  142 |     if (mparams->load_mtp) {
      |                  ^~~~~~~~
      |                  load_mode
gmake[2]: *** [common/CMakeFiles/llama-common.dir/build.make:219: common/CMakeFiles/llama-common.dir/fit.cpp.o] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:2635: common/CMakeFiles/llama-common.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
[ 53%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/qwen3tts-spkenc.cpp.o
[ 53%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/qwen3tts-gen.cpp.o
[ 53%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/pockettts-seanet.cpp.o
[ 53%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/pockettts-spkenc.cpp.o
[ 54%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/pockettts-gen.cpp.o
[ 54%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/step3vl.cpp.o
[ 54%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/siglip.cpp.o
[ 54%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/whisper-enc.cpp.o
[ 54%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/deepseekocr.cpp.o
[ 55%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/deepseekocr2.cpp.o
[ 55%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/mobilenetv5.cpp.o
[ 55%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/youtuvl.cpp.o
[ 55%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/yasa2.cpp.o
[ 55%] Building CXX object tools/mtmd/CMakeFiles/mtmd.dir/models/parakeet.cpp.o
[ 56%] Linking CXX shared library ../../bin/libmtmd.so
[ 56%] Built target mtmd
gmake: *** [Makefile:146: all] Error 2
```
