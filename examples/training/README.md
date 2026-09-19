# llama.cpp/examples/training

This directory contains the existing `llama-finetune` executable and an experimental teacher-data generator.

## Existing `llama-finetune`

The build target is registered in `examples/training/CMakeLists.txt` (not `tools/CMakeLists.txt`). The current proof-of-concept fine-tuner supports FP32 models on limited hardware configurations. It forces non-mmap weight loading and F32 KV-cache types; it does **not** yet implement ternary full training, ternary LoRA, QLoRA, arbitrary architecture backward support, or tiered VRAM/RAM/disk training.

For CPU training, the original example recommends building without additional GPU backends. For CUDA training, it recommends offloading the maximum number of GPU layers. Original example:

```sh
export model_name=llama_3.2-1b && export quantization=f32
./build/bin/llama-finetune --file wikitext-2-raw/wiki.test.raw -ngl 999 --model models/${model_name}-${quantization}.gguf -c 512 -b 512 -ub 512
./build/bin/llama-perplexity --file wikitext-2-raw/wiki.test.raw -ngl 999 --model finetuned-model.gguf
```

## Isolated teacher dataset generation and training

`teacher_dataset.py` runs an inference-compatible teacher GGUF in a child `llama-server`, requests OpenAI-compatible chat completions, writes a JSONL corpus atomically, and terminates and waits for the teacher process before exiting. Its six built-in prompts are smoke-test coverage only; supply your own task corpus for meaningful training data.

```sh
python3 examples/training/teacher_dataset.py \
  --server ./build/bin/llama-server \
  --model /path/to/base.gguf \
  --tasks /path/to/tasks.jsonl \
  --tools /path/to/tools.json \
  --output /path/to/teacher.jsonl

./build/bin/llama-finetune \
  --model /path/to/student-f32.gguf \
  --teacher-jsonl /path/to/teacher.jsonl \
  -c 512 -b 512 -ub 512
```

Example `tasks.jsonl` (one JSON object per line):

```jsonl
{"category":"coding","prompt":"Write a tested parser for CSV quoted fields."}
{"category":"tool_use","prompt":"Use the calculator to compute 2 + 2.","tool_results":{"calculator":{"result":4}}}
```

`tools.json` must be an OpenAI-compatible JSON array of tool declarations. Tool outputs are fixture data supplied in `tool_results`: this utility deliberately does not execute tools, access search engines, or invent tool results. It supports multiple tool-call rounds; a task with an actual tool call but no matching fixture fails rather than fabricating a result.

The output records contain a `category` and OpenAI-compatible `messages` array. `llama-finetune --teacher-jsonl` parses these records and renders them with the **student model's** chat template and tokenizer before training. An optional per-record `tools` array can be included when preserving the tool schema is necessary for rendering. Do not pass JSONL to `--file`: that option expects ordinary text.

**Current limitations:** The training path concatenates transcripts and optimizes all message roles, not just assistant tokens. It uses the existing in-memory optimizer, not the intended tiered VRAM/RAM/disk engine. Neither ternary training nor QLoRA is implemented. The generator's fixture-based examples are not sufficient for broad coding, research or genuine tool-use proficiency. The JSONL path has been compile-tested, but an actual GGUF training run and chat-template correctness for particular model families have not been validated.

The script requires Python 3 and a running localhost TCP stack; it uses only Python's standard library. Use a free `--port` when the default 18765 is busy. It refuses to overwrite an existing output unless `--overwrite` is set.

Run its local unit and simulated-server integration tests with:

```sh
python3 -m unittest discover -s examples/training -p 'test_teacher_dataset.py' -v
```

See [the project architecture note](../../docs/ternary-training-architecture.md) for the remaining training and Bonsai 2 compatibility work.
