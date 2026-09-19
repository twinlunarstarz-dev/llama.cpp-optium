# llama.cpp/examples/training

This directory contains the existing `llama-finetune` executable and experimental teacher-data utilities.

## Existing `llama-finetune`

The build target is registered in `examples/training/CMakeLists.txt` (not `tools/CMakeLists.txt`). The existing example fine-tuner supports FP32 models on limited hardware configurations. It forces non-mmap weight loading and F32 KV-cache types; it does **not** yet implement ternary full training, ternary LoRA, QLoRA, arbitrary architecture backward support, or tiered VRAM/RAM/disk training.

For CPU training, the original example recommends building without additional GPU backends. For CUDA training, it recommends offloading the maximum number of GPU layers. Original example:

```sh
export model_name=llama_3.2-1b && export quantization=f32
./build/bin/llama-finetune --file wikitext-2-raw/wiki.test.raw -ngl 999 --model models/${model_name}-${quantization}.gguf -c 512 -b 512 -ub 512
./build/bin/llama-perplexity --file wikitext-2-raw/wiki.test.raw -ngl 999 --model finetuned-model.gguf
```

## Generate with a teacher, unload it, then train

`teacher_student.py` provides a single entry point for two **sequential** processes. First it calls `teacher_dataset.py`, which launches the teacher in a child `llama-server`, generates an OpenAI-compatible JSONL corpus, and terminates and waits for the server. The launcher validates that corpus and only then invokes the existing `llama-finetune` with a separately specified student GGUF. Any teacher failure stops the pipeline before training. The student must already be in a format supported by the existing fine-tuner; this script performs no ternary conversion or training itself.

```sh
python3 examples/training/teacher_student.py \
  --server ./build/bin/llama-server \
  --finetune ./build/bin/llama-finetune \
  --teacher-model /path/to/teacher.gguf \
  --student-model /path/to/student-f32.gguf \
  --tasks /path/to/tasks.jsonl \
  --tools /path/to/tools.json \
  --dataset /path/to/teacher.jsonl \
  --finetune-arg=-c --finetune-arg=512 \
  --finetune-arg=-b --finetune-arg=512 \
  --finetune-arg=-ub --finetune-arg=512
```

Use one `--teacher-server-arg=VALUE` or `--finetune-arg=VALUE` per forwarded argument. For flags beginning with `-`, use the `=VALUE` syntax shown above. The teacher server is not kept resident while training. The JSONL is retained after training for inspection; an existing dataset is not overwritten unless `--overwrite` is specified. Supply an appropriate `--port` if the default 18765 is occupied. The six built-in prompts are smoke-test examples only: a useful corpus requires a substantially larger curated task file.

To generate a corpus without immediately training, run:

```sh
python3 examples/training/teacher_dataset.py \
  --server ./build/bin/llama-server \
  --model /path/to/base.gguf \
  --tasks /path/to/tasks.jsonl \
  --tools /path/to/tools.json \
  --output /path/to/teacher.jsonl
```

Example `tasks.jsonl` (one JSON object per line):

```jsonl
{"category":"coding","prompt":"Write a tested parser for CSV quoted fields."}
{"category":"tool_use","prompt":"Use the calculator to compute 2 + 2.","tool_results":{"calculator":{"result":4}}}
```

`tools.json` must be an OpenAI-compatible JSON array of tool declarations. Tool outputs are fixture data supplied in `tool_results`: this utility deliberately does not execute tools, access search engines, or invent tool results. It supports multiple tool-call rounds; a task with an actual tool call but no matching fixture fails rather than fabricating a result.

The output records contain a `category`, OpenAI-compatible `messages` array, and optional `tools` array. `llama-finetune --teacher-jsonl` parses these records and renders them with the **student model's** chat template and tokenizer before training. Do not pass JSONL to `--file`: that option expects ordinary text.

**Current limitations:** The training path concatenates transcripts and optimizes all message roles, not just assistant tokens. It uses the existing in-memory optimizer, not the intended tiered VRAM/RAM/disk engine. Neither ternary training nor QLoRA is implemented. The fixture-based examples are not sufficient for broad coding, research or genuine tool-use proficiency. The JSONL path has been compile-tested, but an actual GGUF training run and chat-template correctness for particular model families have not been validated.

The scripts require Python 3 and a localhost TCP stack; they use only the Python standard library. Run their unit and simulated-server tests with:

```sh
python3 -m unittest discover -s examples/training -p 'test_teacher_*.py' -v
```

See [the project architecture note](../../docs/ternary-training-architecture.md) for the remaining training and Bonsai 2 compatibility work.
