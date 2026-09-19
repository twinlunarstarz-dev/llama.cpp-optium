#!/usr/bin/env python3
"""Run GGUF teacher generation, release the teacher, then launch llama-finetune.

This is an orchestration entry point, NOT ternary training or out-of-core training.
The student must already be a format supported by the existing fine-tuner.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", required=True, help="inference-compatible teacher GGUF")
    parser.add_argument("--student-model", required=True, help="existing trainable student GGUF")
    parser.add_argument("--server", required=True, help="llama-server executable")
    parser.add_argument("--finetune", required=True, help="llama-finetune executable")
    parser.add_argument("--dataset", required=True, help="output teacher chat JSONL")
    parser.add_argument("--tasks", help="JSONL prompt and optional fixture tasks")
    parser.add_argument("--tools", help="OpenAI-compatible tool definitions JSON array")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-tool-rounds", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--teacher-server-arg", action="append", default=[], metavar="ARG",
                        help="one llama-server argument; repeat for flags and their values")
    parser.add_argument("--finetune-arg", action="append", default=[], metavar="ARG",
                        help="one llama-finetune argument; repeat for flags and their values")
    return parser.parse_args(argv)


def validate_corpus(path):
    count = 0
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
                raise ValueError(f"{path}:{lineno}: expected a messages array")
            messages = record["messages"]
            if not messages or not any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages):
                raise ValueError(f"{path}:{lineno}: no assistant message")
            count += 1
    if not count:
        raise ValueError(f"empty teacher dataset: {path}")
    return count


def run(args, *, execute=subprocess.run):
    dataset = Path(args.dataset)
    if dataset.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite dataset: {dataset}")
    # Keep the child teacher server separate from the student fine-tune process.
    generator = Path(__file__).with_name("teacher_dataset.py")
    teacher = [sys.executable, str(generator), "--server", args.server,
               "--model", args.teacher_model, "--output", str(dataset),
               "--port", str(args.port), "--max-tokens", str(args.max_tokens),
               "--max-tool-rounds", str(args.max_tool_rounds)]
    for flag, value in (("--tasks", args.tasks), ("--tools", args.tools)):
        if value:
            teacher.extend((flag, value))
    if args.overwrite:
        teacher.append("--overwrite")
    if args.teacher_server_arg:
        teacher.append("--")
        teacher.extend(args.teacher_server_arg)
    # check=True is critical: failed or incomplete teacher generation never starts training.
    execute(teacher, check=True)
    # teacher_dataset.py terminates AND waits for its server before returning.
    if not dataset.is_file():
        raise FileNotFoundError(f"teacher exited without producing {dataset}")
    count = validate_corpus(dataset)
    print(f"teacher stopped; validated {count} records; starting fine-tuning", flush=True)
    student = [args.finetune, "--model", args.student_model, "--teacher-jsonl", str(dataset)]
    student.extend(args.finetune_arg)
    execute(student, check=True)
    return count


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"teacher/student pipeline failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
