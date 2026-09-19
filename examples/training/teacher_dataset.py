#!/usr/bin/env python3
"""Generate reproducible OpenAI chat training records using an isolated GGUF teacher.

The teacher process is terminated before this program returns. No external tools
are executed: supplied fixture responses are the only allowed tool results.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_TASKS = [
    {"category": "general", "prompt": "Explain how a hash table handles collisions."},
    {"category": "coding", "prompt": "Write a Python function to merge overlapping intervals and provide tests."},
    {"category": "reasoning", "prompt": "Explain the difference between correlation and causation with an example."},
    {"category": "tool_use", "prompt": "Describe when to use a calculator rather than mental arithmetic."},
    {"category": "research", "prompt": "Explain how to verify claims using primary sources; do not invent citations."},
    {"category": "multitool", "prompt": "Explain how a developer might combine search and a calculator to verify a statistic."},
]


def request_json(base_url, route, data=None, timeout=30):
    payload = None if data is None else json.dumps(data).encode("utf-8")
    req = Request(base_url + route, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def wait_for_server(base_url, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"teacher server exited with code {process.returncode}")
        try:
            request_json(base_url, "/health", timeout=2)
            return
        except (URLError, HTTPError, ValueError, TimeoutError):
            time.sleep(0.25)
    raise TimeoutError("teacher server did not become ready")


def load_tasks(path):
    if path is None:
        return DEFAULT_TASKS
    tasks = []
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            task = json.loads(line)
            if not isinstance(task, dict) or not isinstance(task.get("prompt"), str):
                raise ValueError(f"line {number}: expected an object containing prompt")
            tasks.append(task)
    if not tasks:
        raise ValueError("input contains no tasks")
    return tasks


def normalize_assistant(message):
    result = {"role": "assistant"}
    if message.get("content") is not None:
        result["content"] = message["content"]
    if message.get("tool_calls"):
        result["tool_calls"] = message["tool_calls"]
    if "content" not in result and "tool_calls" not in result:
        raise ValueError("teacher returned an empty assistant response")
    return result


def generate_record(base_url, task, model, tools, max_tokens, max_tool_rounds=4):
    messages = [{"role": "user", "content": task["prompt"]}]
    fixtures = task.get("tool_results", {})
    if not isinstance(fixtures, dict):
        raise ValueError("tool_results must map call IDs or tool names to fixture outputs")
    used = {}

    for round_index in range(max_tool_rounds + 1):
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "none" if round_index == max_tool_rounds else "auto"
        result = request_json(base_url, "/v1/chat/completions", body, timeout=600)
        assistant = normalize_assistant(result["choices"][0]["message"])
        messages.append(assistant)
        calls = assistant.get("tool_calls", [])
        if not calls:
            return {"category": task.get("category", "custom"), "messages": messages}
        if round_index == max_tool_rounds:
            raise ValueError("teacher exceeded the configured maximum tool rounds")
        for call in calls:
            name = call["function"]["name"]
            call_id = call["id"]
            fixture_key = call_id if call_id in fixtures else name
            if fixture_key not in fixtures:
                raise ValueError(f"missing fixture for tool {name!r} ({call_id}); no tool was executed")
            value = fixtures[fixture_key]
            if isinstance(value, list):
                next_index = used.get(fixture_key, 0)
                if next_index >= len(value):
                    raise ValueError(f"exhausted fixture list for tool {fixture_key!r}")
                value = value[next_index]
                used[fixture_key] = next_index + 1
            content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
    raise AssertionError("unreachable")


def run(args):
    if args.max_tool_rounds < 0:
        raise ValueError("max-tool-rounds must be nonnegative")
    tasks = load_tasks(args.tasks)
    tools = json.loads(Path(args.tools).read_text(encoding="utf-8")) if args.tools else []
    if not isinstance(tools, list):
        raise ValueError("tools must be an OpenAI-compatible JSON array")
    cmd = [args.server, "-m", args.model, "--host", "127.0.0.1", "--port", str(args.port)]
    if args.server_args:
        cmd += args.server_args
    base_url = f"http://127.0.0.1:{args.port}"
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=None)
    try:
        wait_for_server(base_url, process, args.startup_timeout)
        # Atomic promotion prevents mistaking a partial dataset for a complete one.
        temporary = output.with_name(output.name + ".partial")
        with open(temporary, "w", encoding="utf-8") as stream:
            for task in tasks:
                record = generate_record(base_url, task, args.model, tools, args.max_tokens, args.max_tool_rounds)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        process.terminate() if process.poll() is None else None
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    print(f"wrote {len(tasks)} OpenAI chat records to {output}; teacher process stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, help="path to llama-server executable")
    parser.add_argument("--model", required=True, help="teacher GGUF path")
    parser.add_argument("--output", required=True, help="output OpenAI messages JSONL path")
    parser.add_argument("--tasks", help="optional JSONL with prompt, category and tool_results")
    parser.add_argument("--tools", help="optional OpenAI-compatible tool definitions JSON array")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-tool-rounds", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=120)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("server_args", nargs="*", help="extra server arguments after --")
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError, TimeoutError, OSError, KeyError, IndexError) as exc:
        print(f"teacher dataset generation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
