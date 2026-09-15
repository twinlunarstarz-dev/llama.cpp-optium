#!/usr/bin/env bash
# benchmark-llama.cpp.sh — long-context llama.cpp service benchmark
#
# Usage:
#   ./benchmark-llama.cpp.sh [MODEL] [PORT] [RUNS_PER_SIZE] [OUTPUT_TOKENS]
#
# Defaults:
#   MODEL=Qwen3.8-27B-Q5-Layer, PORT=8025, RUNS_PER_SIZE=1,
#   OUTPUT_TOKENS=2000.
#
# The script measures the running llama-server through its OpenAI-compatible
# API. It creates prompts near 32k/128k/180k tokens, verifies their actual
# token count with /tokenize, sends cache_prompt=false requests, records the
# server's prompt/generation timings, samples GPU memory, and writes results
# under ./benchmark-results/<timestamp>-<model-slug>/.

set -euo pipefail

MODEL="${1:-Qwen3.8-27B-Q5-Layer}"
PORT="${2:-8025}"
RUNS_PER_SIZE="${3:-1}"
OUTPUT_TOKENS="${4:-2000}"
BASE_URL="${LLAMA_BENCHMARK_URL:-http://127.0.0.1:${PORT}}"
TARGETS=(32000 128000 180000)
if [ -n "${LLAMA_BENCHMARK_TARGETS:-}" ]; then
    read -r -a TARGETS <<< "$LLAMA_BENCHMARK_TARGETS"
fi
for target in "${TARGETS[@]}"; do
    [[ "$target" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid target token count: %s\n' "$target" >&2; exit 2; }
done

if ! [[ "$PORT" =~ ^[0-9]+$ && "$RUNS_PER_SIZE" =~ ^[1-9][0-9]*$ && "$OUTPUT_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Invalid numeric argument.\n' >&2
    exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MODEL_SLUG="$(printf '%s' "$MODEL" | tr -cs '[:alnum:]._-' '_' | sed 's/^_*//; s/_*$//')"
OUT_DIR="${LLAMA_BENCHMARK_OUT_DIR:-${ROOT_DIR}/benchmark-results/${STAMP}-${MODEL_SLUG}}"
mkdir -p "$(dirname "$OUT_DIR")"
LOCK_PATH="${LLAMA_BENCHMARK_LOCK:-${ROOT_DIR}/benchmark-results/.benchmark.lock}"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    printf 'Another benchmark is already running (lock: %s)\n' "$LOCK_PATH" >&2
    exit 3
fi
TMP_DIR="${OUT_DIR}/tmp"
mkdir -p "$TMP_DIR"

cleanup() {
    local exit_code=$?
    if [ "$exit_code" -ne 0 ] && [ -n "${BENCH_START_EPOCH:-}" ]; then
        capture_service_kv_logs "$BENCH_START_EPOCH"
    fi
    rm -rf "$TMP_DIR"
    exit "$exit_code"
}
trap cleanup EXIT

RESULTS_TSV="${OUT_DIR}/results.tsv"
SUMMARY_JSON="${OUT_DIR}/summary.json"
RUN_META_JSON="${OUT_DIR}/run-metadata.json"
SERVICE_LOG="${OUT_DIR}/service-kv.log"
printf 'target_tokens\tactual_prompt_tokens\trun\tpp_tok_s\ttg_tok_s\tpredicted_tokens\tdraft_accepted\tdraft_tokens\twall_ms\tgpu0_used_mib\tgpu1_used_mib\thttp_code\n' > "$RESULTS_TSV"

# Capture the service-side evidence needed to interpret benchmark timings.
# This includes actual KV/RS allocation sizes, memory breakdowns, and every
# RAM/disk active-state transition. If the invoking user cannot read the
# system journal, the benchmark still runs and records that limitation.
capture_service_kv_logs() {
    local since_epoch="${1:-0}"
    {
        printf '\n===== journal snapshot since @%s (%s) =====\n' "$since_epoch" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        if ! command -v journalctl >/dev/null 2>&1; then
            printf 'journalctl unavailable\n'
            return 0
        fi
        journalctl -u llama-cpp --since "@${since_epoch}" --no-pager -o short-iso 2>&1 | \
            python3 -c 'import re,sys; rx=re.compile(r"KV|RS buffer|recurrent|memory breakdown|active KV|GPU agent cache|prompt cache|suspend|restore|spill|disk tier|RAM tier|OOM|out of memory", re.I); print("\n".join(line.rstrip() for line in sys.stdin if rx.search(line)))'
    } >> "$SERVICE_LOG"
}

# Do not put the generated prompt in a shell variable. Large shell arguments
# are truncated or rejected on several WSL/Windows paths.
python3 - "$TMP_DIR" "${TARGETS[@]}" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
targets = [int(x) for x in sys.argv[2:]]

# This block is deliberately information-dense but semantically neutral. It is
# repeated to obtain long-context workloads without asking the model to produce
# or repeat the entire input. Actual token counts are verified by /tokenize.
block = """A distributed systems engineer evaluates a service by measuring the work performed, the resources consumed, and the correctness of the result. A useful experiment changes one variable at a time, records the exact configuration, and repeats the observation under comparable conditions. Latency, throughput, memory pressure, queue depth, cache reuse, and failure behavior can interact, so a result is meaningful only when the request, model, sampler, and hardware placement are held constant. The report should distinguish measured evidence from assumptions, identify uncertainty, and preserve enough metadata for another operator to reproduce the result.

Modern language-model inference has separate prompt-processing and token-generation phases. Prompt processing usually benefits from larger batches and parallel matrix operations, while generation repeatedly advances a small active batch and is often limited by memory bandwidth, synchronization, or inter-device communication. A unified key-value cache allows multiple sequences to share a bounded pool, but logical context capacity is not the same as simultaneously resident physical state. Safe admission must account for model weights, KV tensors, compute buffers, recurrent state, allocator headroom, host memory, and durable spill capacity before work is accepted.

Performance engineering is most reliable when optimizations are evaluated with an accuracy guard. Deterministic decoding, stable tokenization, exact prompt rendering, and byte-for-byte or token-ID comparison can reveal regressions that a throughput number hides. A faster kernel is not an improvement if it changes selected tokens, corrupts a restored sequence, drops a speculative branch, or causes a later request to time out. The final result should state the prompt token count, generated token count, prompt tokens per second, generation tokens per second, memory readings, cache state, and any incomplete or rejected request.

A heterogeneous dual-GPU host can improve capacity while reducing throughput if tensors are split poorly. The faster device may need a larger share, but the best ratio depends on quantization, layer placement, KV type, batch size, graph shape, PCIe transfer behavior, and the current memory envelope. Splitting modes have different semantics: layer splitting assigns layers, tensor splitting divides tensor work, and a compatibility fallback must be explicit rather than silently pretending that a native mode is available. The safest benchmark records the mode and validates the same answer before comparing speed.

Durable state management should use transactional writes. A state is not considered spilled until the complete target state, draft state, metadata, and any speculative sidecar have been written, flushed, and reopened successfully. If storage is full or a restore fails, the active state must remain available and admission should wait or fail closed. Temporary files, atomic renames, checksums, quotas, and cleanup after successful restore prevent partial state from becoming a false cache hit. These principles apply equally to a local RAM cache and a disk-backed cache.
"""

for target in targets:
    # The repeated block is intentionally oversized; /tokenize determines the
    # exact count and the service response records the authoritative prompt_n.
    repeats = max(1, (target - 50) // 556)
    text = (block + "\n") * repeats
    prompt = (
        "Read the following technical working document. Analyze its themes and "
        "trade-offs, then produce a concise final answer containing the phrase "
        "BENCHMARK_COMPLETED and a short list of the most important conclusions. "
        "Do not quote the document.\n\nDOCUMENT:\n\n" + text
    )
    (out / "benchmark-block.txt").write_text(block, encoding="utf-8")
    (out / f"prompt-{target}.txt").write_text(prompt, encoding="utf-8")
    (out / f"prompt-{target}.meta.json").write_text(
        json.dumps({"target_tokens": target, "repeats": repeats, "chars": len(prompt)}),
        encoding="utf-8",
    )
PY

# Confirm the service is reachable before creating expensive requests.
python3 - "$BASE_URL/health" "$BASE_URL/v1/models" <<'PY'
import json
import sys
import urllib.error
import urllib.request

for url in sys.argv[1:]:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read()
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}")
            print(f"preflight {url}: HTTP {response.status}")
            if url.endswith('/v1/models'):
                data = json.loads(body)
                ids = [item.get('id') for item in data.get('data', [])]
                print(f"models: {ids}")
    except Exception as exc:
        print(f"preflight failed for {url}: {exc}", file=sys.stderr)
        sys.exit(1)
PY

python3 - "$MODEL" "$BASE_URL" "$RUNS_PER_SIZE" "$OUTPUT_TOKENS" "${TARGETS[@]}" > "$RUN_META_JSON" <<'PY'
import json
import os
import pathlib
import subprocess
import sys
import time

model, base_url, runs, output_tokens = sys.argv[1:5]
targets = [int(x) for x in sys.argv[5:]]

def gpu_snapshot():
    queries = "index,name,memory.total,memory.used,memory.free,utilization.gpu"
    commands = [
        ["nvidia-smi", f"--query-gpu={queries}", "--format=csv,noheader,nounits"],
        ["/usr/lib/wsl/lib/nvidia-smi", f"--query-gpu={queries}", "--format=csv,noheader,nounits"],
    ]
    for cmd in commands:
        try:
            text = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=15)
            rows = []
            for line in text.splitlines():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    rows.append({"index": parts[0], "name": parts[1], "total_mib": parts[2], "used_mib": parts[3], "free_mib": parts[4], "utilization": parts[5] if len(parts) > 5 else ""})
            return rows
        except Exception:
            pass
    return []

print(json.dumps({
    "model": model,
    "base_url": base_url,
    "runs_per_size": int(runs),
    "output_tokens": int(output_tokens),
    "targets": targets,
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "pid": os.getpid(),
    "gpu_before": gpu_snapshot(),
}, indent=2))
PY

# Warm only the HTTP/router path. This is not included in timing results.
BENCH_START_EPOCH="$(date +%s)"
WARMUP_PAYLOAD="$TMP_DIR/warmup.json"
python3 - "$WARMUP_PAYLOAD" "$MODEL" <<'PY'
import json
import sys
path, model = sys.argv[1:]
json.dump({
    "model": model,
    "messages": [{"role": "user", "content": "Reply with WARMUP_READY."}],
    "max_tokens": 8,
    "temperature": 0.0,
    "cache_prompt": False,
}, open(path, "w"))
PY
WARMUP_CODE="$(curl -sS --max-time 900 -o "$TMP_DIR/warmup-response.json" -w '%{http_code}' "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "@$WARMUP_PAYLOAD" || true)"
if [ "$WARMUP_CODE" != "200" ]; then
    printf 'Warmup failed with HTTP %s; see %s\n' "$WARMUP_CODE" "$TMP_DIR/warmup-response.json" >&2
    exit 1
fi

printf '\nBENCHMARK model=%s url=%s runs/size=%s output_tokens=%s\n' "$MODEL" "$BASE_URL" "$RUNS_PER_SIZE" "$OUTPUT_TOKENS"
printf 'Results directory: %s\n' "$OUT_DIR"

for target in "${TARGETS[@]}"; do
    PROMPT_FILE="$TMP_DIR/prompt-${target}.txt"
    for attempt in 1 2 3 4; do
        TOKENIZE_PAYLOAD="$TMP_DIR/tokenize-${target}.json"
        python3 - "$TOKENIZE_PAYLOAD" "$PROMPT_FILE" "$MODEL" <<'PY'
import json
import sys
payload, prompt_file, model = sys.argv[1:]
text = open(prompt_file, encoding="utf-8").read()
json.dump({"content": text, "model": model}, open(payload, "w"))
PY
        TOKENIZE_CODE="$(curl -sS --max-time 900 -o "$TMP_DIR/tokenize-${target}.json" -w '%{http_code}' "$BASE_URL/tokenize" -H 'Content-Type: application/json' --data-binary "@$TOKENIZE_PAYLOAD" || true)"
        if [ "$TOKENIZE_CODE" != "200" ]; then
            printf 'Tokenization failed for target %s with HTTP %s\n' "$target" "$TOKENIZE_CODE" >&2
            exit 1
        fi
        ACTUAL_PROMPT_TOKENS="$(python3 - "$TMP_DIR/tokenize-${target}.json" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1]))
tokens = data.get("tokens")
if not isinstance(tokens, list) or not tokens:
    raise SystemExit("tokenize response has no token list")
print(len(tokens))
PY
)"
        if [ "$ACTUAL_PROMPT_TOKENS" -ge $((target * 95 / 100)) ] && [ "$ACTUAL_PROMPT_TOKENS" -le $((target * 115 / 100)) ]; then
            break
        fi
        if [ "$attempt" -eq 4 ]; then
            printf 'Prompt calibration failed for target %s: actual=%s\n' "$target" "$ACTUAL_PROMPT_TOKENS" >&2
            exit 1
        fi
        python3 - "$PROMPT_FILE" "$TMP_DIR/benchmark-block.txt" "$target" "$ACTUAL_PROMPT_TOKENS" <<'PY'
import pathlib
import sys
prompt_file, block_file, target, actual = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
prefix = ("Read the following technical working document. Analyze its themes and "
          "trade-offs, then produce a concise final answer containing the phrase "
          "BENCHMARK_COMPLETED and a short list of the most important conclusions. "
          "Do not quote the document.\n\nDOCUMENT:\n\n")
block = pathlib.Path(block_file).read_text(encoding="utf-8")
old = pathlib.Path(prompt_file).read_text(encoding="utf-8")
repeats = max(1, round(max(1, old.count(block)) * target / max(1, actual)))
pathlib.Path(prompt_file).write_text(prefix + (block + "\n") * repeats, encoding="utf-8")
PY
    done

    for run in $(seq 1 "$RUNS_PER_SIZE"); do
        PAYLOAD="$TMP_DIR/payload-${target}-${run}.json"
        RESPONSE="$TMP_DIR/response-${target}-${run}.json"
        GPU_BEFORE="$TMP_DIR/gpu-before-${target}-${run}.txt"
        GPU_AFTER="$TMP_DIR/gpu-after-${target}-${run}.txt"
        python3 - "$PAYLOAD" "$PROMPT_FILE" "$MODEL" "$OUTPUT_TOKENS" <<'PY'
import json
import sys
path, prompt_file, model, output_tokens = sys.argv[1:]
prompt = open(prompt_file, encoding="utf-8").read()
json.dump({
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(output_tokens),
    "temperature": 0.0,
    "top_k": 1,
    "cache_prompt": False,
    "ignore_eos": True,
}, open(path, "w"))
PY
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits > "$GPU_BEFORE" || true
        elif [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
            /usr/lib/wsl/lib/nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits > "$GPU_BEFORE" || true
        else
            : > "$GPU_BEFORE"
        fi
        RUN_START_EPOCH="$(date +%s)"
        START_NS="$(date +%s%N)"
        HTTP_CODE="$(curl -sS --max-time 3600 -o "$RESPONSE" -w '%{http_code}' "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "@$PAYLOAD" || true)"
        END_NS="$(date +%s%N)"
        WALL_MS="$(( (END_NS - START_NS) / 1000000 ))"
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits > "$GPU_AFTER" || true
        elif [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
            /usr/lib/wsl/lib/nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits > "$GPU_AFTER" || true
        else
            : > "$GPU_AFTER"
        fi
        capture_service_kv_logs "$RUN_START_EPOCH"

        python3 - "$RESPONSE" "$RESULTS_TSV" "$target" "$ACTUAL_PROMPT_TOKENS" "$run" "$WALL_MS" "$HTTP_CODE" "$GPU_BEFORE" "$GPU_AFTER" <<'PY'
import json
import pathlib
import sys

response_file, results_file, target, actual, run, wall, code, gpu_before_file, gpu_after_file = sys.argv[1:]
try:
    data = json.load(open(response_file))
except Exception as exc:
    print(f"run={run} HTTP={code} invalid JSON: {exc}", file=sys.stderr)
    raise SystemExit(1)
if code != "200" or "error" in data:
    print(json.dumps(data, indent=2)[:2000], file=sys.stderr)
    raise SystemExit(1)

t = data.get("timings", {})
pp = float(t.get("prompt_per_second", 0.0) or 0.0)
tg = float(t.get("predicted_per_second", 0.0) or 0.0)
pred = int(t.get("predicted_n", 0) or 0)
accepted = int(t.get("draft_n_accepted", 0) or 0)
draft_n = int(t.get("draft_n", 0) or 0)

def used(path, index):
    try:
        for line in pathlib.Path(path).read_text().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if parts and parts[0] == str(index) and len(parts) > 1:
                return parts[1]
    except Exception:
        pass
    return ""

row = [target, actual, run, f"{pp:.6f}", f"{tg:.6f}", pred, accepted, draft_n, wall, used(gpu_after_file, 0), used(gpu_after_file, 1), code]
with open(results_file, "a", encoding="utf-8") as f:
    f.write("\t".join(map(str, row)) + "\n")
print(f"run={run} PP={pp:.3f} TG={tg:.3f} prompt_n={actual} predicted_n={pred} draft={accepted}/{draft_n} wall_ms={wall}")
PY
    done
done

capture_service_kv_logs "$BENCH_START_EPOCH"
printf '\nSERVICE KV/RESIDENCY LOG: %s\n' "$SERVICE_LOG"
python3 - "$SERVICE_LOG" <<'PY'
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
print("\nSERVICE KV/RESIDENCY EXCERPT")
for line in lines[-120:]:
    print(line)
PY

python3 - "$RESULTS_TSV" "$SUMMARY_JSON" "$RUN_META_JSON" "$SERVICE_LOG" <<'PY'
import csv
import json
import math
import pathlib
import statistics
import sys

results, summary_path, metadata_path, service_log = sys.argv[1:]
rows = []
with open(results, newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        for key in ("target_tokens", "actual_prompt_tokens", "run", "predicted_tokens", "draft_accepted", "draft_tokens", "wall_ms", "http_code"):
            row[key] = int(row[key])
        for key in ("pp_tok_s", "tg_tok_s"):
            row[key] = float(row[key])
        rows.append(row)
if not rows:
    raise SystemExit("No successful benchmark rows")

def group_stats(items, key):
    vals = [float(x[key]) for x in items if float(x[key]) > 0]
    if not vals:
        return {}
    return {
        "n": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }

by_target = {}
for target in sorted({r["target_tokens"] for r in rows}):
    items = [r for r in rows if r["target_tokens"] == target]
    by_target[str(target)] = {
        "actual_prompt_tokens": [r["actual_prompt_tokens"] for r in items],
        "prompt": group_stats(items, "pp_tok_s"),
        "generation": group_stats(items, "tg_tok_s"),
        "wall_ms": group_stats(items, "wall_ms"),
        "predicted_tokens": [r["predicted_tokens"] for r in items],
        "draft_accepted": sum(r["draft_accepted"] for r in items),
        "draft_tokens": sum(r["draft_tokens"] for r in items),
    }

meta = json.load(open(metadata_path))
meta["finished_utc"] = __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())
meta["service_kv_log"] = service_log
meta["rows"] = rows
meta["by_target"] = by_target
json.dump(meta, open(summary_path, "w"), indent=2)

print("\nSUMMARY")
for target, data in by_target.items():
    p, g = data["prompt"], data["generation"]
    print(f"{target:>6} tokens: actual={data['actual_prompt_tokens']} PP-median={p.get('median', 0):.3f} TG-median={g.get('median', 0):.3f} wall-median={data['wall_ms'].get('median', 0):.0f}ms")
print(f"JSON: {summary_path}")
print(f"TSV:  {results}")
PY
