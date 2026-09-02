#!/usr/bin/env python3
"""Black-box A -> B -> A agent-resume regression test for llama-server.

The test uses the native /completion endpoint so it can request token IDs and
server timing/cache fields.  It compares a warm A continuation (after B used the
slot) with a cold recomputation of the exact same prompt.

Requirements:
  - server running with --cache-prompt and cache-idle-slots enabled (default)
  - for long contexts, enough cache-ram to hold at least two serialized states
  - deterministic greedy target sampling for the equality check

Example:
  python3 scripts/test-agent-resume.py --url http://127.0.0.1:8025 --model Qwen3.8-27B-Q4

The test does not assume parallel > 1.  With parallel=1 it explicitly exercises
host-state swapping.  With parallel=2 it remains useful when more logical agents
than live slots are scheduled.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def post_json(url: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def completion(base: str, model: str, prompt: str, n_predict: int, cache_prompt: bool, timeout: float) -> dict:
    body = {
        "prompt": prompt,
        "model": model,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 123456789,
        "stream": False,
        "cache_prompt": cache_prompt,
        "return_tokens": True,
    }
    return post_json(base.rstrip("/") + "/completion", body, timeout)


def extract_tokens(resp: dict):
    toks = resp.get("tokens")
    return toks if isinstance(toks, list) else None


def cached_tokens(resp: dict) -> int:
    for key in ("tokens_cached", "n_tokens_cached"):
        if isinstance(resp.get(key), int):
            return int(resp[key])
    # Some builds expose the prompt cache count through verbose/timing fields.
    usage = resp.get("usage", {})
    detail = usage.get("prompt_tokens_details", {}) if isinstance(usage, dict) else {}
    if isinstance(detail.get("cached_tokens"), int):
        return int(detail["cached_tokens"])
    return -1


def content(resp: dict) -> str:
    value = resp.get("content", "")
    return value if isinstance(value, str) else str(value)


def make_prefix(label: str, repeat: int) -> str:
    # Repetitive but deterministic tokens make a sizable prefix without relying
    # on external files. Agent identities are deliberately disjoint.
    unit = (
        f"[{label}] persistent working context. "
        "Preserve every fact and continue deterministically. "
        "alpha beta gamma delta epsilon zeta eta theta.\n"
    )
    return unit * repeat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8025")
    ap.add_argument("--model", required=True, help="models.ini preset/model name")
    ap.add_argument("--repeat", type=int, default=800, help="size of each synthetic agent prefix")
    ap.add_argument("--tokens", type=int, default=64, help="generation tokens per turn")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--min-reuse-fraction", type=float, default=0.70,
                    help="minimum warm cached tokens as fraction of estimated A2 prompt tokens")
    args = ap.parse_args()

    a_root = make_prefix("AGENT-A", args.repeat)
    b_root = make_prefix("AGENT-B", args.repeat)

    a1_prompt = a_root + "\nUser: Give a compact checkpoint phrase.\nAssistant:"
    print("1/4 agent A initial turn")
    a1 = completion(args.url, args.model, a1_prompt, args.tokens, True, args.timeout)
    a1_text = content(a1)
    if not a1_text:
        print("ERROR: A1 returned no content", file=sys.stderr)
        return 2

    # Force another logical agent to take the only live slot when parallel=1.
    b1_prompt = b_root + "\nUser: Give a different compact checkpoint phrase.\nAssistant:"
    print("2/4 agent B interruption")
    b1 = completion(args.url, args.model, b1_prompt, args.tokens, True, args.timeout)
    if not content(b1):
        print("ERROR: B1 returned no content", file=sys.stderr)
        return 2

    # A's next prompt contains its full previous branch.  This is the critical
    # resume operation: serialized target/draft/spec state should be restored.
    a2_prompt = (
        a1_prompt + a1_text +
        "\nUser: Continue from the checkpoint and output exactly one concise sentence.\nAssistant:"
    )
    print("3/4 agent A warm resume")
    warm = completion(args.url, args.model, a2_prompt, args.tokens, True, args.timeout)

    print("4/4 identical A continuation, cold recompute")
    cold = completion(args.url, args.model, a2_prompt, args.tokens, False, args.timeout)

    warm_text = content(warm)
    cold_text = content(cold)
    warm_toks = extract_tokens(warm)
    cold_toks = extract_tokens(cold)
    cached = cached_tokens(warm)

    # The native response exposes tokens_evaluated as total input tokens on the
    # inspected server. Use it only as a denominator when available.
    total_prompt = warm.get("tokens_evaluated")
    if not isinstance(total_prompt, int) or total_prompt <= 0:
        total_prompt = len(a2_prompt.split())  # conservative diagnostic estimate

    print("\nResults")
    print(f"  warm cached tokens: {cached}")
    print(f"  prompt token/word estimate: {total_prompt}")
    if cached >= 0:
        print(f"  reuse fraction: {cached / max(total_prompt, 1):.3f}")
    print(f"  warm/cold text identical: {warm_text == cold_text}")
    if warm_toks is not None and cold_toks is not None:
        print(f"  warm/cold token IDs identical: {warm_toks == cold_toks}")

    stable = warm_text == cold_text
    if warm_toks is not None and cold_toks is not None:
        stable = stable and warm_toks == cold_toks

    reuse_ok = cached < 0 or cached >= int(args.min_reuse_fraction * total_prompt)

    if not stable:
        print("FAIL: warm restored state changed deterministic target output.", file=sys.stderr)
        return 3
    if not reuse_ok:
        print("FAIL: A was not resumed from enough cached context; inspect cache-ram and cache logs.", file=sys.stderr)
        return 4

    print("PASS: A -> B -> A resume preserved deterministic output and reused the cached prefix.")
    if cached < 0:
        print("NOTE: this server response did not expose a cached-token count; token stability passed, but verify reuse in server logs.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {body}", file=sys.stderr)
        raise SystemExit(10)
    except urllib.error.URLError as e:
        print(f"connection error: {e}", file=sys.stderr)
        raise SystemExit(11)
