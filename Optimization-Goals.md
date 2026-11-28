Goal: Maximize llama.cpp-optium Inference Performance
Read First
wiki/projects/llama.cpp-optium/current-state.md
Objective

Continuously optimize llama.cpp-optium for maximum accurate inference performance on this machine. Own the fork and execute improvements autonomously. Do not stop at investigation or planning, and do not ask what to do next.

Constraints
Code-only changes: Modify source code to improve performance. Do not modify model configurations or presets.
/models/models.ini is strictly read-only. Use its predefined presets for all testing and benchmarking.
No global setting or context-size changes.
Preserve unrelated user changes.
Keep dense Qwen3.8-27B on the dense execution path.
Validate MoE optimizations on Inkling-2-Mini, Nemotron-3.5-Bolt-74B, and Qwen3.8-Flash-Next (must only be done once in final verification after full optimization and must make sure to monitor ram because Qwen3.8-Flash-Next cannot fit in ram+vram and requires disk offloading).
Never sacrifice inference accuracy for speed.
Commit each independent code change separately and push accepted changes to origin.
Optimization Priorities
Speculative decoding and acceptance rates.
Reduce expert data movement per generated token.
Improve CPU/GPU execution overlap.
Optimize expert placement and memory residency.
Improve kernels, scheduling, and memory transfers.
Address any other measurable inference bottlenecks.

Do not repeat Arms A, B, or C. Continue from outstanding experiments and findings in the project state.

Benchmarking

Use short by default and only run long when in final verification:

bash
~/.hermes/workspace/optium-bench/bench_real.py \
  --model NAME --mode short|long --reps N --label L


Results: results/real.jsonl

Use existing predefined model presets without modification.
Measure prompt processing (PP), generation throughput (TG), and accuracy.
Test realistic short-chat and 49,972-token workloads.
Compare against established baselines and investigate regressions.
Build and validate every code change before benchmarking.
Never claim improvements without measurements.
Execution Rules
Prioritize isolated code optimizations.
Make one independent change at a time.
Measure every change against established baselines.
Retain verified improvements and revert regressions.
Document meaningful findings and benchmark results.
Commit and push accepted changes separately.
Never overwrite or discard unrelated user changes.
Completion

Continue autonomously until paused, explicitly cleared, or the execution budget is exhausted.

There is no fixed checklist. Continuously identify bottlenecks, implement code-level improvements, build, validate, benchmark, and retain verified gains.

Mission: Make llama.cpp-optium as fast as possible without modifying a single model preset or compromising inference accuracy.













______________________________________________________________________________















# Goal: Optimize Inkling Inference Performance

Find the fastest accurate configuration for Inkling by systematically testing viable setting combinations with `llama-cli`.

## Configuration Sources

Account for all effective settings from:

- Environment variables in the `llama-cpp` user systemd service.
- Launch options in `/opt/scripts/launch-server.sh`.
- Global settings in `/models/models.ini`.
- Inkling preset settings in `/models/models.ini`.

Preset settings override global settings.

## Optimization

- Test combinations of all relevant llama.cpp performance settings.
- Optimize PP, TG, CPU/GPU utilization, and memory usage.
- Account for interactions between settings.
- Monitor RAM and VRAM to prevent OOM and swapping.
- Never sacrifice inference accuracy for speed.
- Benchmark short-chat and long-context workloads.
- Record results and compare against the original baseline.

## Applying Results

Once the fastest verified configuration is found:

1. Write the winning settings exclusively into Inkling's preset in `/models/models.ini`.
2. Preserve global settings and every unrelated preset.
3. Ensure `llama-server` receives the same effective settings tested with `llama-cli`.
4. Verify using the actual `/opt/scripts/launch-server.sh` and systemd service.
5. Benchmark the server and confirm performance, accuracy, and stability.

## Execution Rules

- Inspect existing project state before starting.
- Make no unnecessary changes.
- Preserve unrelated user modifications.
- Never claim improvements without measurements.
- Revert regressions.
- Continue autonomously until the optimal configuration is applied or the execution budget is exhausted.

**Mission:** Systematically find the fastest accurate Inkling configuration with `llama-cli` and persist it in Inkling's `/models/models.ini` preset so `llama-server` benefits from the same optimization.
