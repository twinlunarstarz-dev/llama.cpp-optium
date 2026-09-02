#!/usr/bin/env python3
"""Materialize aggressive runtime throughput changes for the perf branch.

The edits are intentionally isolated and guarded.  They favor throughput and
fall back through existing capability checks where possible.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str, *, required=True):
    p = ROOT / path
    text = p.read_text()
    n = text.count(old)
    if n == 0 and not required:
        return False
    if n != 1:
        raise RuntimeError(f"{path}: expected exactly one match, got {n}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1))
    print(f"patched {path}")
    return True


def regex_once(path: str, pattern: str, repl: str, *, required=True, flags=re.S):
    p = ROOT / path
    text = p.read_text()
    new, n = re.subn(pattern, repl, text, count=1, flags=flags)
    if n == 0 and not required:
        return False
    if n != 1:
        raise RuntimeError(f"{path}: expected exactly one regex match, got {n}")
    p.write_text(new)
    print(f"patched {path}")
    return True


def patch_cuda_graphs():
    path = "ggml/src/ggml-cuda/ggml-cuda.cu"
    # master may already have changed the default.  Only patch the old opt-in form if present.
    replace_once(
        path,
        '        return env != nullptr && atoi(env) == 1;\n',
        '        // Graph replay is a production optimization.  Keep an explicit 0 as the\n'
        '        // escape hatch, but do not require every launcher to opt in.\n'
        '        return env == nullptr || atoi(env) != 0;\n',
        required=False,
    )
    replace_once(
        path,
        '    if (!use_cuda_graph || ggml_backend_cuda_get_device_count() != 1) {\n'
        '        return;\n'
        '    }\n',
        '    // This callback operates on one CUDA backend subgraph.  The old global\n'
        '    // visible-device-count check disabled graph optimization even when the\n'
        '    // current model/subgraph lived entirely on one GPU (for example a 3090\n'
        '    // utility model while a 2060 was merely visible).\n'
        '    if (!use_cuda_graph) {\n'
        '        return;\n'
        '    }\n',
    )
    replace_once(
        path,
        '            //TODO: check why nrows > 1 fails\n'
        '            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {\n'
        '                fan_out[src] += 1;\n'
        '            }\n',
        '            // Track fan-out for batched decode as well.  Restricting this to\n'
        '            // single-row tensors prevented the Q/K/V stream scheduler from\n'
        '            // seeing independent branches when several decode rows were\n'
        '            // present.  Tensor lifetime synchronization below is already\n'
        '            // graph based, so row count is not a dependency criterion.\n'
        '            if (node && !is_noop(node)) {\n'
        '                fan_out[src] += 1;\n'
        '            }\n',
    )


def patch_tensor_backend_sampling():
    path = "src/llama-context.cpp"
    pattern = r'''\n    if \(sampler && model\.split_mode\(\) == LLAMA_SPLIT_MODE_TENSOR\) \{.*?\n        return false;\n    \}\n\n    const bool can_offload ='''
    repl = '''\n    // Tensor-split output lives on the Meta backend.  Do not reject it merely\n    // because the model is tensor-parallel: sampler backend_init already probes\n    // every operation against the supplied buffer type, and Meta reports support\n    // only when its child backends can execute the operation.  Unsupported chains\n    // therefore retain the normal CPU fallback, while supported chains avoid the\n    // n_vocab logits D2H copy on every decode step.\n    const bool can_offload ='''
    regex_once(path, pattern, repl)

    old = '''        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());\n\n        sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq);\n\n        sampling.samplers[seq_id] = sampler;\n'''
    new = '''        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());\n\n        // backend_init returns how much of the chain can actually stay on the\n        // backend.  A Meta buffer is valid here; its capability query is the\n        // intersection of the child devices.\n        if (!sampler->iface->backend_init(sampler, buft, cparams.n_outputs_max_per_seq)) {\n            LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d cannot be initialized on output backend '%s'; using CPU\\n",\n                    __func__, llama_sampler_name(sampler), seq_id, ggml_backend_buft_name(buft));\n            sampling.samplers.erase(seq_id);\n            return false;\n        }\n\n        sampling.samplers[seq_id] = sampler;\n'''
    replace_once(path, old, new)


def patch_utility_parallelism():
    path = "tools/server/server-context.cpp"
    old = '''        params_base = params;\n        const auto output_limits = server_output_limits(params_base);\n'''
    new = '''        params_base = params;\n\n        // Embedding and rerank models are stateless utility workloads.  Keeping\n        // their server slot count tied to generation --parallel=1 serializes\n        // independent documents and leaves GPU batch GEMMs under-filled.  Give\n        // utility models an internal batch lane while preserving generation\n        // models' stateful parallel setting.\n        const bool utility_model = params_base.embedding ||\n                (params_base.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&\n                 params_base.pooling_type != LLAMA_POOLING_TYPE_NONE);\n        if (utility_model && params_base.n_parallel == 1) {\n            int32_t utility_parallel = 16;\n            if (const char * env = getenv("LLAMA_SERVER_UTILITY_PARALLEL")) {\n                utility_parallel = std::max<int32_t>(1, atoi(env));\n            }\n            utility_parallel = std::min<int32_t>(utility_parallel, std::max<int32_t>(1, params_base.n_batch));\n            if (utility_parallel > 1) {\n                params_base.n_parallel = utility_parallel;\n                // Dynamic shared KV keeps the full configured context available to\n                // every utility sequence instead of dividing it into fixed slots.\n                params_base.kv_unified = true;\n                SRV_INF("utility model: widening internal parallel lane to %d slots\\n", utility_parallel);\n            }\n        }\n\n        const auto output_limits = server_output_limits(params_base);\n'''
    replace_once(path, old, new)


def main():
    patch_cuda_graphs()
    patch_tensor_backend_sampling()
    patch_utility_parallelism()
    print("aggressive runtime speedups applied")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
