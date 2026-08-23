from pathlib import Path

p = Path('src/llama-context.cpp')
s = p.read_text()
old = '''        if (sequential_weight_budget > 0) {\n            ggml_backend_sched_set_max_weight_bytes_per_split(sched.get(), sequential_weight_budget);\n        }\n'''
new = '''        if (sequential_weight_budget > 0) {\n            for (ggml_backend_t backend : backend_ptrs) {\n                if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {\n                    ggml_backend_sched_set_max_weight_bytes_per_split(sched.get(), backend, sequential_weight_budget);\n                }\n            }\n        }\n'''
count = s.count(old)
if count != 2:
    raise SystemExit(f'expected 2 old max-weight-budget calls, found {count}')
s = s.replace(old, new)
p.write_text(s)
