from pathlib import Path
import re

p = Path('src/llama-context.cpp')
s = p.read_text()
needle = 'ggml_backend_sched_set_max_weight_bytes_per_split(sched.get(), sequential_weight_budget);'
if s.count(needle) != 2:
    raise SystemExit(f'expected 2 old max-weight-budget calls, found {s.count(needle)}')
pattern = re.compile(r'(?m)^(?P<indent>\s*)ggml_backend_sched_set_max_weight_bytes_per_split\(sched\.get\(\), sequential_weight_budget\);$')

def repl(match):
    i = match.group('indent')
    return (
        f'{i}for (ggml_backend_t backend : backend_ptrs) {{\n'
        f'{i}    if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {{\n'
        f'{i}        ggml_backend_sched_set_max_weight_bytes_per_split(sched.get(), backend, sequential_weight_budget);\n'
        f'{i}    }}\n'
        f'{i}}}'
    )
s, n = pattern.subn(repl, s)
if n != 2:
    raise SystemExit(f'expected 2 replacements, performed {n}')
p.write_text(s)
