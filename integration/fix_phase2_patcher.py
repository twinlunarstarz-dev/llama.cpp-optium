from pathlib import Path

p = Path('integration/apply_pending.py')
s = p.read_text()
old = '''count = s.count(needle)\nif count != 2:\n    raise SystemExit(f'expected two sequential scheduler setup blocks, found {count}')\ns = s.replace(needle, insert)\np.write_text(s)\n'''
new = '''count = s.count(needle)\nif count != 1:\n    raise SystemExit(f'expected one top-level sequential scheduler setup block, found {count}')\ns = s.replace(needle, insert, 1)\nnested_needle = ''.join(('            ' + line) if line.strip() else line for line in needle.splitlines(True))\nnested_insert = ''.join(('            ' + line) if line.strip() else line for line in insert.splitlines(True))\ncount_nested = s.count(nested_needle)\nif count_nested != 1:\n    raise SystemExit(f'expected one nested sequential scheduler setup block, found {count_nested}')\ns = s.replace(nested_needle, nested_insert, 1)\np.write_text(s)\n'''
if old not in s:
    raise SystemExit('phase2 context patcher tail not found')
p.write_text(s.replace(old, new, 1))
