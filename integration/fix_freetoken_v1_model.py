from pathlib import Path

p = Path('src/llama-model-loader.cpp')
s = p.read_text()
old = '    this->load_mtp = load_mtp;\n'
if old not in s:
    raise SystemExit('stale load_mtp assignment not found')
s = s.replace(old, '    GGML_UNUSED(load_mtp);\n', 1)
p.write_text(s)

p = Path('src/llama-model.cpp')
s = p.read_text()
anchor = '    std::string desc_str;\n\n'
field = '    std::vector<float> tensor_split_owned;\n\n'
if 'tensor_split_owned;' not in s:
    if anchor not in s:
        raise SystemExit('model impl tensor-split insertion anchor not found')
    s = s.replace(anchor, anchor + field, 1)
p.write_text(s)
