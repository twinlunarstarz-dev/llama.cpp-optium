from pathlib import Path

p = Path('src/llama-model-loader.h')
s = p.read_text()
anchor = '    bool no_alloc;\n'
field = '    bool load_mtp = false;\n'
if field not in s:
    if anchor not in s:
        raise SystemExit('loader state insertion anchor not found')
    s = s.replace(anchor, anchor + field, 1)
p.write_text(s)

p = Path('src/llama-model-loader.cpp')
s = p.read_text()
if '    GGML_UNUSED(load_mtp);\n' in s:
    s = s.replace('    GGML_UNUSED(load_mtp);\n', '    this->load_mtp = load_mtp;\n', 1)
elif '    this->load_mtp = load_mtp;\n' not in s:
    raise SystemExit('load_mtp constructor assignment location not found')
p.write_text(s)

p = Path('src/llama-model.cpp')
s = p.read_text()
anchor = '    std::string desc_str;\n\n'
field = '    std::vector<float> tensor_split_owned;\n\n'
if 'std::vector<float> tensor_split_owned;' not in s:
    if anchor not in s:
        raise SystemExit('model impl tensor-split insertion anchor not found')
    s = s.replace(anchor, anchor + field, 1)
p.write_text(s)
