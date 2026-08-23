from pathlib import Path

p = Path('common/fit.cpp')
s = p.read_text()
old = '''    llama_model_params mparams_copy = *mparams;\n    mparams_copy.no_alloc        = true;\n    mparams_copy.use_mmap        = false;\n    mparams_copy.use_mlock       = false;\n    mparams_copy.sequential_load = false;\n'''
new = '''    llama_model_params mparams_copy = *mparams;\n    mparams_copy.no_alloc        = true;\n    mparams_copy.load_mode       = LLAMA_LOAD_MODE_NONE;\n    mparams_copy.sequential_load = false;\n'''
if old not in s:
    raise SystemExit('fit load-mode probing anchor not found')
s = s.replace(old, new, 1)
old = '''    hp_ngl         = llama_model_n_layer(model);\n    if (mparams->load_mtp) {\n        hp_ngl    += llama_model_n_layer_nextn(model);\n    }\n'''
new = '''    hp_ngl         = llama_model_n_layer(model);\n    if (cparams->ctx_type == LLAMA_CONTEXT_TYPE_MTP) {\n        hp_ngl    += llama_model_n_layer_nextn(model);\n    }\n'''
if old not in s:
    raise SystemExit('fit MTP probing anchor not found')
s = s.replace(old, new, 1)
p.write_text(s)
