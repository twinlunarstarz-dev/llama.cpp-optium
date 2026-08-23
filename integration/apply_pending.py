from pathlib import Path

p = Path('common/common.cpp')
s = p.read_text()
old = '''    if (!params.use_mmap) {\n        throw std::runtime_error("sequential MVP requires mmap");\n    }\n    if (params.use_direct_io) {\n        throw std::runtime_error("sequential MVP does not support direct I/O");\n    }\n    if (params.use_mlock) {\n        throw std::runtime_error("sequential MVP does not support mlock");\n    }\n'''
new = '''    switch (params.load_mode) {\n        case LLAMA_LOAD_MODE_AUTO:\n        case LLAMA_LOAD_MODE_MMAP:\n        case LLAMA_LOAD_MODE_DIRECT_IO:\n            break;\n        case LLAMA_LOAD_MODE_MLOCK:\n        case LLAMA_LOAD_MODE_MMAP_MLOCK:\n            throw std::runtime_error("sequential loading does not support mlock");\n        case LLAMA_LOAD_MODE_NONE:\n            throw std::runtime_error("sequential loading requires mmap or direct I/O");\n        default:\n            throw std::runtime_error("sequential loading received an unsupported load mode");\n    }\n'''
if old not in s:
    raise SystemExit('common sequential load-mode validation anchor not found')
s = s.replace(old, new, 1)

old = '''        // sequential mode requires mmap and non-no_alloc - restore after fit probing\n        if (params.sequential_load) {\n            mparams.use_mmap  = true;\n            mparams.no_alloc  = false;\n            mparams.use_mlock = false;\n        }\n'''
new = '''        // sequential mode requires real tensor descriptors after fit probing.\n        if (params.sequential_load) {\n            mparams.load_mode = params.load_mode;\n            mparams.no_alloc  = false;\n        }\n'''
if old not in s:
    raise SystemExit('common post-fit sequential anchor not found')
s = s.replace(old, new, 1)

old = '''    mparams.load_mode       = params.load_mode;\n    mparams.tensor_split    = params.tensor_split;\n    mparams.use_mmap        = params.use_mmap || params.sequential_load; // force mmap for streaming\n    mparams.use_direct_io   = params.sequential_load ? false : params.use_direct_io;\n    mparams.use_mlock       = params.sequential_load ? false : params.use_mlock; // no mlock for streaming\n    mparams.check_tensors   = params.check_tensors;\n'''
new = '''    mparams.load_mode       = params.load_mode;\n    mparams.tensor_split    = params.tensor_split;\n    mparams.check_tensors   = params.check_tensors;\n'''
if old not in s:
    raise SystemExit('common model-param legacy load booleans anchor not found')
s = s.replace(old, new, 1)
p.write_text(s)
