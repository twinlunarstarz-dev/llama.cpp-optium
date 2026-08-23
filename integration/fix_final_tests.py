from pathlib import Path

# Keep the sequential CUDA fixture useful on current APIs without relying on the
# removed architecture allowlist helper from the older branch base.
p = Path('tests/test-llama-archs.cpp')
s = p.read_text()
if '#include <filesystem>\n' not in s:
    s = s.replace('#include <cstdint>\n', '#include <cstdint>\n#include <filesystem>\n', 1)
start = s.find('static void test_sequential_arch_allowlist() {')
if start >= 0:
    end = s.find('struct error_stats {', start)
    if end < 0:
        raise SystemExit('sequential allowlist end anchor missing')
    s = s[:start] + s[end:]
s = s.replace(
    '    model_params.sequential_load = sequential;\n    model_params.use_mmap = true;\n',
    '    model_params.sequential_load = sequential;\n    model_params.load_mode = LLAMA_LOAD_MODE_MMAP;\n    model_params.n_gpu_layers = 99;\n',
    1)
s = s.replace(
    '    if (!llama_model_arch_supports_sequential_load(arch)) {\n        throw std::runtime_error("architecture is not in the sequential allowlist");\n    }\n',
    '',
    1)
s = s.replace('get_gguf_ctx(arch, false, true)', 'get_gguf_ctx(arch, false)', 1)
p.write_text(s)

# test-chat is a server test and must not be synthesized as a bare -lserver-context
# dependency when LLAMA_BUILD_SERVER=OFF.
p = Path('tests/CMakeLists.txt')
s = p.read_text()
old = '''    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})\n    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)\n    target_link_libraries(test-chat PRIVATE server-context)\n'''
new = '''    if (TARGET server-context)\n        llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})\n        target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)\n        target_link_libraries(test-chat PRIVATE server-context)\n    endif()\n'''
if old not in s:
    raise SystemExit('test-chat CMake anchor missing')
s = s.replace(old, new, 1)
p.write_text(s)

Path(__file__).unlink()
