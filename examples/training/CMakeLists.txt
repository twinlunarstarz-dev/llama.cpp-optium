set(TARGET llama-finetune)
add_executable(${TARGET} finetune.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# These reference primitives are not wired into llama_opt yet. Build their
# numerical tests explicitly so CI cannot mistake uncompiled code for support.
if (TARGET ggml-cpu)
    add_executable(llama-train-quant-test
        ../../tests/test-train-quant.cpp
        ../../src/llama-train-quant.cpp)
    target_include_directories(llama-train-quant-test PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(llama-train-quant-test PRIVATE ggml ggml-cpu)
    target_compile_features(llama-train-quant-test PRIVATE cxx_std_17)
endif()
