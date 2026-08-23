# Sequential allocation compatibility validation

- driver: v13
- configure rc: 0
- build rc: 2
- focused tests rc: 99

## Build
```text
[  1%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml.c.o
[  1%] Building CXX object vendor/hash/CMakeFiles/vendor-hash.dir/hash.cpp.o
[  1%] Building C object vendor/hash/CMakeFiles/vendor-hash.dir/xxhash/xxhash.c.o
[  1%] Building CXX object vendor/hash/CMakeFiles/vendor-hash.dir/sha1/sha1.c.o
[  1%] Building C object vendor/hash/CMakeFiles/vendor-hash.dir/sha256/sha256.c.o
[  1%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml.cpp.o
[  2%] Linking CXX static library libvendor-hash.a
[  2%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml-alloc.c.o
[  2%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-backend.cpp.o
[  2%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-backend-meta.cpp.o
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend-meta.cpp: In function ‘void ggml_backend_meta_device_get_props(ggml_backend_dev_t, ggml_backend_dev_props*)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend-meta.cpp:136:5: error: no match for ‘operator=’ (operand types are ‘ggml_backend_dev_caps’ and ‘<brace-enclosed initializer list>’)
  136 |     };
      |     ^
In file included from /home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend-meta.cpp:3:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/../include/ggml-backend.h:148:12: note: candidate: ‘constexpr ggml_backend_dev_caps& ggml_backend_dev_caps::operator=(const ggml_backend_dev_caps&)’
  148 |     struct ggml_backend_dev_caps {
      |            ^~~~~~~~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/../include/ggml-backend.h:148:12: note:   no known conversion for argument 1 from ‘<brace-enclosed initializer list>’ to ‘const ggml_backend_dev_caps&’
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/../include/ggml-backend.h:148:12: note: candidate: ‘constexpr ggml_backend_dev_caps& ggml_backend_dev_caps::operator=(ggml_backend_dev_caps&&)’
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/../include/ggml-backend.h:148:12: note:   no known conversion for argument 1 from ‘<brace-enclosed initializer list>’ to ‘ggml_backend_dev_caps&&’
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend-meta.cpp:144:21: error: ‘struct ggml_backend_dev_caps’ has no member named ‘mmap_support’
  144 |         props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
      |                     ^~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend-meta.cpp:144:56: error: ‘struct ggml_backend_dev_caps’ has no member named ‘mmap_support’
  144 |         props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
      |                                                        ^~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/ggml/src/ggml-backend-meta.cpp:144:95: error: ‘struct ggml_backend_dev_caps’ has no member named ‘mmap_support’
  144 |         props->caps.mmap_support         = props->caps.mmap_support         && tmp_props.caps.mmap_support;
      |                                                                                               ^~~~~~~~~~~~
gmake[2]: *** [ggml/src/CMakeFiles/ggml-base.dir/build.make:135: ggml/src/CMakeFiles/ggml-base.dir/ggml-backend-meta.cpp.o] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:2487: ggml/src/CMakeFiles/ggml-base.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
[  2%] Built target vendor-hash
gmake: *** [Makefile:146: all] Error 2
```
