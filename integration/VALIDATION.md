# Sequential compatibility validation

- driver: v14
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
[  3%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-opt.cpp.o
[  3%] Built target vendor-hash
[  3%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/ggml-threading.cpp.o
[  3%] Building CXX object vendor/cpp-httplib/CMakeFiles/cpp-httplib.dir/httplib.cpp.o
[  3%] Building C object ggml/src/CMakeFiles/ggml-base.dir/ggml-quants.c.o
[  3%] Building CXX object ggml/src/CMakeFiles/ggml-base.dir/gguf.cpp.o
[  3%] Linking CXX shared library ../../bin/libggml-base.so
[  3%] Built target ggml-base
[  4%] Building CXX object common/CMakeFiles/llama-common-base.dir/build-info.cpp.o
[  4%] Linking CXX static library libllama-common-base.a
[  4%] Built target llama-common-base
[  4%] Building CXX object tools/ui/CMakeFiles/llama-ui-embed.dir/embed.cpp.o
[  4%] Linking CXX executable llama-ui-embed
[  4%] Built target llama-ui-embed
[  5%] Building CXX object tools/mtmd/CMakeFiles/llama-llava-cli.dir/deprecation-warning.cpp.o
[  5%] Linking CXX executable ../../bin/llama-llava-cli
[  5%] Built target llama-llava-cli
[  5%] Building CXX object tools/mtmd/CMakeFiles/llama-gemma3-cli.dir/deprecation-warning.cpp.o
[  5%] Linking CXX executable ../../bin/llama-gemma3-cli
[  5%] Built target llama-gemma3-cli
[  5%] Building CXX object tools/mtmd/CMakeFiles/llama-minicpmv-cli.dir/deprecation-warning.cpp.o
[  5%] Linking CXX executable ../../bin/llama-minicpmv-cli
[  5%] Built target llama-minicpmv-cli
[  6%] Building CXX object tools/mtmd/CMakeFiles/llama-qwen2vl-cli.dir/deprecation-warning.cpp.o
[  6%] Linking CXX executable ../../bin/llama-qwen2vl-cli
[  6%] Built target llama-qwen2vl-cli
[  6%] Building C object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu.c.o
[  7%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ggml-cpu.cpp.o
[  7%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/repack.cpp.o
[  7%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/hbm.cpp.o
[  7%] Building C object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/quants.c.o
[  7%] Linking CXX static library libcpp-httplib.a
[  7%] Built target cpp-httplib
[  7%] Provisioning UI assets
-- UI: running npm ci
[  7%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/traits.cpp.o
[  8%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/amx/amx.cpp.o
[  8%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/amx/mmq.cpp.o
[  8%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/binary-ops.cpp.o
[  8%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/unary-ops.cpp.o
[  8%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/vec.cpp.o
[  8%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/ops.cpp.o

added 1067 packages, and audited 1068 packages in 16s

363 packages are looking for funding
  run `npm fund` for details

2 vulnerabilities (1 moderate, 1 high)

To address issues that do not require attention, run:
  npm audit fix

To address all issues, run:
  npm audit fix --force

Run `npm audit` for details.
-- UI: running npm run build, output -> /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/dist

> llama-ui@1.0.0 build
> npm run build-pwa-assets && vite build


> llama-ui@1.0.0 build-pwa-assets
> npx @vite-pwa/assets-generator --root . --config pwa-assets.config.ts && npx @vite-pwa/assets-generator --root . --config pwa-assets-dark.config.ts && node scripts/make-icons-circular.js

[log] [32mZero Config PWA Assets Generator v1.0.2[39m
[start] Preparing to generate PWA assets...
[start] Resolving instructions...
[ready] PWA assets ready to be generated, instructions resolved
[start] Generating PWA assets from static/favicon.svg image
[start] Generating assets for static/favicon.svg...
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/pwa-64x64.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/pwa-192x192.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/pwa-512x512.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/maskable-icon-512x512.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-touch-icon-180x180.png[39m
[ready] [32mGenerated ICO file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/favicon.ico[39m
[  9%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/llamafile/sgemm.cpp.o
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-2532x1170.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-1170x2532.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-2778x1284.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-1284x2778.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-2796x1290.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-1179x2556.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-1290x2796.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-2556x1179.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-1206x2622.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-2622x1206.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-1320x2868.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-640x1136.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-1136x640.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-1334x750.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-750x1334.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-2868x1320.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-1640x2360.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-1668x2388.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-2360x1640.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-2048x2732.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-2732x2048.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-1488x2266.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-2388x1668.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-dark-1170x2532.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-dark-2532x1170.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-dark-1284x2778.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-2266x1488.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-dark-1179x2556.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-dark-2556x1179.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-dark-2778x1284.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-dark-1290x2796.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-dark-2796x1290.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-dark-640x1136.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-dark-1136x640.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-dark-1206x2622.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-dark-2622x1206.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-dark-2868x1320.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-dark-750x1334.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-dark-1334x750.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-dark-1320x2868.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-dark-1640x2360.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-dark-2360x1640.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-dark-1668x2388.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-dark-2388x1668.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-dark-2048x2732.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-portrait-dark-1488x2266.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-dark-2732x2048.png[39m
[ready] [32mGenerated PNG file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/apple-splash-landscape-dark-2266x1488.png[39m
[ready] Assets generated for static/favicon.svg
[start] Generating Html Head Links...
<link rel="icon" href="/favicon.ico" sizes="48x48">
<link rel="icon" href="/favicon.svg" sizes="any" type="image/svg+xml">
<link rel="apple-touch-icon" href="/apple-touch-icon-180x180.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 390px) and (device-height: 844px) and (-webkit-device-pixel-ratio: 3) and (orientation: portrait)" href="./apple-splash-portrait-1170x2532.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 390px) and (device-height: 844px) and (-webkit-device-pixel-ratio: 3) and (orientation: landscape)" href="./apple-splash-landscape-2532x1170.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 390px) and (device-height: 844px) and (-webkit-device-pixel-ratio: 3) and (orientation: portrait)" href="./apple-splash-portrait-dark-1170x2532.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 390px) and (device-height: 844px) and (-webkit-device-pixel-ratio: 3) and (orientation: landscape)" href="./apple-splash-landscape-dark-2532x1170.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 428px) and (device-height: 926px) and (-webkit-device-pixel-ratio: 3) and (orientation: portrait)" href="./apple-splash-portrait-1284x2778.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 428px) and (device-height: 926px) and (-webkit-device-pixel-ratio: 3) and (orientation: landscape)" href="./apple-splash-landscape-2778x1284.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 428px) and (device-height: 926px) and (-webkit-device-pixel-ratio: 3) and (orientation: portrait)" href="./apple-splash-portrait-dark-1284x2778.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 428px) and (device-height: 926px) and (-webkit-device-pixel-ratio: 3) and (orientation: landscape)" href="./apple-splash-landscape-dark-2778x1284.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 393px) and (device-height: 852px) and (-webkit-device-pixel-ratio: 3) and (orientation: portrait)" href="./apple-splash-portrait-1179x2556.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 393px) and (device-height: 852px) and (-webkit-device-pixel-ratio: 3) and (orientation: landscape)" href="./apple-splash-landscape-2556x1179.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 393px) and (device-height: 852px) and (-webkit-device-pixel-ratio: 3) and (orientation: portrait)" href="./apple-splash-portrait-dark-1179x2556.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 393px) and (device-height: 852px) and (-webkit-device-pixel-ratio: 3) and (orientation: landscape)" href="./apple-splash-landscape-dark-2556x1179.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 430px) and (device-height: 932px) and (-webkit-device-pixel-ratio: 3) and (orientation: portrait)" href="./apple-splash-portrait-1290x2796.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 430px) and (device-height: 932px) and (-webkit-device-pixel-ratio: 3) and (orientation: landscape)" href="./apple-splash-landscape-2796x1290.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 430px) and (device-height: 932px) and (-webkit-device-pixel-ratio: 3) and (orientation: portrait)" href="./apple-splash-portrait-dark-1290x2796.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 430px) and (device-height: 932px) and (-webkit-device-pixel-ratio: 3) and (orientation: landscape)" href="./apple-splash-landscape-dark-2796x1290.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 402px) and (device-height: 874px) and (-webkit-device-pixel-ratio: 3) and (orientation: portrait)" href="./apple-splash-portrait-1206x2622.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 402px) and (device-height: 874px) and (-webkit-device-pixel-ratio: 3) and (orientation: landscape)" href="./apple-splash-landscape-2622x1206.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 402px) and (device-height: 874px) and (-webkit-device-pixel-ratio: 3) and (orientation: portrait)" href="./apple-splash-portrait-dark-1206x2622.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 402px) and (device-height: 874px) and (-webkit-device-pixel-ratio: 3) and (orientation: landscape)" href="./apple-splash-landscape-dark-2622x1206.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 440px) and (device-height: 956px) and (-webkit-device-pixel-ratio: 3) and (orientation: portrait)" href="./apple-splash-portrait-1320x2868.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 440px) and (device-height: 956px) and (-webkit-device-pixel-ratio: 3) and (orientation: landscape)" href="./apple-splash-landscape-2868x1320.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 440px) and (device-height: 956px) and (-webkit-device-pixel-ratio: 3) and (orientation: portrait)" href="./apple-splash-portrait-dark-1320x2868.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 440px) and (device-height: 956px) and (-webkit-device-pixel-ratio: 3) and (orientation: landscape)" href="./apple-splash-landscape-dark-2868x1320.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 320px) and (device-height: 568px) and (-webkit-device-pixel-ratio: 2) and (orientation: portrait)" href="./apple-splash-portrait-640x1136.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 320px) and (device-height: 568px) and (-webkit-device-pixel-ratio: 2) and (orientation: landscape)" href="./apple-splash-landscape-1136x640.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 320px) and (device-height: 568px) and (-webkit-device-pixel-ratio: 2) and (orientation: portrait)" href="./apple-splash-portrait-dark-640x1136.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 320px) and (device-height: 568px) and (-webkit-device-pixel-ratio: 2) and (orientation: landscape)" href="./apple-splash-landscape-dark-1136x640.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 375px) and (device-height: 667px) and (-webkit-device-pixel-ratio: 2) and (orientation: portrait)" href="./apple-splash-portrait-750x1334.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 375px) and (device-height: 667px) and (-webkit-device-pixel-ratio: 2) and (orientation: landscape)" href="./apple-splash-landscape-1334x750.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 375px) and (device-height: 667px) and (-webkit-device-pixel-ratio: 2) and (orientation: portrait)" href="./apple-splash-portrait-dark-750x1334.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 375px) and (device-height: 667px) and (-webkit-device-pixel-ratio: 2) and (orientation: landscape)" href="./apple-splash-landscape-dark-1334x750.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 820px) and (device-height: 1180px) and (-webkit-device-pixel-ratio: 2) and (orientation: portrait)" href="./apple-splash-portrait-1640x2360.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 820px) and (device-height: 1180px) and (-webkit-device-pixel-ratio: 2) and (orientation: landscape)" href="./apple-splash-landscape-2360x1640.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 820px) and (device-height: 1180px) and (-webkit-device-pixel-ratio: 2) and (orientation: portrait)" href="./apple-splash-portrait-dark-1640x2360.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 820px) and (device-height: 1180px) and (-webkit-device-pixel-ratio: 2) and (orientation: landscape)" href="./apple-splash-landscape-dark-2360x1640.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 1024px) and (device-height: 1366px) and (-webkit-device-pixel-ratio: 2) and (orientation: portrait)" href="./apple-splash-portrait-2048x2732.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 1024px) and (device-height: 1366px) and (-webkit-device-pixel-ratio: 2) and (orientation: landscape)" href="./apple-splash-landscape-2732x2048.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 1024px) and (device-height: 1366px) and (-webkit-device-pixel-ratio: 2) and (orientation: portrait)" href="./apple-splash-portrait-dark-2048x2732.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 1024px) and (device-height: 1366px) and (-webkit-device-pixel-ratio: 2) and (orientation: landscape)" href="./apple-splash-landscape-dark-2732x2048.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 834px) and (device-height: 1194px) and (-webkit-device-pixel-ratio: 2) and (orientation: portrait)" href="./apple-splash-portrait-1668x2388.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 834px) and (device-height: 1194px) and (-webkit-device-pixel-ratio: 2) and (orientation: landscape)" href="./apple-splash-landscape-2388x1668.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 834px) and (device-height: 1194px) and (-webkit-device-pixel-ratio: 2) and (orientation: portrait)" href="./apple-splash-portrait-dark-1668x2388.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 834px) and (device-height: 1194px) and (-webkit-device-pixel-ratio: 2) and (orientation: landscape)" href="./apple-splash-landscape-dark-2388x1668.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 744px) and (device-height: 1133px) and (-webkit-device-pixel-ratio: 2) and (orientation: portrait)" href="./apple-splash-portrait-1488x2266.png">
<link rel="apple-touch-startup-image" media="screen and (device-width: 744px) and (device-height: 1133px) and (-webkit-device-pixel-ratio: 2) and (orientation: landscape)" href="./apple-splash-landscape-2266x1488.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 744px) and (device-height: 1133px) and (-webkit-device-pixel-ratio: 2) and (orientation: portrait)" href="./apple-splash-portrait-dark-1488x2266.png">
<link rel="apple-touch-startup-image" media="screen and (prefers-color-scheme: dark) and (device-width: 744px) and (device-height: 1133px) and (-webkit-device-pixel-ratio: 2) and (orientation: landscape)" href="./apple-splash-landscape-dark-2266x1488.png">
[ready] Html Head Links generated
[start] Generating PWA web manifest icons entry...
{
  "icons": [
    {
      "src": "pwa-64x64.png",
      "sizes": "64x64",
      "type": "image/png"
    },
    {
      "src": "pwa-192x192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "pwa-512x512.png",
      "sizes": "512x512",
      "type": "image/png"
    },
    {
      "src": "maskable-icon-512x512.png",
      "sizes": "512x512",
      "type": "image/png",
      "purpose": "maskable"
    }
  ]
}
[ready] PWA web manifest icons entry generated
[ready] PWA assets generated
[log] [32mZero Config PWA Assets Generator v1.0.2[39m
[start] Preparing to generate PWA assets...
[start] Resolving instructions...
[ready] PWA assets ready to be generated, instructions resolved
[start] Generating PWA assets from static/favicon-dark.svg image
[start] Generating assets for static/favicon-dark.svg...
[ready] [32mGenerated ICO file: /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui-src/static/favicon-dark.ico[39m
[ready] Assets generated for static/favicon-dark.svg
[start] Generating Html Head Links...
<link rel="icon" href="/favicon-dark.ico" sizes="48x48">
<link rel="icon" href="/favicon-dark.svg" sizes="any" type="image/svg+xml">
[ready] Html Head Links generated
[start] Generating PWA web manifest icons entry...
{
  "icons": []
}
[ready] PWA web manifest icons entry generated
[ready] PWA assets generated
Circular mask: 0% padding, 85% scale, source=maskable-icon-512x512.png

✓ pwa-64x64.png → circle from maskable-icon-512x512.png, 0% padding (size=64, r=32, scale=85%, circleDiameter=64)
✓ pwa-192x192.png → circle from maskable-icon-512x512.png, 0% padding (size=192, r=96, scale=85%, circleDiameter=192)
✓ pwa-512x512.png → circle from maskable-icon-512x512.png, 0% padding (size=512, r=256, scale=85%, circleDiameter=512)

Unchanged:
  maskable-icon-512x512.png (1894 bytes)
  apple-touch-icon-180x180.png (806 bytes)
[  9%] Building C object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/x86/quants.c.o
[  9%] Building CXX object ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/arch/x86/repack.cpp.o
[36mvite v7.3.6 [32mbuilding ssr environment for production...[36m[39m
transforming...
[  9%] Linking CXX shared library ../../bin/libggml-cpu.so
[  9%] Built target ggml-cpu
[  9%] Building CXX object ggml/src/CMakeFiles/ggml.dir/ggml-backend-dl.cpp.o
[  9%] Building CXX object ggml/src/CMakeFiles/ggml.dir/ggml-backend-reg.cpp.o
[  9%] Linking CXX shared library ../../bin/libggml.so
[  9%] Built target ggml
[  9%] Building CXX object src/CMakeFiles/llama.dir/llama.cpp.o
/home/runner/work/llama.cpp-optium/llama.cpp-optium/src/llama.cpp: In function ‘std::pair<int, llama_model*> llama_model_load(gguf_context*, llama_model_set_tensor_data_t, void*, const std::string&, std::vector<std::__cxx11::basic_string<char> >&, FILE*, llama_model_params&)’:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/src/llama.cpp:319:59: error: ‘struct llama_model_params’ has no member named ‘load_mtp’; did you mean ‘load_mode’?
  319 |             params.check_tensors, params.no_alloc, params.load_mtp, params.kv_overrides, params.tensor_buft_overrides);
      |                                                           ^~~~~~~~
      |                                                           load_mode
/home/runner/work/llama.cpp-optium/llama.cpp-optium/src/llama.cpp:350:18: error: ‘llama_model_arch_supports_sequential_load’ was not declared in this scope
  350 |             if (!llama_model_arch_supports_sequential_load(model->arch)) {
      |                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/src/llama.cpp:369:24: error: ‘struct llama_model_params’ has no member named ‘use_mlock’
  369 |             if (params.use_mlock) {
      |                        ^~~~~~~~~
/home/runner/work/llama.cpp-optium/llama.cpp-optium/src/llama.cpp: At global scope:
/home/runner/work/llama.cpp-optium/llama.cpp-optium/src/llama.cpp:413:6: warning: no previous declaration for ‘bool llama_model_arch_supports_sequential_load(llm_arch)’ [-Wmissing-declarations]
  413 | bool llama_model_arch_supports_sequential_load(const llm_arch arch) {
      |      ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
gmake[2]: *** [src/CMakeFiles/llama.dir/build.make:79: src/CMakeFiles/llama.dir/llama.cpp.o] Error 1
gmake[1]: *** [CMakeFiles/Makefile2:2586: src/CMakeFiles/llama.dir/all] Error 2
gmake[1]: *** Waiting for unfinished jobs....
[32m✓[39m 5306 modules transformed.
rendering chunks...
[36mvite v7.3.6 [32mbuilding client environment for production...[36m[39m
transforming...
[32m✓[39m 8374 modules transformed.
rendering chunks...
computing gzip size...
[2m.svelte-kit/output/client/[22m[32m_app/version.json                          [39m[1m[2m    0.03 kB[22m[1m[22m[2m │ gzip:     0.05 kB[22m
[2m.svelte-kit/output/client/[22m[32m.vite/manifest.json                        [39m[1m[2m    0.30 kB[22m[1m[22m[2m │ gzip:     0.19 kB[22m
[2m.svelte-kit/output/client/[22m[32mmanifest.webmanifest                       [39m[1m[2m    0.52 kB[22m[1m[22m
[2m.svelte-kit/output/client/[22m[35m_app/immutable/assets/bundle.opWWFVFD.css  [39m[1m[2m  538.45 kB[22m[1m[22m[2m │ gzip:   291.94 kB[22m
[2m.svelte-kit/output/client/[22m[36m_app/immutable/bundle.D68ZKiS6.js          [39m[1m[33m8,845.42 kB[39m[22m[2m │ gzip: 2,609.22 kB[22m
[32m✓ built in 24.56s[39m
Relativized base refs in index.html and sw.js
[2m.svelte-kit/output/server/[22m[32m.vite/manifest.json                                                        [39m[1m[2m 15.50 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[35m_app/immutable/assets/_page.CV-KWLNP.css                                   [39m[1m[2m  0.29 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[35m_app/immutable/assets/dialog-description.bHHIbcsu.css                      [39m[1m[2m  0.35 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[35m_app/immutable/assets/_layout.BNz1JCli.css                                 [39m[1m[2m147.51 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[35m_app/immutable/assets/SidebarNavigationConversationItem.Cd5WOTuf.css       [39m[1m[2m389.75 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/app.constants.js                                                    [39m[1m[2m  0.07 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/internal.js                                                         [39m[1m[2m  0.07 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36menv.js                                                                     [39m[1m[2m  0.15 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/server.js                                                           [39m[1m[2m  0.20 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/arrow-right.js                                                      [39m[1m[2m  0.22 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/device.svelte.js                                                    [39m[1m[2m  0.27 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/keyboard.enums.js                                                   [39m[1m[2m  0.29 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/pencil.js                                                           [39m[1m[2m  0.33 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/refresh-cw.js                                                       [39m[1m[2m  0.35 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36minternal.js                                                                [39m[1m[2m  0.37 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/card.js                                                             [39m[1m[2m  0.45 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/api-key-validation.js                                               [39m[1m[2m  0.46 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/init.js                                                             [39m[1m[2m  0.48 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/index.js                                                            [39m[1m[2m  0.50 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/utils2.js                                                           [39m[1m[2m  0.60 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mentries/pages/(chat)/_page.ts.js                                           [39m[1m[2m  0.62 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/use-marquee-selection.svelte.js                                     [39m[1m[2m  0.65 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mentries/pages/(chat)/chat/_id_/_page.ts.js                                 [39m[1m[2m  0.66 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mentries/pages/settings/_layout.svelte.js                                   [39m[1m[2m  0.88 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/TruncatedText.js                                                    [39m[1m[2m  0.98 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mentries/pages/(chat)/_page.svelte.js                                       [39m[1m[2m  1.08 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mentries/pages/(chat)/chat/_id_/_page.svelte.js                             [39m[1m[2m  1.17 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/alert-dialog-cancel.js                                              [39m[1m[2m  1.29 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/label.js                                                            [39m[1m[2m  1.56 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/ActionIcon.js                                                       [39m[1m[2m  1.57 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/input.js                                                            [39m[1m[2m  1.67 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mentries/pages/search/_page.svelte.js                                       [39m[1m[2m  2.01 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/DialogConfirmation.js                                               [39m[1m[2m  2.43 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/exports.js                                                          [39m[1m[2m  2.56 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/DialogModelNotAvailable.js                                          [39m[1m[2m  2.74 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/internal2.js                                                        [39m[1m[2m  2.84 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/SidebarNavigationSearchResults.js                                   [39m[1m[2m  5.22 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/workbox-window.prod.es5.js                                          [39m[1m[2m  5.75 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/dialog-footer.js                                                    [39m[1m[2m  5.77 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/index2.js                                                           [39m[1m[2m  6.85 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mentries/pages/_error.svelte.js                                             [39m[1m[2m  8.21 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/checkbox.js                                                         [39m[1m[2m  8.52 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/index3.js                                                           [39m[1m[2m 12.07 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/utils.js                                                            [39m[1m[2m 15.02 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/popper-layer-force-mount.js                                         [39m[1m[2m 15.45 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mremote-entry.js                                                            [39m[1m[2m 15.69 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/index4.js                                                           [39m[1m[2m 21.71 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/dialog-description.js                                               [39m[1m[2m 24.57 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mentries/pages/mcp-servers/_page.svelte.js                                  [39m[1m[2m 33.49 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mentries/pages/_layout.svelte.js                                            [39m[1m[2m 39.96 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/root.js                                                             [39m[1m[2m 41.41 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/index.svelte.js                                                     [39m[1m[2m 41.90 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mindex.js                                                                   [39m[1m[2m 57.22 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/uuid.js                                                             [39m[1m[2m 69.86 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mentries/pages/settings/__section__/_page.svelte.js                         [39m[1m[2m 76.09 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/SidebarNavigationConversationItem.svelte_svelte_type_style_lang.js  [39m[1m[2m207.41 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mentries/pages/(chat)/_layout.svelte.js                                     [39m[1m[2m348.04 kB[22m[1m[22m
[2m.svelte-kit/output/server/[22m[36mchunks/_virtual_nerdamer.js                                                [39m[1m[2m483.95 kB[22m[1m[22m
[32m✓ built in 40.33s[39m

[36mPWA v1.3.0[39m
mode      [35mgenerateSW[39m
precache  [32m67 entries[39m [2m(9597.69 KiB)[22m
files generated
  [2m.svelte-kit/output/server/sw.js[22m
  [2m.svelte-kit/output/server/workbox-b3c04f83.js[22m

Run npm run preview to preview your production build locally.

> Using @sveltejs/adapter-static
Overwriting /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/dist/index.html with fallback page. Consider using a different name for the fallback.
  Wrote site to "/home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/dist"
  ✔ done
Generated 48 apple-splash link tags
Updated index.html
Created build.json (version: 10598)
Relativized base refs in index.html and sw.js
-- UI: npm build succeeded
-- UI: gzip compression applied (/home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/dist/_gzip)
embed: write output file /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui.h
embed: write output file /home/runner/work/llama.cpp-optium/llama.cpp-optium/build/tools/ui/ui.cpp
[  9%] Built target llama-ui-assets
gmake: *** [Makefile:146: all] Error 2
```
