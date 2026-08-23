# Current-upstream sequential integration

Base: upstream `70adb1b4cea5ee39f867792c78dc59320921eda7`.

Ported from llama.cpp-optium:
- storage-backed sequential loading
- DeepSeek sequential support/validation
- multi-GPU sequential routing and per-device weight windows

The complete model remains storage-backed and need not fit RAM or VRAM.
Overlapping textual hunks in the fork's sequential commits use the fork side while non-overlapping current-upstream code remains current. The obsolete upstream-deleted `tests/test-backend-sched.cpp` remains deleted.
