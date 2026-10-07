# muscle_gpu frontend verification (2026-10-08)

The new frontend was compiled and smoke-checked independently of the historical performance results.

- Local macOS CPU-only build: `muscle` and `muscle_gpu` compiled, CTest 9/9 passed, tiny3 output byte-identical using the same original `-align/-output/-threads` arguments. `muscle_gpu` visibly reported CPU fallback because CUDA was not compiled in.
- RTX A5000 CUDA build: `muscle_gpu` compiled with two build workers. Running tiny3 with `-align`, `-output`, `-threads 2`, `-perturb 1`, `-randseed 1`, and a diagnostic trace, without `-backend gpu`, produced three `cuda-pair-v1` commit records and no commit fallback. Output was byte-identical to explicit `-backend cpu` through the same executable.
- Target-host `muscle_gpu` SHA-256: `43156e05e1b18c3fcd23cfb52e697403f468d4d477b0321144ff8d7d3530b403`.

This verifies default dispatch and the minimal frontend integration. It does not establish exhaustive compatibility across all commands, GPU quality on long/plant-family inputs, or a newly measured performance result. CUDA recurrence code was not changed during release preparation.

2026-10-08 Super5 extension: default `muscle_gpu -super5 BB11001` actually submitted CUDA pairs and matched the original CPU output; explicit 1MiB resource fallback also matched. Super4 smoke matched with GPU verification. See SUPER5.md for scope and the original EACluster reproducibility limit. Super5 speed measurements have separate recorded binary identities; subsequent help-text/CMake-test edits do not change posterior code but produce a new executable hash.
