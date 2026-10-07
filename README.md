# muscle_gpu

CUDA acceleration for amino-acid MUSCLE5 PPP and Super5 Pair-HMM posterior stages. This is a modified research implementation derived from Robert C. Edgar's MUSCLE, not an official MUSCLE release or certification.

Use the same MUSCLE command-line arguments, replacing the executable name:

```bash
muscle_gpu -align proteins.fa -output proteins.afa -threads 8
muscle_gpu -super5 proteins.fa -output proteins.super5.afa -threads 8
muscle_gpu -align proteins.fa -diversified -replicates 10 -output ensemble.efa -threads 8
```

The same parser, commands, alignment parameters and output formats are used by both executables. `muscle` defaults to the original CPU posterior loop. `muscle_gpu` defaults to CUDA for `-align`, `-super5` and `-super4` when a usable CUDA device is available, and reports CPU fallback otherwise. Unsupported paths (Mega/structure inputs, nucleotide inputs) use the existing CPU implementation. Super5 is a hybrid implementation: UClust candidate pairs, cluster-local PPP, consensus distances and PProg sampled pairs use CUDA posteriors; EACluster decisions, traceback, sparse consistency, profile alignment and member insertion remain on CPU. See [Super5 implementation and verification](docs/SUPER5.md). Tiny workloads may run slower on CUDA.

Optional controls from this fork remain available: `-backend cpu|gpu|auto`, `-gpu_device N`, `-gpu_mem_mb N`, `-gpu_verify`, `-stage_trace FILE`, and `-cpu_batch`. Explicit `-backend gpu` fails if CUDA is unavailable. `auto` retains the existing CPU behavior pending a validated workload selector. `-gpu_verify` recomputes pairs on the CPU and is for correctness checks, not speed measurements.

## Build

Requires CMake >=3.18, C++17 and OpenMP. CUDA builds additionally require an NVIDIA CUDA toolkit compatible with the host compiler/driver. Tested GPU: RTX A5000 (24 GB, compute capability 8.6), CUDA toolkit 13.2.51. Other combinations are not yet validated.

```bash
cmake -S muscle -B build/gpu -DCMAKE_BUILD_TYPE=Release \
  -DMUSCLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build/gpu --parallel 4
./build/gpu/muscle_gpu -align muscle/tests/fixtures/tiny3.fa -output tiny.afa
cmake --install build/gpu --prefix "$HOME/.local"
```

CPU-only build (both executable names work, with reported CPU fallback for `muscle_gpu`):

```bash
cmake -S muscle -B build/cpu -DCMAKE_BUILD_TYPE=Release -DMUSCLE_CUDA=OFF
cmake --build build/cpu --parallel 4
ctest --test-dir build/cpu --output-on-failure
```

The GPU preserves the five-state log-domain Forward/Backward recurrences, polynomial LOG_ADD, thresholds and operand order. Posterior conversion, expected-accuracy scoring and sparse matrix construction remain on the CPU. No FP16, fast math, banding or pruning is introduced.

## Measured results

Super5 on the first48 bundled RdRp sequences, `-perm all`, took **21.07 s CPU8 vs10.73 s GPU8** median: **1.96x** end-to-end, three alternating repetitions, all four outputs byte-identical each time. Original CPU EACluster non-reproducibility was observed on a60-sequence input; that failed equality case is documented and excluded from validated speedup claims. See [Super5 results](docs/BENCHMARKS.md#super5-measurements-with-the-current-executable).

For BB11005 diversified 10-replicate ensembles, CPU8 took 7.43 s median (6.66–7.66 s) and GPU8 took 5.01 s (4.79–5.14 s), a **1.48x** median end-to-end speedup. Each backend ran three times, serially, on a shared i9-14900K/A5000 server with other CPU work. All three CPU/GPU EFA pairs were byte-identical. This is a preliminary loaded-server result, not a general speedup guarantee or a comparison against unrestricted CPU concurrency.

These measurements used the previous `muscle -backend gpu` entry point, not the newly added default-selection frontend. See [benchmark protocol and raw timings](docs/BENCHMARKS.md). Previously recorded bounded verification included CPU CTest 9/9, 75 pair-oracle cases with zero reported numeric/support differences, tiny-suite CUDA memcheck with zero errors, and byte-identical selected BAliBASE outputs. Complete T06 validation, long plant families, racecheck, R1 memory-safety fixes and further Super5 validation remain outstanding.

## License and attribution

Distributed under GNU GPLv3; see [LICENSE](LICENSE) and [modification notice](NOTICE.md) and [third-party notices](THIRD_PARTY_NOTICES.md). Upstream copyrights and notices are retained. When distributing binaries, also provide the corresponding source and build scripts under the applicable GPL terms. No claim of upstream endorsement is made.

Please cite the original method:

Robert C. Edgar (2022). *Muscle5: High-accuracy alignment ensembles enable unbiased assessments of sequence homology and phylogeny*. Nature Communications 13, 6968. [DOI: 10.1038/s41467-022-34630-w](https://doi.org/10.1038/s41467-022-34630-w).

Upstream: [rcedgar/muscle](https://github.com/rcedgar/muscle). This project was developed from a supplied source archive without Git metadata; no upstream commit identity is invented. Historical R0 identity and modification provenance are documented in [NOTICE.md](NOTICE.md).
