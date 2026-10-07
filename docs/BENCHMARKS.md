# Preliminary benchmark record (2026-10-08)

Inputs: bundled BAliBASE-family examples in `muscle/test_data/fa/`. The MUSCLE5 paper evaluates protein accuracy on BAliBASE; these seven bundled examples are a small subset, not a replication of its complete benchmark or structural-quality evaluation.

Hardware: Intel Core i9-14900K, 32 logical CPUs; NVIDIA RTX A5000 24 GB. Shared server, with existing `hmmsearch` jobs and load averages around 19–21 during thread screening/comparison. No affinity, isolation or uncontrolled-CPU-concurrency baseline was used.

Toolchain: historical CUDA 13.2.51, architecture sm_86, CPU and GPU host compilation using conda GCC 14.3. Historical executable `build/gpu/muscle` SHA-256 `aa51ef6d64449f96374ecd709878c275d196d23b1d76579c24899024ffd3eb40`.

Timing includes a new process, CUDA initialization (GPU runs), transfers, all CPU stages and output writing. `/usr/bin/time` columns are wall seconds, user seconds, system seconds and maximum RSS in KiB. No `-gpu_verify` or stage trace was enabled during timing. Tasks were serial; CPU8/GPU8 ensemble runs alternated by repetition. No separate warmup was performed. Inputs/page cache may be warm.

```bash
/usr/bin/time -f '%e %U %S %M' env OMP_NUM_THREADS=8 \
  build/gpu/muscle -align muscle/test_data/fa/BB11005 \
  -diversified -replicates 10 -output cpu.efa -backend cpu -threads 8 -quiet
# Repeat with -backend gpu; compare every output using cmp.
```

Single-alignment thread screening used `-perturb 1 -randseed 1`, each thread count three times. Median seconds: CPU1 2.77, CPU2 1.68, CPU4 1.22, CPU8 0.68. Eight threads were fastest among the tested choices; this is not proof of a global optimum or optimum for every family.

Ensemble CPU8: 7.43, 6.66, 7.66 seconds (median 7.43). GPU8: 5.01, 5.14, 4.79 (median 5.01). Ratio of medians: 1.483x. Three paired EFA outputs were byte-identical. Output SHA-256: `d3d9ce27bbcec73eb214f413f76cda55d8a6bbf9952aec5e6bf85776b8f85090`.

Small-family median times (three repetitions): BB11009 CPU2 0.06 s / GPU2 0.28 s; BB11002 CPU2 0.05 s / GPU2 0.21 s. CUDA overhead exceeds compute savings at this scale. Earlier one-shot CPU2/GPU2 ensemble observations of 21.38/11.10 seconds are superseded by the CPU8 comparison for the best tested CPU baseline.

Raw TSVs are in [benchmarks/](benchmarks/). They describe historical measurements preceding the new `muscle_gpu` entry point; validation of that entry point is a separate release check. No structural reference quality score or plant-family generalization is claimed.


## Super5 measurements with the current executable

Performance executable `muscle_gpu` SHA-256 `222a547890e8cd01d010e4d2f957b504a24ba086e29efe4bdf55ceb529324b3e`; final help/CTest-only update SHA-256 `3aa4da64aca78a1b457db1cd7935b01ac2d129358b9cd791e1f88b296db112e4`. Posterior code is unchanged between them; timings belong to the performance identity.

Date 2026-10-08; same shared i9-14900K/A5000 server, conda GCC14.3/CUDA13.2, sm_86. Compilation used at most2 jobs; all runs were serial and used at most8 threads. CPU and CUDA selected by `-backend` in the same new `muscle_gpu` binary, controlling compiler/build differences. A subsequent one-shot CPU4 screen took34.35s on the same48-record four-permutation workload, with byte-identical output to CPU8; CPU8 was faster in this bounded comparison. CPU4 has only one measurement, not a three-run median. CPU8 is the chosen bounded baseline, not a proven Super5 global optimum. No timing run enabled verification or trace. `/usr/bin/time` includes process/CUDA startup, transfers, CPU pipeline and four output writes. No deliberate warmup; caches may be warm. Other server work was not suspended. GPU per-stage time and peak GPU memory were not sampled in these timing runs.

Input: first48 records of the supplied `muscle/test_data/rdrp/rdrp.fa` in its original order, without synthetic replication. This is a small real RdRp subset associated with the supplied MUSCLE test data, not a reconstruction of the paper's complete RdRp benchmark or a plant-family quality assessment. Input SHA-256 `618979738ce1052f9476748db759ae428c08470a804c151171a818aa738144df`.

```bash
python3 tools/make_rdrp_subset.py --count 48 --output rdrp_first48.fa
/usr/bin/time -f '%e %U %S %M' env OMP_NUM_THREADS=8 \
  build/gpu/muscle_gpu -super5 rdrp_first48.fa -threads 8 \
  -perturb 1 -randseed 1 -perm all -backend cpu -output 'cpu.@.afa'
# Repeat with -backend gpu and a different output pattern.
# Compare all four perm.seed files using cmp; seed=1.
```

| Backend | Run1 | Run2 | Run3 | Median | Range |
|---|---:|---:|---:|---:|---:|
| CPU8 |21.16 s|21.07 s|20.25 s|21.07 s|20.25–21.16 s|
| GPU8 |11.06 s|10.31 s|10.73 s|10.73 s|10.31–11.06 s|

Ratio of medians **1.964x**. Three alternating CPU/GPU pairs, each with4 output files, were byte-identical; CPU repetitions also share the same output hashes. Maximum host RSS in the measured runs: CPU209760–212396KiB, GPU289668–290356KiB. GPU peak memory was not measured; these RSS values are host process memory only. Raw public records: [super5_timings.json](benchmarks/super5_timings.json).

For the same48-record input with the default single tree permutation, separate3-run times were CPU8 12.78/12.07/13.32s, GPU8 5.47/5.55/5.49s; medians12.78/5.49s, **2.328x**. All three output pairs were byte-identical. These results describe a different output workload from `-perm all` and must not be mixed with it.

### Retained failed equality case

The first60-record input SHA-256 is `a5b0738a87351909bafb76a8a56d9c1831a4795f816c4bf4f50c62749c02df1b`. Its first CPU8/GPU8 pair took17.64/8.58s but outputs differed, so the speed experiment stopped. No validated acceleration ratio is claimed for this case. Two additional CPU8 diagnostic runs also differed (EACluster membership and output hashes); UClust membership/path hashes agreed. GPU bitwise Post verification passed on all accelerated pairs, with58 cluster PPP commits and UClust/consensus/PProg stages present. See [Super5 limitations](SUPER5.md). The failure is retained in [super5_diagnosis.json](benchmarks/super5_diagnosis.json), including the failed timing pair.

This bounded implementation verification does not establish biological equivalence on all inputs, a repaired original race, structural-reference accuracy, or acceleration of every Super5 stage.
