# Super5 hybrid CUDA implementation

The amino-acid Super5 path now uses the same validated five-state Pair-HMM CUDA backend as PPP. Invoke the original MUSCLE parameters through `muscle_gpu`:

```bash
muscle_gpu -super5 input.fa -output output.afa -threads 8
muscle_gpu -super5 input.fa -perm all -output 'output.@.afa' -threads 8
```

`muscle` retains CPU defaults. `muscle_gpu` selects CUDA by default for amino-acid `-align`, `-super5` and `-super4` when a device is available. `-backend cpu` selects the original CPU posterior loops, and explicit `-backend gpu` requires CUDA. Mega/structure and nucleotide inputs remain CPU. Super5 accepts its original `-perm`/`-perturb` options; PPP `-diversified`, `-stratified` and `-replicates` are not Super5 options. This is a modified implementation, without upstream certification.

## Implemented stages

| Source/function | GPU work | CPU behavior retained |
|---|---|---|
| `UClust::Search`, `src/uclust.cpp` | Candidate posteriors, up to original MAX_REJECTS | Word ranking, candidate acceptance in original order, EA DP and member path |
| `Super4::AlignClusters` → `MPCFlat::CalcPosteriors` | Cluster-local PPP posteriors through existing dispatch | Sparse storage, consistency, tree, refinement |
| `CalcEADistMx`, `src/eadistmx.cpp` | Consensus all-pairs posteriors when SparsePostVec is null | EA traceback and distance entries |
| `PProg::GetPostPairsAlignedFlat`, `src/getpostpairsalignedflat.cpp` | Original sampled pairs' posteriors | RNG sampling, sparse conversion, profile merge DP |

EACluster pair scoring and its original parallel early-stop decisions remain CPU. Dereplication, member insertion and duplicate restoration remain CPU. No claim that the whole Super5 algorithm runs on GPU is made. The shared Super4 entry is connected, with a byte-identical verified Super4/BB11001 smoke check; a comprehensive independent Super4 suite remains open.

The adapter `src/gpu/super5_adapter.cpp` is called on the coordinator thread. It resolves global labels to immutable ungapped sequence bytes, preserves pair orientation, and submits at most 64 pairs with approximately 64 MiB of estimated pair workspace per chunk. A single larger pair is submitted alone and subject to the existing backend budget. The backend context/workspace is reused between sequential stages, while the actual current HMM snapshot and sequences are uploaded for every batch. No probability cache crosses a model change or a consensus sequence change.

Posteriors are borrowed during the ordered consume callback only. Result buffers belong to the backend result vector and die after the chunk; callers own the sparse matrices they construct. New DP/traceback scratch uses `myfree` to match `myalloc`. Resource failures are reported and recomputed through the original CPU pair routine; device errors, invalid results and failed explicit verification stop the run. GPU budgets default to min(16 GiB, 75% of available memory), with the existing `-gpu_mem_mb` override. Total host memory also includes the original cluster/profile/sparse data; the chunk limit does not bound whole-process memory.

UClust may calculate extra candidate posteriors after the candidate that CPU would accept, but consumes decisions in the original order and does not consume randomness. PProg's diagnostic mean EA now sums in input order on the GPU adapter path; this differs from the original parallel completion-order sum. Super5's guide-tree merge does not use this returned diagnostic mean to choose its path. Random generators and the profile alignment/tie rules are unchanged.

## Verification and known limit

```bash
python3 muscle/tests/super5_backend.py --binary build/gpu/muscle_gpu \
  --input muscle/test_data/fa/BB11005 --threads 8 --perm all --out /tmp/super5-check
```

The verifier compares all expected outputs byte-for-byte, checks UClust member/path and EACluster member hashes, requires actual CUDA commit records and rejects resource fallback. `-gpu_verify` independently recomputes each accelerated pair's Post on CPU and requires bitwise equality. Optional `--require-stage super5.consensus` / `--require-stage super5.pprog` and `--require-ppp` make coverage expectations explicit. Verification overhead is excluded from speed measurements.

On RTX A5000, BB11001 (1 thread, perm none), BB11005 (1 thread, perm none), and BB11005 (8 threads, perm all) passed bitwise posterior verification, membership/path hashes and byte-identical final output. The last case checked four output files. Local CPU-only CTest and target CUDA-build CPU/static-labelled CTest both passed9/9. The default Super5 CUDA frontend, controlled 1MiB-budget CPU fallback, and Super4/BB11001 smoke check passed. A Super5/BB11001 CUDA memcheck reported0 errors.

**The original multi-threaded EACluster is not reproducible on every input.** Its shared `Done` flag is read outside synchronization and candidate decisions depend on completion order in `EACluster::GetBestCentroid`. For the first 60 bundled RdRp sequences, two CPU8 diagnostic runs produced different EACluster hashes and different final outputs, while UClust hashes agreed. GPU per-pair bitwise verification passed on this case, including all four accelerated stages. A CPU8/GPU8 byte mismatch was therefore retained as a failed end-to-end equality case, not described as passing or used as a validated speedup. This change does not repair or conceal that existing algorithmic race. Use a single-thread CPU/GPU comparison when proving equality on an affected input; comprehensive deterministic-clustering repair is separate work.

See [measured timings and reproducible inputs](BENCHMARKS.md). Long families, broad biological quality/phylogeny evaluation, racecheck and the existing R1 memory-safety audit are still outstanding.
