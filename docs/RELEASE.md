# Source release preparation

Run `python3 tools/prepare_github.py --name muscle_gpu-super5` from the workspace to generate `dist/muscle_gpu-super5/` and `dist/muscle_gpu-super5.tar.gz`. The exporter copies the complete supplied MUSCLE source and bundled examples, the public documentation, license, CI and raw timing TSVs. It omits private operational records, machine-specific connection scripts, build/toolchain directories and article PDFs. The unmodified received-source manifest is retained for provenance; differences in this modified release are expected.

Upload the contents of the exported directory to your own GitHub repository. Keep README, LICENSE, NOTICE, source and build scripts together. When publishing binaries, accompany them with the corresponding source release. Do not publish the entire working directory through a drag-and-drop upload; `.gitignore` only affects Git and does not filter web uploads.

The public tree includes `docs/release_tree.json`, which lists SHA-256 for each exported file except the manifest itself. Check the source package on a fresh checkout using the CPU build and CTest commands in README. GPU correctness checks require a CUDA device; CPU CI does not claim GPU coverage.

The historical benchmark is CPU8/GPU8 on BB11005 diversified 10 replicates: medians 7.43 s and 5.01 s, 1.48x, three repetitions on a shared server. It was measured before the new frontend. Do not reuse the historical binary SHA as the identity of the new `muscle_gpu` executable or claim a new performance benchmark merely because frontend smoke validation passes.

The Super5 release is a subsequent build with its own binary SHA and new current-executable timings in BENCHMARKS.md. Its hybrid posterior stages are verified on bounded examples; the retained original CPU EACluster race prevents a universal multi-thread byte-equality guarantee. Do not describe the release as a fully GPU-resident Super5 or an officially certified MUSCLE build.

The initial GitHub tree is prepared with `python3 tools/prepare_github.py --name muscle_gpu-publish --omit-large-example`. The optional27MiB RdRp structure archive is omitted; this does not remove any source, build input, amino-acid benchmark input or CI fixture. The earlier complete local package remains separate. The exporter works from a public checkout using docs/provenance when private planning files are absent.
