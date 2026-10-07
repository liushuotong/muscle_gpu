#!/usr/bin/env python3
"""CLI-level backend contract checks (small, deterministic, no GPU needed).

1. `-cpu_batch` must reproduce the reference output byte for byte.
2. `-backend cpu` and `-backend auto` must reproduce the reference output.
3. An explicit `-backend gpu` without CUDA must exit non-zero with a clear reason
   (and must not be silently downgraded when -cpu_batch is also passed).
"""

import argparse
import hashlib
import os
import subprocess
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(argv):
    # Bound smoke-test CPU work on shared servers, independent of host size.
    if "-threads" not in argv:
        argv = argv + ["-threads", "2"]
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout.decode(errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muscle", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--expect-cuda", action="store_true",
                    help="the binary was built with MUSCLE_CUDA=ON")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    failures = []

    def out(name):
        return os.path.join(args.workdir, name)

    rc, log = run([args.muscle, "-align", args.input, "-output", out("reference.afa"),
                   "-quiet"])
    if rc != 0:
        print(log)
        return 1
    ref_sha = sha256(out("reference.afa"))

    for name, extra in (("cpu", ["-backend", "cpu"]),
                        ("auto", ["-backend", "auto"]),
                        ("batch", ["-cpu_batch"])):
        rc, log = run([args.muscle, "-align", args.input, "-output", out("%s.afa" % name),
                       "-quiet"] + extra)
        if rc != 0:
            failures.append("%s: exit %d\n%s" % (name, rc, log))
            continue
        got = sha256(out("%s.afa" % name))
        if got != ref_sha:
            failures.append("%s: output differs from the reference (%s != %s)"
                            % (name, got, ref_sha))
        else:
            print("%s: byte-identical to the reference" % name)

    # Explicit GPU request: either it works (CUDA build + device) or it must fail
    # loudly.  Combining it with -cpu_batch must never silently run the CPU path.
    rc, log = run([args.muscle, "-align", args.input, "-output", out("gpu.afa"),
                   "-quiet", "-backend", "gpu"])
    if args.expect_cuda:
        print("gpu: exit %d (CUDA build; device test is a separate gpu-labelled test)" % rc)
    else:
        if rc == 0:
            failures.append("-backend gpu exited 0 although CUDA is not compiled in")
        elif "CUDA" not in log:
            failures.append("-backend gpu failed without explaining the CUDA problem:\n" + log)
        else:
            print("gpu: refused with a clear reason (exit %d)" % rc)

    rc, log = run([args.muscle, "-align", args.input, "-output", out("gpu_batch.afa"),
                   "-quiet", "-backend", "gpu", "-cpu_batch"])
    if rc == 0:
        failures.append("-backend gpu -cpu_batch silently succeeded (the GPU request "
                        "was dropped instead of failing)")
    else:
        print("gpu+cpu_batch: refused (exit %d)" % rc)

    if failures:
        print("\ncli_backend: FAILED")
        for f in failures:
            print("  - %s" % f)
        return 1
    print("\nCLI_BACKEND_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
