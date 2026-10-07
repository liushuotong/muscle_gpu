#!/usr/bin/env python3
"""T06 GPU integration check.

Runs the real binary with -backend gpu under several configurations and compares
every result with the CPU reference through compare_alignments.py:
  * plain gpu
  * gpu + -gpu_verify (per-pair CPU oracle, failing pairs fall back)
  * gpu + -gpu_mem_mb 1 (forces a controlled fallback instead of OOM)
  * gpu + -stage_trace FILE (JSONL must be valid and must not touch the MSA)

Exit codes: 0 = all configurations match, 1 = mismatch, 77 = skipped (no GPU).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, log_path):
    with open(log_path, "wb") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def gpu_available(muscle):
    # `-backend gpu -h` cannot tell us; probe by running a tiny alignment and
    # inspecting the exit code plus the message.
    proc = subprocess.run([muscle, "-align", os.devnull, "-output", os.devnull,
                           "-backend", "gpu"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = proc.stdout.decode(errors="replace")
    if "CUDA" in text and ("not compiled in" in text or "no CUDA device" in text
                           or "no usable CUDA runtime" in text):
        return False, text.strip().splitlines()[-1] if text.strip() else "no CUDA"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muscle", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--reference", required=True, help="CPU reference alignment")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed-args", default="-perturb 1 -randseed 1")
    ap.add_argument("--skip-probe", action="store_true")
    args = ap.parse_args()

    if not args.skip_probe:
        ok, why = gpu_available(args.muscle)
        if not ok:
            print("SKIP: %s" % why)
            return 77

    os.makedirs(args.workdir, exist_ok=True)
    configs = [
        ("gpu", []),
        ("gpu_verify", ["-gpu_verify"]),
        ("gpu_mem1", ["-gpu_mem_mb", "1"]),
        ("gpu_trace", ["-stage_trace", os.path.join(args.workdir, "trace.jsonl")]),
    ]

    failures = 0
    for name, extra in configs:
        out = os.path.join(args.workdir, "%s.afa" % name)
        log = os.path.join(args.workdir, "%s.log" % name)
        cmd = [args.muscle, "-align", args.input, "-output", out,
               "-threads", str(args.threads), "-backend", "gpu"]
        cmd += args.seed_args.split()
        cmd += extra
        rc = run(cmd, log)
        if rc != 0:
            print("FAIL %s: exit %d (see %s)" % (name, rc, log))
            failures += 1
            continue

        cmp_cmd = [sys.executable, os.path.join(HERE, "compare_alignments.py"),
                   "--input", args.input, "--reference", args.reference,
                   "--candidate", out, "--require", "bytes"]
        cmp_proc = subprocess.run(cmp_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print("config %-12s exit=%d compare_exit=%d" % (name, rc, cmp_proc.returncode))
        if cmp_proc.returncode != 0:
            print(cmp_proc.stdout.decode(errors="replace"))
            failures += 1

    trace = os.path.join(args.workdir, "trace.jsonl")
    if not os.path.exists(trace):
        print("FAIL: -stage_trace produced no file (the trace check cannot be skipped)")
        failures += 1
    else:
        n = 0
        bad = 0
        cuda_records = 0
        with open(trace) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    rec = json.loads(line)
                except ValueError:
                    bad += 1
                    continue
                required = ("schema", "run_id", "family_id", "replicate_id",
                            "pair_index", "stage", "thread_id", "wall_ns",
                            "cpu_work_ns", "device_ms", "cells", "nnz", "bytes",
                            "backend", "fallback_reason")
                if rec.get("schema") != 1 or any(k not in rec for k in required):
                    bad += 1
                if rec.get("backend") == "cuda-pair-v1":
                    cuda_records += 1
        print("stage_trace: lines=%d invalid=%d cuda_backend_records=%d"
              % (n, bad, cuda_records))
        if bad or n == 0:
            failures += 1
        if cuda_records == 0:
            print("FAIL: the trace contains no record produced by the cuda-pair-v1 "
                  "backend (structured check, independent of the log text search)")
            failures += 1

    # A silent whole-run CPU fallback reproduces the CPU reference byte for byte,
    # so correctness of the output is not evidence that the GPU ran.  Require the
    # CUDA backend to appear in a log or trace.
    gpu_evidence = False
    for name, _ in configs:
        log = os.path.join(args.workdir, "%s.log" % name)
        if os.path.exists(log):
            with open(log, errors="replace") as f:
                if "cuda-pair-v1" in f.read():
                    gpu_evidence = True
        trace = os.path.join(args.workdir, "trace.jsonl")
        if os.path.exists(trace):
            with open(trace, errors="replace") as f:
                if "cuda-pair-v1" in f.read():
                    gpu_evidence = True
    if not gpu_evidence:
        print("FAIL: no evidence that the cuda-pair-v1 backend actually ran "
              "(output correctness alone cannot prove a GPU was used)")
        failures += 1

    if failures:
        print("gpu_integration: FAILED (%d)" % failures)
        return 1
    print("GPU_INTEGRATION_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
