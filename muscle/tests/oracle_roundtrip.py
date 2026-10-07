#!/usr/bin/env python3
"""Dump round-trip: two independent oracle runs of the same suite must produce
two dump trees that compare_dumps.py accepts (structure, gates, support sets),
and a deliberately corrupted copy must be rejected.

This is the small guard that keeps the authoritative comparator honest: a
comparator that always prints OK would fail the second half of this test.
"""

import argparse
import os
import shutil
import struct
import subprocess
import sys


def run(argv):
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout.decode(errors="replace")


def corrupt_post(case_dir):
    """Change one post value so the numeric gates must fire."""
    path = os.path.join(case_dir, "post.bin")
    if not os.path.exists(path):
        return False
    with open(path, "r+b") as f:
        first = f.read(4)
        if len(first) != 4:
            return False
        value = struct.unpack("<f", first)[0]
        f.seek(0)
        f.write(struct.pack("<f", value + 0.5))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--suite", default="tiny")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    a = os.path.join(args.workdir, "a")
    b = os.path.join(args.workdir, "b")
    bad = os.path.join(args.workdir, "bad")
    for d in (a, b, bad):
        shutil.rmtree(d, ignore_errors=True)

    for out in (a, b):
        rc, log = run([args.oracle, "--backend", "cpu", "--suite", args.suite,
                       "--dump", out])
        if rc != 0:
            print(log)
            return 1

    rc, log = run([sys.executable, args.compare, a, b])
    print(log.strip().splitlines()[-1])
    if rc != 0:
        print("oracle_roundtrip: identical dumps were rejected:\n%s" % log)
        return 1

    shutil.copytree(a, bad)
    cases = sorted(d for d in os.listdir(bad) if os.path.isdir(os.path.join(bad, d)))
    if not cases or not corrupt_post(os.path.join(bad, cases[0])):
        print("oracle_roundtrip: could not build a corrupted copy")
        return 1

    rc, log = run([sys.executable, args.compare, a, bad])
    if rc == 0:
        print("oracle_roundtrip: a corrupted dump was accepted (comparator is "
              "not effective)")
        return 1
    print("oracle_roundtrip: corrupted dump rejected as expected")
    print("ORACLE_ROUNDTRIP_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
