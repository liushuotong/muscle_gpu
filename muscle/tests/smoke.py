#!/usr/bin/env python3
"""T00 smoke test: exit code is not enough.

Checks performed on the produced alignment:
  * the process exits 0;
  * every input label appears exactly once in the output;
  * every input residue (gaps removed, upper-cased by the reference) appears in
    the output in the same order for that label;
  * the output is valid FASTA (no stray lines, consistent widths).

Two fresh runs are compared byte-for-byte for the same command line; this is a
repeatability observation for the smoke case only, not a claim that every mode is
reproducible.
"""

import argparse
import os
import subprocess
import sys
import hashlib


def read_fasta(path):
    labels = []
    seqs = []
    cur = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                cur = line[1:].strip()
                labels.append(cur)
                seqs.append([])
            else:
                if cur is None:
                    raise SystemExit("FASTA parse error: sequence data before header")
                seqs[-1].append(line.strip())
    joined = ["".join(chunks) for chunks in seqs]
    if len(set(labels)) != len(labels):
        raise SystemExit("FASTA parse error: duplicate labels in output")
    return labels, joined


def ungapped(seq):
    return seq.replace("-", "").replace(".", "")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muscle", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--extra", default="")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    out1 = os.path.join(args.workdir, "smoke1.afa")
    out2 = os.path.join(args.workdir, "smoke2.afa")

    cmd = [args.muscle, "-align", args.input, "-output", out1,
           "-threads", str(args.threads)]
    if args.extra:
        cmd += args.extra.split()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout.decode(errors="replace"))
        sys.stderr.write(proc.stderr.decode(errors="replace"))
        raise SystemExit("smoke: muscle exited %d" % proc.returncode)

    in_labels, in_seqs = read_fasta(args.input)
    out_labels, out_seqs = read_fasta(out1)

    if sorted(in_labels) != sorted(out_labels):
        raise SystemExit("smoke: label sets differ\nin : %s\nout: %s"
                         % (sorted(in_labels), sorted(out_labels)))

    out_map = dict(zip(out_labels, out_seqs))
    for label, seq in zip(in_labels, in_seqs):
        got = ungapped(out_map[label])
        want = ungapped(seq).upper()
        if got != want:
            raise SystemExit("smoke: residues not preserved for %s\nin : %s\nout: %s"
                             % (label, want, got))

    lengths = set(len(s) for s in out_seqs)
    if len(lengths) != 1:
        raise SystemExit("smoke: output is not a single alignment (column counts %s)"
                         % sorted(lengths))

    # Fresh process, same command line: compare bytes.
    cmd2 = [args.muscle, "-align", args.input, "-output", out2,
            "-threads", str(args.threads)]
    if args.extra:
        cmd2 += args.extra.split()
    proc2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc2.returncode != 0:
        raise SystemExit("smoke: second run exited %d" % proc2.returncode)

    h1, h2 = sha256_file(out1), sha256_file(out2)
    print("smoke: labels=%d columns=%d sha256_1=%s sha256_2=%s"
          % (len(out_labels), lengths.pop(), h1, h2))
    if h1 != h2:
        raise SystemExit("smoke: two fresh runs produced different bytes")
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
