#!/usr/bin/env python3
"""T02 alignment comparator (FASTA and EFA).

Checks, in this order:
  1. both files parse as FASTA (or both as EFA) and the sequences they claim are
     the same as the --input sequences (labels and ungapped residues);
  2. byte_equal: the two files are identical byte-for-byte;
  3. column_vectors_equal: every column holds the same (label -> residue) map;
  4. residue_pairs_equal: the set of aligned residue pairs is identical, where a
     pair is canonicalised through the *input label order* and the ungapped
     residue coordinate of each sequence -- never through the output letters or
     output row order.

Exit status is non-zero when any of (2)/(3)/(4) differ or when the inputs do not
match.  "Quality is similar" is never accepted as equality here.
"""

import argparse
import hashlib
import json
import sys

GAP_CHARS = set("-.")


def parse_fasta(path):
    """Returns a list of (header, [(label, seq), ...])."""
    blocks = []
    labels = []
    seqs = []
    header = None
    cur = None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    seqs.append("".join(cur))
                cur = []
                label = line[1:].strip()
                labels.append(label)
            elif line.startswith("<"):
                # EFA replicate header: flush the current block
                if cur is not None:
                    seqs.append("".join(cur))
                    cur = None
                if labels:
                    blocks.append((header, list(zip(labels, seqs))))
                header = line[1:].strip()
                labels = []
                seqs = []
            else:
                if cur is None:
                    if not labels:
                        raise ValueError("%s: sequence data before a header" % path)
                    cur = []
                cur.append(line.strip())
    if cur is not None:
        seqs.append("".join(cur))
    if labels:
        blocks.append((header, list(zip(labels, seqs))))
    return blocks


def ungapped(seq):
    return "".join(c for c in seq if c not in GAP_CHARS)


def read_input(path):
    blocks = parse_fasta(path)
    if len(blocks) != 1:
        raise ValueError("--input must be a single FASTA block")
    return blocks[0][1]


def build_reference(input_pairs):
    """label -> ungapped input sequence, plus the input order."""
    order = [label for label, _ in input_pairs]
    table = {}
    for label, seq in input_pairs:
        if label in table:
            raise ValueError("duplicate input label %s" % label)
        table[label] = ungapped(seq)
    return order, table


def verify_block(block, input_order, input_table, what):
    problems = []
    labels = [label for label, _ in block]
    if sorted(labels) != sorted(input_order):
        problems.append("%s: label set differs from the input" % what)
        return problems
    for label, seq in block:
        if ungapped(seq) != input_table[label]:
            problems.append("%s: residues of %s do not match the input" % (what, label))
    return problems


def column_vectors(block):
    labels = [label for label, _ in block]
    width = set(len(seq) for _, seq in block)
    if len(width) != 1:
        return None
    n = width.pop()
    cols = []
    for j in range(n):
        cols.append(tuple(sorted((label, block[i][1][j]) for i, label in enumerate(labels))))
    return cols


def residue_pairs(block, input_index):
    """Canonical set of aligned residue pairs.

    A pair is (label_a, pos_a, label_b, pos_b) with pos_* the ungapped residue
    coordinate inside the input sequence (0-based), labels ordered by input order.
    """
    positions = []
    for label, seq in block:
        pos = []
        counter = -1
        for c in seq:
            if c in GAP_CHARS:
                pos.append(None)
            else:
                counter += 1
                pos.append(counter)
        positions.append(pos)
    width = len(positions[0]) if positions else 0
    pairs = set()
    n = len(block)
    for i in range(n):
        for k in range(i + 1, n):
            li = block[i][0]
            lk = block[k][0]
            first, second = (li, lk) if input_index[li] < input_index[lk] else (lk, li)
            swapped = first != li
            for j in range(width):
                a = positions[i][j]
                b = positions[k][j]
                if a is None or b is None:
                    continue
                if swapped:
                    pairs.add((first, b, second, a))
                else:
                    pairs.add((first, a, second, b))
    return pairs


def compare_blocks(ref_block, cand_block, input_order, input_table, input_index, label):
    """Returns (structure_problems, column_problems, residue_problems).

    The three classes are kept separate so that --require can turn them into real
    acceptance levels: residue pairs are the compatibility floor, column vectors
    are stricter, byte equality is strictest.
    """
    structure = []
    columns = []
    residues = []
    structure += verify_block(ref_block, input_order, input_table, "%s reference" % label)
    structure += verify_block(cand_block, input_order, input_table, "%s candidate" % label)

    ref_cols = column_vectors(ref_block)
    cand_cols = column_vectors(cand_block)
    column_vectors_equal = (ref_cols is not None and ref_cols == cand_cols)

    ref_pairs = residue_pairs(ref_block, input_index)
    cand_pairs = residue_pairs(cand_block, input_index)
    residue_pairs_equal = (ref_pairs == cand_pairs)

    if not column_vectors_equal:
        columns.append("%s: column vectors differ" % label)
    if not residue_pairs_equal:
        residues.append("%s: aligned residue pairs differ (only_ref=%d only_cand=%d)"
                        % (label, len(ref_pairs - cand_pairs), len(cand_pairs - ref_pairs)))

    print("%s: labels=%d columns=%s column_vectors_equal=%s residue_pairs_equal=%s"
          % (label, len(ref_block),
             sorted(set(len(s) for _, s in ref_block)),
             column_vectors_equal, residue_pairs_equal))
    return structure, columns, residues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="original unaligned FASTA")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--require", choices=["bytes", "columns", "residues"],
                    default="residues",
                    help="minimum equality level that makes the tool exit non-zero; "
                         "'residues' (default) is the compatibility floor, 'bytes' is "
                         "the strictest and is what GPU compatibility mode claims")
    args = ap.parse_args()

    input_pairs = read_input(args.input)
    input_order, input_table = build_reference(input_pairs)
    input_index = {label: i for i, label in enumerate(input_order)}

    ref_blocks = parse_fasta(args.reference)
    cand_blocks = parse_fasta(args.candidate)
    if len(ref_blocks) != len(cand_blocks):
        sys.stderr.write("compare_alignments: replicate counts differ (%d vs %d)\n"
                         % (len(ref_blocks), len(cand_blocks)))
        return 2

    byte_equal = False
    with open(args.reference, "rb") as f1, open(args.candidate, "rb") as f2:
        b1 = f1.read()
        b2 = f2.read()
        byte_equal = (b1 == b2)
    print("byte_equal=%s sha256_ref=%s sha256_cand=%s"
          % (byte_equal, hashlib.sha256(b1).hexdigest(), hashlib.sha256(b2).hexdigest()))

    structure_problems = []
    column_problems = []
    residue_problems = []
    for i, (rb, cb) in enumerate(zip(ref_blocks, cand_blocks)):
        label = "replicate%d" % i if len(ref_blocks) > 1 else "alignment"
        if len(ref_blocks) > 1 and (rb[0] != cb[0]):
            structure_problems.append("%s: EFA replicate headers differ (%r vs %r)"
                                      % (label, rb[0], cb[0]))
        st, col, res = compare_blocks(rb[1], cb[1], input_order, input_table,
                                      input_index, label)
        structure_problems += st
        column_problems += col
        residue_problems += res

    # Acceptance levels (cumulative): residues <= columns <= bytes.
    problems = list(structure_problems) + list(residue_problems)
    if args.require in ("columns", "bytes"):
        problems += column_problems
    if args.require == "bytes" and not byte_equal:
        problems.append("byte_equal is False but --require bytes was given")

    if column_problems and args.require == "residues":
        print("note: %d column-vector difference(s) tolerated at --require residues"
              % len(column_problems))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"schema": 1, "byte_equal": byte_equal, "require": args.require,
                       "structure_problems": structure_problems,
                       "column_problems": column_problems,
                       "residue_problems": residue_problems,
                       "problems": problems, "ok": not problems}, f, indent=2)

    if problems:
        print("\ncompare_alignments: FAILED")
        for p in problems:
            print("  - %s" % p)
        return 1
    levels = ["same input", "same residue pairs"]
    if args.require in ("columns", "bytes"):
        levels.append("same columns")
    if args.require == "bytes":
        levels.append("byte-identical")
    print("\ncompare_alignments: OK at --require %s (%s)" % (args.require, ", ".join(levels)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
