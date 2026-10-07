#!/usr/bin/env python3
"""T02 dump comparator (authoritative report tool).

Compares two pair_oracle dump trees and reports, per case and per file:
  * structural problems (missing case/file, size, dtype, layout, endianness)
  * first differing index, max absolute / relative / ULP difference
  * NaN/Inf counts
  * counts of values near the two decision thresholds
  * sparse support-set symmetric difference

Exit status is non-zero for a structural mismatch, any NaN/Inf, any support-set
difference, or (unless --no-gates) any violation of the documented diagnostic
gates:
    F/B/Z/q      abs <= 2e-4 + 2e-6*|ref|
    Post/values  abs <= 2e-6 + 2e-5*|ref|
    EA           abs <= 2e-6
These gates are diagnostic: they never permit a changed discrete decision.
"""

import argparse
import json
import math
import os
import struct
import sys

FLOAT_FILES = {
    "fwd.bin": "fb",
    "bwd.bin": "fb",
    "z.bin": "fb",
    "q.bin": "fb",
    "post.bin": "post",
    "sparse_values.bin": "post",
    "ea.bin": "ea",
}
INT_FILES = ["sparse_row_ptr.bin", "sparse_col_idx.bin"]

GATES = {
    "fb": lambda ref: 2e-4 + 2e-6 * abs(ref),
    "post": lambda ref: 2e-6 + 2e-5 * abs(ref),
    "ea": lambda ref: 2e-6,
}

LOG_THRESHOLD = math.log(0.01)
PROB_THRESHOLD = 0.01


def read_floats(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0:
        raise ValueError("%s: size %d is not a multiple of 4" % (path, len(data)))
    return list(struct.unpack("<%df" % (len(data) // 4), data)), data


def read_u32(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0:
        raise ValueError("%s: size %d is not a multiple of 4" % (path, len(data)))
    return list(struct.unpack("<%dI" % (len(data) // 4), data))


def ulp_distance(a, b):
    if math.isnan(a) or math.isnan(b):
        return None
    ia = struct.unpack("<i", struct.pack("<f", a))[0]
    ib = struct.unpack("<i", struct.pack("<f", b))[0]
    if ia < 0:
        ia = -(ia + 1) if ia != -2147483648 else 2147483647
    if ib < 0:
        ib = -(ib + 1) if ib != -2147483648 else 2147483647
    return abs(ia - ib)


def compare_case(ref_dir, cand_dir, name, use_gates):
    problems = []
    report = {"case": name, "files": {}}

    ref_meta_path = os.path.join(ref_dir, "meta.json")
    cand_meta_path = os.path.join(cand_dir, "meta.json")
    if not os.path.exists(ref_meta_path) or not os.path.exists(cand_meta_path):
        problems.append("case %s: meta.json missing" % name)
        return problems, report

    with open(ref_meta_path) as f:
        ref_meta = json.load(f)
    with open(cand_meta_path) as f:
        cand_meta = json.load(f)

    for key in ("schema", "dtype", "endianness"):
        if ref_meta.get(key) != cand_meta.get(key):
            problems.append("case %s: meta.%s differs (%r vs %r)"
                            % (name, key, ref_meta.get(key), cand_meta.get(key)))
    if ref_meta.get("layout") != cand_meta.get("layout"):
        problems.append("case %s: layout metadata differs" % name)
    if (ref_meta.get("Lx"), ref_meta.get("Ly")) != (cand_meta.get("Lx"), cand_meta.get("Ly")):
        problems.append("case %s: Lx/Ly differ" % name)
    if ref_meta.get("alphabet") != cand_meta.get("alphabet"):
        problems.append("case %s: alphabet differs" % name)
    # Identity metadata must match: a dump produced with a different HMM model or
    # semantics version is not comparable even if every file parses.
    for key in ("hmm_digest_sha256", "source_id", "semantics_version", "alphabet",
                "synthetic_q"):
        if ref_meta.get(key) != cand_meta.get(key):
            problems.append("case %s: meta.%s differs (%r vs %r)"
                            % (name, key, ref_meta.get(key), cand_meta.get(key)))
    if ref_meta.get("nnz") != cand_meta.get("nnz"):
        problems.append("case %s: meta.nnz differs (%r vs %r)"
                        % (name, ref_meta.get("nnz"), cand_meta.get("nnz")))
    report["ref_meta"] = {k: ref_meta.get(k) for k in
                          ("Lx", "Ly", "z", "nnz", "hmm_digest_sha256", "source_id",
                           "semantics_version")}
    report["cand_meta"] = {k: cand_meta.get(k) for k in
                           ("Lx", "Ly", "z", "nnz", "hmm_digest_sha256", "source_id",
                            "semantics_version")}

    ref_files = sorted(f["name"] for f in ref_meta.get("files", []))
    cand_files = sorted(f["name"] for f in cand_meta.get("files", []))
    if ref_files != cand_files:
        problems.append("case %s: file lists differ (%s vs %s)" % (name, ref_files, cand_files))

    for fname, kind in sorted(FLOAT_FILES.items()):
        rp = os.path.join(ref_dir, fname)
        cp = os.path.join(cand_dir, fname)
        r_exists = os.path.exists(rp)
        c_exists = os.path.exists(cp)
        if not r_exists and not c_exists:
            if fname in ref_files or fname in cand_files:
                problems.append("case %s: %s is listed in meta.json but missing on "
                                "both sides" % (name, fname))
            continue
        if not os.path.exists(rp) or not os.path.exists(cp):
            problems.append("case %s: %s present on only one side" % (name, fname))
            continue
        if os.path.getsize(rp) != os.path.getsize(cp):
            problems.append("case %s: %s size differs (%d vs %d)"
                            % (name, fname, os.path.getsize(rp), os.path.getsize(cp)))
            continue
        ref, ref_bytes = read_floats(rp)
        cand, _ = read_floats(cp)
        stats = {
            "count": len(ref),
            "first_diff_index": None,
            "max_abs": 0.0,
            "max_rel": 0.0,
            "max_ulp": 0,
            "nan_inf_ref": 0,
            "nan_inf_cand": 0,
            "gate_violations": 0,
            "first_gate_violation": None,
            "near_log_threshold": 0,
            "near_prob_threshold": 0,
            "near_zero": 0,
        }
        for i, (a, b) in enumerate(zip(cand, ref)):
            if not math.isfinite(a):
                stats["nan_inf_cand"] += 1
                continue
            if not math.isfinite(b):
                stats["nan_inf_ref"] += 1
                continue
            diff = abs(a - b)
            if diff != 0.0 and stats["first_diff_index"] is None:
                stats["first_diff_index"] = i
            stats["max_abs"] = max(stats["max_abs"], diff)
            if b != 0.0:
                stats["max_rel"] = max(stats["max_rel"], diff / abs(b))
            u = ulp_distance(a, b)
            if u is not None:
                stats["max_ulp"] = max(stats["max_ulp"], u)
            if use_gates and diff > GATES[kind](b):
                stats["gate_violations"] += 1
                if stats["first_gate_violation"] is None:
                    stats["first_gate_violation"] = {
                        "index": i, "cand": a, "ref": b, "diff": diff,
                        "allowed": GATES[kind](b)}
            if kind == "post" and abs(a - PROB_THRESHOLD) < 1e-6:
                stats["near_prob_threshold"] += 1
            if kind == "fb" and fname == "q.bin":
                if abs(a - LOG_THRESHOLD) < 1e-5:
                    stats["near_log_threshold"] += 1
                if abs(a) < 1e-6:
                    stats["near_zero"] += 1
        report["files"][fname] = stats
        if stats["nan_inf_cand"] or stats["nan_inf_ref"]:
            problems.append("case %s: %s contains NaN/Inf (cand %d, ref %d)"
                            % (name, fname, stats["nan_inf_cand"], stats["nan_inf_ref"]))
        if use_gates and stats["gate_violations"]:
            v = stats["first_gate_violation"]
            problems.append(
                "case %s: %s gate violation at %d: cand=%r ref=%r diff=%r allowed=%r"
                % (name, fname, v["index"], v["cand"], v["ref"], v["diff"], v["allowed"]))

    # Support sets: exact index equality is the compatibility requirement.
    ref_row = os.path.join(ref_dir, "sparse_row_ptr.bin")
    cand_row = os.path.join(cand_dir, "sparse_row_ptr.bin")
    declared_sparse = any(f.startswith("sparse_") for f in ref_files + cand_files)
    if declared_sparse and not (os.path.exists(ref_row) and os.path.exists(cand_row)):
        problems.append("case %s: sparse support files are declared but missing on "
                        "one side (the discrete-decision check cannot be skipped)"
                        % name)
    elif os.path.exists(ref_row) and os.path.exists(cand_row):
        r_row = read_u32(ref_row)
        c_row = read_u32(cand_row)
        for side, d in (("reference", ref_dir), ("candidate", cand_dir)):
            for companion in ("sparse_col_idx.bin", "sparse_values.bin"):
                path = os.path.join(d, companion)
                if not os.path.exists(path):
                    problems.append("case %s: %s is missing %s" % (name, side, companion))
                    return problems, report
        r_col = read_u32(os.path.join(ref_dir, "sparse_col_idx.bin"))
        c_col = read_u32(os.path.join(cand_dir, "sparse_col_idx.bin"))
        if r_row != c_row:
            problems.append("case %s: sparse row_ptr differs" % name)
        support = {"ref_nnz": len(r_col), "cand_nnz": len(c_col)}
        if r_col != c_col:
            # symmetric difference over (row, col) pairs
            def pairs(row_ptr, cols):
                out = set()
                for i in range(len(row_ptr) - 1):
                    for k in range(row_ptr[i], row_ptr[i + 1]):
                        out.add((i, cols[k]))
                return out
            rs = pairs(r_row, r_col)
            cs = pairs(c_row, c_col)
            support["only_ref"] = len(rs - cs)
            support["only_cand"] = len(cs - rs)
            support["symmetric_difference"] = len(rs ^ cs)
            problems.append("case %s: sparse support differs (%d only-ref, %d only-cand)"
                            % (name, support["only_ref"], support["only_cand"]))
        else:
            support["symmetric_difference"] = 0
        report["support"] = support

    return problems, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--json", default=None, help="write the full report as JSON")
    ap.add_argument("--no-gates", action="store_true",
                    help="report differences without applying the numeric gates")
    ap.add_argument("--case", default=None, help="compare only this case")
    args = ap.parse_args()

    if not os.path.isdir(args.reference) or not os.path.isdir(args.candidate):
        sys.stderr.write("compare_dumps: both arguments must be dump directories\n")
        return 2

    ref_cases = sorted(d for d in os.listdir(args.reference)
                       if os.path.isdir(os.path.join(args.reference, d)))
    cand_cases = sorted(d for d in os.listdir(args.candidate)
                        if os.path.isdir(os.path.join(args.candidate, d)))
    if args.case:
        ref_cases = [c for c in ref_cases if c == args.case]
        cand_cases = [c for c in cand_cases if c == args.case]
    if not ref_cases:
        sys.stderr.write("compare_dumps: no cases in the reference dump\n")
        return 2
    if ref_cases != cand_cases:
        sys.stderr.write("compare_dumps: case sets differ\n  ref : %s\n  cand: %s\n"
                         % (ref_cases, cand_cases))
        return 2

    all_problems = []
    report = {"schema": 1, "reference": args.reference, "candidate": args.candidate,
              "cases": {}}
    for name in ref_cases:
        problems, case_report = compare_case(os.path.join(args.reference, name),
                                             os.path.join(args.candidate, name), name,
                                             not args.no_gates)
        report["cases"][name] = case_report
        all_problems.extend(problems)

    report["problems"] = all_problems
    report["ok"] = not all_problems

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)

    for name in ref_cases:
        cr = report["cases"][name]
        bits = []
        for fname, stats in sorted(cr.get("files", {}).items()):
            bits.append("%s:max_abs=%.3g,max_ulp=%d" % (fname, stats["max_abs"], stats["max_ulp"]))
        sup = cr.get("support", {})
        sup_txt = ("support_diff=%d" % sup["symmetric_difference"]) if "symmetric_difference" in sup else "support=n/a"
        print("case %-24s %s %s" % (name, sup_txt, " ".join(bits)))

    if all_problems:
        print("\ncompare_dumps: FAILED")
        for p in all_problems:
            print("  - %s" % p)
        return 1
    print("\ncompare_dumps: OK (structure, gates and support sets match)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
