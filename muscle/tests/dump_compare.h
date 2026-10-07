#pragma once

// Small in-process dump comparator used by pair_oracle --compare.  The
// authoritative report tool is muscle/tests/compare_dumps.py; this exists so the
// CLI can fail fast without a Python dependency.
//
// Rules:
//   * a missing case directory or file, a size mismatch, or an unexpected file
//     type is a structural error (non-zero return);
//   * float payloads are compared with the documented diagnostic gates
//     (F/B/Z/q: abs <= 2e-4 + 2e-6*|ref|; Post/sparse values:
//      abs <= 2e-6 + 2e-5*|ref|; EA: abs <= 2e-6);
//   * index payloads (sparse row_ptr / col_idx) must match exactly, which is what
//     "zero support-set difference" means in practice;
//   * meta.json is intentionally not compared (it carries the binary path/SHA).

#include <cstdint>
#include <string>

namespace muscle_gpu {

// Returns the number of failing cases; `err` is non-empty for structural errors
// and for the first numeric failure of each kind.
uint32_t CompareDumpTrees(const std::string &reference_dir,
                          const std::string &candidate_dir, std::string &err);

} // namespace muscle_gpu
