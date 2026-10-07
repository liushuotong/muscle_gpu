#pragma once

// CPU reference pair computation: the exact R0 operation sequence, wrapped in
// RAII so that a failed path cannot leak the myalloc() buffers.
//
// This is deliberately *not* a re-implementation: it calls the frozen R0
// routines CalcFwdFlat, CalcBwdFlat, CalcTotalProbFlat, CalcPostFlat and
// CalcAlnScoreFlat from the received sources.  It is used by:
//   * the CPU batch backend (T03),
//   * the per-pair CPU oracle inside -gpu_verify (T06),
//   * the pair_oracle test tool (T02).
//
// deviating from the reference would invalidate every oracle comparison, so the
// call order and the buffer sizes must stay as in calcpost.cpp/CalcPosterior.

#include <cstdint>
#include <string>
#include <vector>

namespace muscle_gpu {

struct ReferencePairOutput {
    std::vector<float> fwd;    // (lx+1)*(ly+1)*5, cell-major, only if keep_fb
    std::vector<float> bwd;    // same, only if keep_fb
    std::vector<float> q;      // lx*ly row-major, only if want_q
    std::vector<float> post;   // lx*ly row-major
    float z;                   // CalcTotalProbFlat result
    float ea;                  // CalcAlnScoreFlat(post)/min(lx,ly)

    ReferencePairOutput() : z(0), ea(0) {}
};

// Same overflow contract as CalcPosterior/CalcFwdFlat ("max ~21k").
bool PairLengthsAllowed(uint32_t lx, uint32_t ly);

// Run the reference DP for one pair.  Returns false with `err` for InvalidInput
// (lengths/overflow) and for allocation failure.  keep_fb/want_q add the debug
// outputs; they never change the numerical result.
bool RunReferencePair(const uint8_t *x, uint32_t lx, const uint8_t *y, uint32_t ly,
                      bool keep_fb, bool want_q, ReferencePairOutput &out,
                      std::string &err);

// q(i,j) = (F_M(i+1,j+1) + B_M(i+1,j+1)) - Z, matching the index walk and the
// operand order of CalcPostFlat.  `fwd`/`bwd` are cell-major five-state arrays.
void ComputeQFromFB(const float *fwd, const float *bwd, uint32_t lx, uint32_t ly,
                    float z, float *q);

} // namespace muscle_gpu
