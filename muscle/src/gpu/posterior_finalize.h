#pragma once

// Host-side finalization shared by every backend and by the tests/oracle.
//
// The reference CPU path computes the total probability Z, then q, then applies
// *two* thresholds (a log threshold followed by a probability threshold) and
// finally expf() in the host math library:
//
//   Score = F_M(i+1,j+1) + B_M(i+1,j+1) - Z         (float, this operand order)
//   Post  = 0                     if Score < logf(0.01f)
//         = 1                     if Score >= 0.0f
//         = expf(Score)           otherwise
//
// Keeping expf on the host isolates the cross-platform math library difference;
// it does not by itself make the GPU F/B bitwise identical, which is why the
// compatibility gates are tolerance-based and the support set must match exactly.

#include <cstdint>
#include <string>
#include <vector>

#include "gpu/pair_backend.h"

namespace muscle_gpu {

// q -> Post exactly as calcposteriorflat.cpp::CalcPostFlat does.
void FinalizePostFromQ(const float *q, uint32_t lx, uint32_t ly,
                       float min_sparse_score, std::vector<float> &post);

// Reference expected-accuracy score: CalcAlnScoreFlat(Post)/min(lx,ly).
// Uses the real MUSCLE routine (zero gap penalty DP, Best3 tie order B>X>Y).
float ComputeEaFromPost(const float *post, uint32_t lx, uint32_t ly);

// Successive-threshold decision for a single q value, for threshold-suite tests.
enum class PostDecision { BelowLogThreshold, One, Expf };
PostDecision ClassifyQ(float q, float min_sparse_score);

// ---------------------------------------------------------------------------
// Verification gates (RESEARCH_PLAN.md section 6); diagnostic only.
// ---------------------------------------------------------------------------

struct PairGateLimits {
    float q_abs, q_rel;         // |q_gpu - q_cpu| <= q_abs + q_rel*|q_cpu|
    float post_abs, post_rel;   // |p_gpu - p_cpu| <= post_abs + post_rel*|p_cpu|
    float ea_abs;               // |ea_gpu - ea_cpu| <= ea_abs
    bool require_support_equal; // sparse support set (p >= 0.01f) identical

    PairGateLimits()
        : q_abs(2e-4f), q_rel(2e-6f), post_abs(2e-6f), post_rel(2e-5f),
          ea_abs(2e-6f), require_support_equal(true) {}
};

struct PairGateReport {
    bool ok;
    std::string reason;
    uint32_t first_diff_index;    // SIZE_MAX when equal
    float max_q_abs_diff, max_post_abs_diff, max_post_rel_diff, ea_abs_diff;
    uint32_t q_nan_inf, post_nan_inf;   // count in candidate
    uint32_t support_only_candidate, support_only_reference;
    uint32_t support_symmetric_difference;

    PairGateReport()
        : ok(false), first_diff_index(0xffffffffu), max_q_abs_diff(0),
          max_post_abs_diff(0), max_post_rel_diff(0), ea_abs_diff(0),
          q_nan_inf(0), post_nan_inf(0), support_only_candidate(0),
          support_only_reference(0), support_symmetric_difference(0) {}
};

// Compare a candidate post/EA against the reference for the same pair.
// `q_candidate`/`q_reference` may be empty when the backend does not export q.
bool ComparePairOutputs(const float *q_candidate, const float *q_reference,
                        const float *post_candidate, const float *post_reference,
                        uint32_t lx, uint32_t ly,
                        float ea_candidate, float ea_reference,
                        float min_sparse_prob, const PairGateLimits &limits,
                        PairGateReport &report);

// Format a gate report for logs/traces.
std::string FormatGateReport(const PairGateReport &report);

} // namespace muscle_gpu
