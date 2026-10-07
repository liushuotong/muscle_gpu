#include "gpu/posterior_finalize.h"

#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <memory>

#include "muscle.h"

namespace muscle_gpu {

namespace {

struct MyfreeDeleter {
    void operator()(void *p) const { myfree(p); }
};

using FloatBuf = std::unique_ptr<float, MyfreeDeleter>;

inline bool IsFinite(float x) { return std::isfinite(x); }

} // namespace

PostDecision ClassifyQ(float q, float min_sparse_score) {
    if (q < min_sparse_score)
        return PostDecision::BelowLogThreshold;
    if (q >= LOG_ONE)
        return PostDecision::One;
    return PostDecision::Expf;
}

void FinalizePostFromQ(const float *q, uint32_t lx, uint32_t ly,
                       float min_sparse_score, std::vector<float> &post) {
    const size_t cells = size_t(lx) * size_t(ly);
    post.resize(cells);
    for (size_t k = 0; k < cells; ++k) {
        const float score = q[k];
        if (score < min_sparse_score)
            post[k] = 0.0f;
        else if (score >= LOG_ONE)
            post[k] = 1.0f;
        else
            post[k] = expf(score);
    }
}

float ComputeEaFromPost(const float *post, uint32_t lx, uint32_t ly) {
    // Same allocation and call as MPCFlat::CalcPosterior.
    float *raw = AllocDPRows(lx, ly);
    FloatBuf dprows(raw);
    float score = CalcAlnScoreFlat(post, lx, ly, raw);
    const uint32_t minlen = (lx < ly) ? lx : ly;
    return score / float(minlen);
}

bool ComparePairOutputs(const float *q_candidate, const float *q_reference,
                        const float *post_candidate, const float *post_reference,
                        uint32_t lx, uint32_t ly,
                        float ea_candidate, float ea_reference,
                        float min_sparse_prob, const PairGateLimits &limits,
                        PairGateReport &report) {
    report = PairGateReport();
    const size_t cells = size_t(lx) * size_t(ly);

    if (!IsFinite(ea_candidate) || !IsFinite(ea_reference)) {
        report.reason = "EA is NaN/Inf";
        return false;
    }
    report.ea_abs_diff = std::fabs(ea_candidate - ea_reference);
    if (report.ea_abs_diff > limits.ea_abs) {
        char buf[256];
        snprintf(buf, sizeof(buf), "EA diff %.9g > %.9g (cand %.9g ref %.9g)",
                 report.ea_abs_diff, limits.ea_abs, ea_candidate, ea_reference);
        report.reason = buf;
        return false;
    }

    if (q_candidate != nullptr && q_reference != nullptr) {
        for (size_t k = 0; k < cells; ++k) {
            const float a = q_candidate[k];
            const float b = q_reference[k];
            if (!IsFinite(a)) {
                ++report.q_nan_inf;
                if (report.reason.empty()) {
                    char buf[256];
                    snprintf(buf, sizeof(buf), "q candidate NaN/Inf at cell %zu (ref %.9g)",
                             k, b);
                    report.reason = buf;
                }
                continue;
            }
            const float diff = std::fabs(a - b);
            if (diff > report.max_q_abs_diff)
                report.max_q_abs_diff = diff;
            const float allowed = limits.q_abs + limits.q_rel * std::fabs(b);
            if (diff > allowed && report.reason.empty()) {
                char buf[256];
                snprintf(buf, sizeof(buf),
                         "q diff %.9g > %.9g at cell %zu (cand %.9g ref %.9g)", diff,
                         allowed, k, a, b);
                report.reason = buf;
                report.first_diff_index = uint32_t(k);
            }
        }
    }

    bool support_mismatch = false;
    for (size_t k = 0; k < cells; ++k) {
        const float a = post_candidate[k];
        const float b = post_reference[k];
        if (!IsFinite(a)) {
            ++report.post_nan_inf;
            if (report.reason.empty()) {
                char buf[256];
                snprintf(buf, sizeof(buf), "post candidate NaN/Inf at cell %zu", k);
                report.reason = buf;
            }
            continue;
        }
        const float diff = std::fabs(a - b);
        if (diff > report.max_post_abs_diff)
            report.max_post_abs_diff = diff;
        if (b != 0.0f) {
            const float rel = diff / std::fabs(b);
            if (rel > report.max_post_rel_diff)
                report.max_post_rel_diff = rel;
        }
        const float allowed = limits.post_abs + limits.post_rel * std::fabs(b);
        if (diff > allowed && report.reason.empty()) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "post diff %.9g > %.9g at cell %zu (cand %.9g ref %.9g)", diff,
                     allowed, k, a, b);
            report.reason = buf;
            report.first_diff_index = uint32_t(k);
        }

        const bool sa = (a >= min_sparse_prob);
        const bool sb = (b >= min_sparse_prob);
        if (sa != sb) {
            support_mismatch = true;
            if (sa)
                ++report.support_only_candidate;
            else
                ++report.support_only_reference;
        }
    }
    report.support_symmetric_difference =
        report.support_only_candidate + report.support_only_reference;

    if (report.q_nan_inf != 0 || report.post_nan_inf != 0) {
        if (report.reason.empty())
            report.reason = "NaN/Inf in candidate output";
        return false;
    }
    if (limits.require_support_equal && support_mismatch) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "sparse support differs: only_cand %u only_ref %u (threshold %.9g)",
                 report.support_only_candidate, report.support_only_reference,
                 double(min_sparse_prob));
        if (report.reason.empty())
            report.reason = buf;
        report.ok = false;
        return false;
    }
    if (!report.reason.empty()) {
        report.ok = false;
        return false;
    }
    report.ok = true;
    report.reason = "ok";
    return true;
}

std::string FormatGateReport(const PairGateReport &report) {
    char buf[512];
    snprintf(buf, sizeof(buf),
             "ok=%d max_q_abs=%.9g max_post_abs=%.9g max_post_rel=%.9g ea_abs=%.9g "
             "nan_q=%u nan_post=%u support_diff=%u (%s)",
             report.ok ? 1 : 0, report.max_q_abs_diff, report.max_post_abs_diff,
             report.max_post_rel_diff, report.ea_abs_diff, report.q_nan_inf,
             report.post_nan_inf, report.support_symmetric_difference,
             report.reason.c_str());
    return std::string(buf);
}

} // namespace muscle_gpu
