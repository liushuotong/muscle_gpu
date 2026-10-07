#include "gpu/pair_reference.h"

#include <climits>
#include <cmath>
#include <memory>

#include "muscle.h"
#include "gpu/posterior_finalize.h"

namespace muscle_gpu {

namespace {

struct MyfreeDeleter {
    void operator()(void *p) const { myfree(p); }
};
using FloatBuf = std::unique_ptr<float, MyfreeDeleter>;

} // namespace

bool PairLengthsAllowed(uint32_t lx, uint32_t ly) {
    // Verbatim contract from CalcFwdFlat/CalcBwdFlat/CalcPosterior.  The estimate
    // is approximate (borders excluded), so this is a compatibility gate and not
    // a substitute for the real allocation checks below.
    if (double(lx) * double(ly) * 5 + 100 > double(INT_MAX))
        return false;
    return true;
}

void ComputeQFromFB(const float *fwd, const float *bwd, uint32_t lx, uint32_t ly,
                    float z, float *q) {
    // Mirror of the CalcPostFlat walk: start at base of cell (1,1), advance one
    // cell per j, skip one cell at the end of each row.
    size_t ix = size_t(HMMSTATE_COUNT) * (size_t(ly) + 2); // (1*(LY+1) + 1)
    size_t out = 0;
    for (uint32_t i = 0; i < lx; ++i) {
        for (uint32_t j = 0; j < ly; ++j) {
            const float score = fwd[ix] + bwd[ix] - z;
            q[out++] = score;
            ix += HMMSTATE_COUNT;
        }
        ix += HMMSTATE_COUNT;
    }
}

bool RunReferencePair(const uint8_t *x, uint32_t lx, const uint8_t *y, uint32_t ly,
                      bool keep_fb, bool want_q, ReferencePairOutput &out,
                      std::string &err) {
    err.clear();
    if (x == nullptr || y == nullptr || lx == 0 || ly == 0) {
        err = "empty sequence";
        return false;
    }
    if (!PairLengthsAllowed(lx, ly)) {
        char buf[160];
        snprintf(buf, sizeof(buf),
                 "HMM overflow, sequence lengths %u, %u (max ~21k)", lx, ly);
        err = buf;
        return false;
    }

    // R0 AllocFB over-allocates by sizeof(float) (RISK-01).  We keep the exact R0
    // behaviour here: the reference identity must not change before R1 exists.
    FloatBuf fwd(AllocFB(lx, ly));
    FloatBuf bwd(AllocFB(lx, ly));
    if (fwd.get() == nullptr || bwd.get() == nullptr) {
        err = "myalloc failed for Forward/Backward buffers";
        return false;
    }

    CalcFwdFlat(x, lx, y, ly, fwd.get());
    CalcBwdFlat(x, lx, y, ly, bwd.get());

    out.z = CalcTotalProbFlat(fwd.get(), bwd.get(), lx, ly);

    FloatBuf post(AllocPost(lx, ly));
    if (post.get() == nullptr) {
        err = "myalloc failed for Post buffer";
        return false;
    }
    // The real reference routine: two thresholds + host expf.
    CalcPostFlat(fwd.get(), bwd.get(), lx, ly, post.get());

    const size_t cells = size_t(lx) * size_t(ly);
    out.post.assign(post.get(), post.get() + cells);
    out.ea = ComputeEaFromPost(out.post.data(), lx, ly);

    if (want_q) {
        out.q.resize(cells);
        ComputeQFromFB(fwd.get(), bwd.get(), lx, ly, out.z, out.q.data());
    } else {
        out.q.clear();
    }

    if (keep_fb) {
        const size_t fb = size_t(HMMSTATE_COUNT) * (size_t(lx) + 1) * (size_t(ly) + 1);
        out.fwd.assign(fwd.get(), fwd.get() + fb);
        out.bwd.assign(bwd.get(), bwd.get() + fb);
    } else {
        out.fwd.clear();
        out.bwd.clear();
    }
    return true;
}

} // namespace muscle_gpu
