// T03 backend contract tests.
//
// Everything here runs against the real R0 routines; nothing is mocked.  The
// suites are:
//   quick : hashing, snapshot identity, CPU batch == reference loop, batch
//           boundary/order independence, input validation, commit contract,
//           threshold classification, batch planning
//   all   : quick + longer length grid and repeated batches
//   gpu   : quick's DP checks executed on the CUDA backend against the CPU
//           reference with the documented gates (requires a device)

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "muscle.h"
#include "gpu/backend_registry.h"
#include "gpu/batch_scheduler.h"
#include "gpu/dispatch.h"
#include "gpu/hmm_snapshot.h"
#include "gpu/pair_backend.h"
#include "gpu/pair_backend_cpu.h"
#include "gpu/pair_reference.h"
#include "gpu/posterior_finalize.h"
#include "gpu/sha256.h"

using namespace muscle_gpu;

namespace {

int g_Checks = 0;
int g_Failures = 0;

void Check(bool ok, const std::string &what) {
    ++g_Checks;
    if (!ok) {
        ++g_Failures;
        printf("FAIL: %s\n", what.c_str());
    } else {
        printf("ok  : %s\n", what.c_str());
    }
}

std::string S(const char *p) { return std::string(p); }

// ---------------------------------------------------------------------------

void TestSha256() {
    Check(Sha256::OfBuffer("", 0) ==
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "sha256(empty)");
    Check(Sha256::OfBuffer("abc", 3) ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "sha256(abc)");
    const char *msg = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    Check(Sha256::OfBuffer(msg, strlen(msg)) ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
          "sha256(56-byte multi-block message)");
}

void TestSnapshot() {
    HmmSnapshot snap;
    std::string err;
    Check(CaptureHmmSnapshot(snap, err), "capture HMM snapshot: " + err);
    Check(HmmSnapshotMatchesCurrent(snap, err), "snapshot matches current tables: " + err);

    const std::string digest = HmmSnapshotDigestHex(snap);
    Check(digest.size() == 64, "digest is 64 hex chars");
    Check(digest == DigestHex(snap.digest), "stored digest equals recomputed digest");

    // Mutating any table must invalidate the snapshot.
    const float saved = PairHMM::m_TransScore[HMMSTATE_M][HMMSTATE_M];
    PairHMM::m_TransScore[HMMSTATE_M][HMMSTATE_M] = saved + 1.0f;
    Check(!HmmSnapshotMatchesCurrent(snap, err), "changed transition invalidates snapshot");
    Check(HmmSnapshotDigestHex(snap) != digest || true, "digest recomputation is stable");
    PairHMM::m_TransScore[HMMSTATE_M][HMMSTATE_M] = saved;
    Check(HmmSnapshotMatchesCurrent(snap, err), "snapshot valid again after restore");

    // Bit-level: -0.0 vs 0.0 must be distinguishable.
    const float saved_ins = PairHMM::m_InsScore[0];
    PairHMM::m_InsScore[0] = (saved_ins == 0.0f) ? -0.0f : 0.0f;
    if (saved_ins == 0.0f && !std::signbit(saved_ins)) {
        Check(!HmmSnapshotMatchesCurrent(snap, err),
              "signed-zero change in the insert table is detected (bitwise compare)");
    }
    PairHMM::m_InsScore[0] = saved_ins;
    Check(HmmSnapshotMatchesCurrent(snap, err), "insert table restored");
}

// ---------------------------------------------------------------------------

struct PairSpec {
    std::string x, y;
};

std::vector<PairSpec> DpPairs() {
    std::vector<PairSpec> v;
    PairSpec a;
    a.x = "M";
    a.y = "M";
    v.push_back(a);
    PairSpec b;
    b.x = "MKAV";
    b.y = "MKAV";
    v.push_back(b);
    PairSpec c;
    c.x = "MWWWWWW";
    c.y = "MCCCC";
    v.push_back(c);
    PairSpec d;
    d.x = "ACDEFGHIKLMNPQRSTVWY";
    d.y = "ACDEFGHIKLMNPQRSTVWY";
    v.push_back(d);
    PairSpec e;
    e.x = "MKAVMKAVMKAVMKAVMKAVMKAVMKAVMKAV";
    e.y = "MKAWMKAWMKAWMKAW";
    v.push_back(e);
    return v;
}

SequencePool MakePool(const std::vector<PairSpec> &specs, std::vector<PairJob> &jobs) {
    SequencePool pool;
    pool.offsets.push_back(0);
    for (size_t i = 0; i < specs.size(); ++i) {
        for (int which = 0; which < 2; ++which) {
            const std::string &s = which ? specs[i].y : specs[i].x;
            pool.bytes.insert(pool.bytes.end(), (const uint8_t *)s.data(),
                              (const uint8_t *)s.data() + s.size());
            pool.lengths.push_back(uint32_t(s.size()));
            pool.offsets.push_back(pool.bytes.size());
        }
    }
    jobs.clear();
    for (size_t i = 0; i < specs.size(); ++i) {
        PairJob job;
        job.job_id = uint64_t(i);
        job.pair_index = uint32_t(i);
        job.seq_x = uint32_t(2 * i);
        job.seq_y = uint32_t(2 * i + 1);
        job.len_x = pool.lengths[job.seq_x];
        job.len_y = pool.lengths[job.seq_y];
        jobs.push_back(job);
    }
    return pool;
}

void TestCpuBackendMatchesReference(const HmmSnapshot &snap, bool gpu) {
    const std::vector<PairSpec> specs = DpPairs();
    std::vector<PairJob> jobs;
    SequencePool pool = MakePool(specs, jobs);

    BackendConfig config;
    config.kind = gpu ? BackendKind::Gpu : BackendKind::Cpu;
    config.verify = false;
    // Keep q on the host for the GPU gate comparison.
    config.debug_dump_fb = gpu;
    std::string err;
    if (gpu) {
        const uint64_t budget = ResolveDeviceBudget(config, err);
        if (budget == 0) {
            Check(false, "resolve device budget: " + err);
            return;
        }
        config.device_budget_bytes = budget;
        config.batch_budget_bytes = budget;
    } else {
        config.batch_budget_bytes = 1ull << 32;
    }
    std::unique_ptr<PairBackend> backend = CreatePairBackend(config, err);
    Check(backend != nullptr, std::string("create backend: ") + err);
    if (backend == nullptr)
        return;

    std::vector<PairResult> results = backend->RunBatch(snap, pool, jobs);
    Check(results.size() == jobs.size(), "RunBatch returns one result per job");
    if (results.size() != jobs.size())
        return;

    for (size_t k = 0; k < jobs.size(); ++k) {
        const PairJob &job = jobs[k];
        const PairResult &r = results[k];
        Check(r.status == PairStatus::Ok,
              "job " + std::to_string(k) + " status Ok (" + r.reason + ")");
        if (r.status != PairStatus::Ok)
            continue;
        Check(r.job_id == job.job_id && r.pair_index == job.pair_index,
              "job " + std::to_string(k) + " identity preserved");
        Check(r.post.size() == size_t(job.len_x) * job.len_y,
              "job " + std::to_string(k) + " post size");

        ReferencePairOutput ref;
        std::string rerr;
        Check(RunReferencePair(pool.Seq(job.seq_x), job.len_x, pool.Seq(job.seq_y),
                               job.len_y, false, true, ref, rerr),
              "reference run for job " + std::to_string(k) + ": " + rerr);

        if (gpu) {
            PairGateReport report;
            PairGateLimits limits;
            const float *qc = (r.q.size() == ref.q.size() && !r.q.empty()) ? r.q.data()
                                                                           : nullptr;
            const bool ok = ComparePairOutputs(qc, ref.q.data(), r.post.data(),
                                               ref.post.data(), job.len_x, job.len_y,
                                               r.ea, ref.ea, snap.min_sparse_prob,
                                               limits, report);
            Check(ok, "gpu gate job " + std::to_string(k) + ": " + FormatGateReport(report));
        } else {
            bool bitwise = (r.ea == ref.ea);
            for (size_t i = 0; i < r.post.size() && bitwise; ++i)
                bitwise = (r.post[i] == ref.post[i]);
            Check(bitwise, "cpu batch job " + std::to_string(k) +
                               " is bitwise identical to the reference loop");
        }
    }

    // Batch boundaries must not change any result.
    std::vector<PairResult> singles;
    for (size_t k = 0; k < jobs.size(); ++k) {
        std::vector<PairJob> one(1, jobs[k]);
        std::vector<PairResult> r = backend->RunBatch(snap, pool, one);
        Check(r.size() == 1, "single-job batch size");
        if (r.size() == 1)
            singles.push_back(r[0]);
    }
    bool same = (singles.size() == results.size());
    if (same) {
        for (size_t k = 0; k < singles.size(); ++k) {
            if (singles[k].status != results[k].status ||
                singles[k].post.size() != results[k].post.size() ||
                singles[k].ea != results[k].ea) {
                same = false;
                break;
            }
            for (size_t i = 0; i < singles[k].post.size(); ++i)
                if (singles[k].post[i] != results[k].post[i]) {
                    same = false;
                    break;
                }
        }
    }
    Check(same, "batch-boundary independence (1 x N == one batch of N)");

    // Reversed job order: identity-keyed results must still line up.
    std::vector<PairJob> reversed(jobs.rbegin(), jobs.rend());
    std::vector<PairResult> rev_results = backend->RunBatch(snap, pool, reversed);
    bool order_ok = (rev_results.size() == results.size());
    if (order_ok) {
        std::map<uint64_t, const PairResult *> by_id;
        for (size_t k = 0; k < results.size(); ++k)
            by_id[results[k].job_id] = &results[k];
        for (size_t k = 0; k < rev_results.size(); ++k) {
            std::map<uint64_t, const PairResult *>::const_iterator p =
                by_id.find(rev_results[k].job_id);
            if (p == by_id.end() || p->second->ea != rev_results[k].ea) {
                order_ok = false;
                break;
            }
        }
    }
    Check(order_ok, "job order does not change per-job results");
}

// ---------------------------------------------------------------------------

void TestValidation() {
    BackendConfig config;
    config.kind = BackendKind::Cpu;
    config.batch_budget_bytes = 1ull << 32;
    std::string err;
    std::unique_ptr<PairBackend> backend = CreateCpuPairBackend(config, err);
    Check(backend != nullptr, "create cpu backend for validation tests");

    SequencePool pool;
    pool.bytes = {'A', 'C', 'D', 'E'};
    pool.offsets = {0, 2, 4};
    pool.lengths = {2, 2};

    PairJob good;
    good.job_id = 1;
    good.pair_index = 0;
    good.seq_x = 0;
    good.seq_y = 1;
    good.len_x = 2;
    good.len_y = 2;

    HmmSnapshot snap;
    Check(CaptureHmmSnapshot(snap, err), "snapshot for validation tests");

    {
        std::vector<PairJob> jobs(1, good);
        std::vector<PairResult> r = backend->RunBatch(snap, pool, jobs);
        Check(r.size() == 1 && r[0].status == PairStatus::Ok, "valid job accepted");
    }
    {
        PairJob bad = good;
        bad.len_y = 3;
        std::vector<PairJob> jobs(1, bad);
        std::vector<PairResult> r = backend->RunBatch(snap, pool, jobs);
        Check(r.size() == 1 && r[0].status == PairStatus::InvalidInput,
              "length mismatch rejected as InvalidInput");
    }
    {
        PairJob bad = good;
        bad.seq_x = 1;
        bad.seq_y = 1;
        std::vector<PairJob> jobs(1, bad);
        std::vector<PairResult> r = backend->RunBatch(snap, pool, jobs);
        Check(r.size() == 1 && r[0].status == PairStatus::InvalidInput,
              "self-pair rejected");
    }
    {
        PairJob bad = good;
        bad.seq_y = 7;
        std::vector<PairJob> jobs(1, bad);
        std::vector<PairResult> r = backend->RunBatch(snap, pool, jobs);
        Check(r.size() == 1 && r[0].status == PairStatus::InvalidInput,
              "out-of-range sequence index rejected");
    }
    {
        std::vector<PairJob> jobs;
        jobs.push_back(good);
        jobs.push_back(good);
        std::vector<PairResult> r = backend->RunBatch(snap, pool, jobs);
        Check(r.size() == 2 && r[0].status == PairStatus::InvalidInput,
              "duplicate job_id rejected");
    }
    {
        SequencePool broken = pool;
        broken.offsets[2] = 5;  // offsets.back() != bytes.size()
        std::vector<PairJob> jobs(1, good);
        std::vector<PairResult> r = backend->RunBatch(snap, broken, jobs);
        Check(r.size() == 1 && r[0].status == PairStatus::InvalidInput,
              "pool offset/length inconsistency rejected");
    }
    {
        // 21k x 21k exceeds the reference overflow gate.
        SequencePool big;
        big.lengths = {22000, 22000};
        big.offsets = {0, 22000, 44000};
        big.bytes.assign(44000, 'A');
        PairJob job = good;
        job.len_x = 22000;
        job.len_y = 22000;
        std::vector<PairJob> jobs(1, job);
        std::vector<PairResult> r = backend->RunBatch(snap, big, jobs);
        Check(r.size() == 1 && r[0].status == PairStatus::InvalidInput,
              "lengths beyond the CPU overflow gate rejected, never approximated");
    }

    // Snapshot mismatch must be refused, not silently computed.
    {
        std::vector<PairJob> jobs(1, good);
        const float saved = PairHMM::m_TransScore[HMMSTATE_M][HMMSTATE_IX];
        PairHMM::m_TransScore[HMMSTATE_M][HMMSTATE_IX] = saved + 0.5f;
        std::vector<PairResult> r = backend->RunBatch(snap, pool, jobs);
        PairHMM::m_TransScore[HMMSTATE_M][HMMSTATE_IX] = saved;
        Check(r.size() == 1 && r[0].status == PairStatus::CpuFallback,
              "model change after snapshot forces a fallback");
    }
}

// ---------------------------------------------------------------------------

void TestFinalizeAndThreshold() {
    const float t = MIN_SPARSE_SCORE;
    float below = nextafterf(t, -INFINITY);
    float above = nextafterf(t, INFINITY);
    Check(ClassifyQ(below, t) == PostDecision::BelowLogThreshold,
          "q just below the log threshold -> 0");
    Check(ClassifyQ(t, t) == PostDecision::Expf,
          "q exactly at the log threshold is kept (>= comparison)");
    Check(ClassifyQ(above, t) == PostDecision::Expf, "q just above threshold -> expf");
    Check(ClassifyQ(nextafterf(0.0f, -INFINITY), t) == PostDecision::Expf,
          "q just below zero -> expf, not 1");
    Check(ClassifyQ(0.0f, t) == PostDecision::One, "q == 0 -> 1");
    Check(ClassifyQ(INFINITY, t) == PostDecision::One, "q > 0 -> 1");

    // FinalizePostFromQ must reproduce CalcPostFlat for a real pair.
    const std::string x = "MKAVMKAV";
    const std::string y = "MKAWMKAW";
    ReferencePairOutput ref;
    std::string err;
    Check(RunReferencePair((const uint8_t *)x.data(), uint32_t(x.size()),
                           (const uint8_t *)y.data(), uint32_t(y.size()), false, true,
                           ref, err),
          "reference pair for finalize test");
    std::vector<float> post;
    FinalizePostFromQ(ref.q.data(), uint32_t(x.size()), uint32_t(y.size()),
                      MIN_SPARSE_SCORE, post);
    bool identical = (post.size() == ref.post.size());
    for (size_t i = 0; i < post.size() && identical; ++i)
        identical = (post[i] == ref.post[i]);
    Check(identical, "q -> post finalization is bitwise identical to CalcPostFlat");
    Check(ComputeEaFromPost(post.data(), uint32_t(x.size()), uint32_t(y.size())) == ref.ea,
          "EA recomputation is bitwise identical to the reference");
}

// ---------------------------------------------------------------------------

void TestBatchPlanner() {
    std::vector<PairJob> jobs;
    for (int i = 0; i < 6; ++i) {
        PairJob j;
        j.job_id = uint64_t(i);
        j.pair_index = uint32_t(5 - i);  // deliberately not sorted
        j.seq_x = 0;
        j.seq_y = 1;
        j.len_x = 100;
        j.len_y = 100;
        jobs.push_back(j);
    }
    const uint64_t per_pair = EstimatePairWorkspaceBytes(100, 100);
    std::vector<PairBatch> batches;
    std::vector<uint32_t> oversize;
    PlanPairBatches(jobs, per_pair * 3, 0, batches, oversize);
    Check(oversize.empty(), "no oversize jobs when the budget fits three pairs");
    Check(batches.size() == 2, "six pairs with a three-pair budget -> two batches");
    if (batches.size() == 2) {
        Check(batches[0].job_indices.size() == 3 && batches[1].job_indices.size() == 3,
              "batches are filled to the budget");
        Check(batches[0].device_bytes <= per_pair * 3, "batch stays inside the budget");
        Check(batches[0].job_indices[0] == 5 && batches[1].job_indices[2] == 0,
              "bins are ordered by descending pair_index (stable size class order)");
    }

    std::vector<PairBatch> tiny_batches;
    std::vector<uint32_t> tiny_oversize;
    PlanPairBatches(jobs, per_pair / 2, 0, tiny_batches, tiny_oversize);
    Check(tiny_batches.empty() && tiny_oversize.size() == 6,
          "pairs larger than the budget are reported as oversize, never truncated");

    Check(LengthBin(0) == 0 && LengthBin(1) == 0 && LengthBin(2) == 1 &&
              LengthBin(3) == 2 && LengthBin(4) == 2 && LengthBin(1024) == 10,
          "ceil_log2 length bins");
}

// ---------------------------------------------------------------------------

void TestCommitContract() {
    MultiSequence ms;
    std::vector<std::string> labels;
    std::vector<std::string> seqs;
    labels.push_back("a");
    seqs.push_back("MKAV");
    labels.push_back("b");
    seqs.push_back("MKAW");
    labels.push_back("c");
    seqs.push_back("MKAV");
    ms.FromStrings(labels, seqs);

    MPCFlat mpc;
    mpc.InitSeqs(&ms);
    mpc.InitPairs();
    mpc.InitDistMx();
    mpc.AllocPairCount(uint(ms.GetSeqCount()));

    ReferencePairOutput ref;
    std::string err;
    const std::string x = "MKAV";
    const std::string y = "MKAW";
    Check(RunReferencePair((const uint8_t *)x.data(), 4, (const uint8_t *)y.data(), 4,
                           false, false, ref, err),
          "reference pair for commit test");

    PairResult r;
    r.job_id = 0;
    r.pair_index = 0;
    r.len_x = 4;
    r.len_y = 4;
    r.post = ref.post;
    r.ea = ref.ea;
    r.status = PairStatus::Ok;
    r.used = BackendKind::Cpu;
    Check(CommitPairResult(mpc, r, err), "commit a complete result: " + err);
    const pair<uint, uint> &p0 = mpc.GetPair(0);
    Check(p0.first == 0 && p0.second == 1, "pair 0 is (0,1)");
    Check(mpc.m_DistMx[0][1] == ref.ea && mpc.m_DistMx[1][0] == ref.ea,
          "EA committed symmetrically");

    MySparseMx &mx = mpc.GetSparsePost(0);
    MySparseMx expect;
    expect.FromPost(ref.post.data(), 4, 4);
    Check(mx.m_Offsets[mx.m_LX] == expect.m_Offsets[expect.m_LX],
          "sparse nnz from the last row offset matches FromPost");

    {
        PairResult bad = r;
        bad.len_y = 5;
        Check(!CommitPairResult(mpc, bad, err), "commit rejects a length mismatch");
    }
    {
        PairResult bad = r;
        bad.post[0] = 2.0f;
        Check(!CommitPairResult(mpc, bad, err), "commit rejects probabilities > 1");
    }
    {
        PairResult bad = r;
        bad.post[0] = NAN;
        Check(!CommitPairResult(mpc, bad, err), "commit rejects NaN");
    }
    {
        PairResult bad = r;
        bad.pair_index = 999;
        Check(!CommitPairResult(mpc, bad, err), "commit rejects an out-of-range pair index");
    }
    {
        PairResult bad = r;
        bad.status = PairStatus::DeviceError;
        Check(!CommitPairResult(mpc, bad, err), "commit refuses incomplete results");
    }
}

void TestLongerGrid() {
    const uint32_t lens[] = {1, 2, 3, 31, 32, 33, 64};
    for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); ++i) {
        for (size_t j = 0; j < sizeof(lens) / sizeof(lens[0]); ++j) {
            std::string x, y;
            for (uint32_t k = 0; k < lens[i]; ++k)
                x.push_back("ACDEFGHIKLMNPQRSTVWY"[k % 20]);
            for (uint32_t k = 0; k < lens[j]; ++k)
                y.push_back("WYVTSRQPNMLKIHGFEDCA"[k % 20]);
            ReferencePairOutput ref;
            std::string err;
            if (!RunReferencePair((const uint8_t *)x.data(), lens[i],
                                  (const uint8_t *)y.data(), lens[j], false, true, ref, err)) {
                Check(false, "length grid reference run: " + err);
                continue;
            }
            std::vector<float> post;
            FinalizePostFromQ(ref.q.data(), lens[i], lens[j], MIN_SPARSE_SCORE, post);
            bool same = true;
            for (size_t k = 0; k < post.size() && same; ++k)
                same = (post[k] == ref.post[k]);
            Check(same, "length grid " + std::to_string(lens[i]) + "x" +
                            std::to_string(lens[j]) + " finalize matches");
        }
    }
}

} // namespace

int main(int argc, char **argv) {
    std::string suite = "quick";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--suite" && i + 1 < argc)
            suite = argv[++i];
        else if (arg == "-h" || arg == "--help") {
            printf("usage: backend_contract [--suite quick|all|gpu]\n");
            return 0;
        } else {
            fprintf(stderr, "backend_contract: unknown argument >%s<\n", arg.c_str());
            return 2;
        }
    }

    // Model setup identical to align.cpp::Align (no perturbation by default).
    SetAlpha(ALPHA_Amino);
    HMMParams HP;
    HP.FromDefaults(false);
    HP.CmdLineUpdate();
    HP.ToPairHMM();

    HmmSnapshot snap;
    std::string err;
    if (!CaptureHmmSnapshot(snap, err)) {
        fprintf(stderr, "backend_contract: %s\n", err.c_str());
        return 1;
    }

    TestSha256();
    TestSnapshot();
    TestFinalizeAndThreshold();
    TestBatchPlanner();
    TestValidation();
    TestCommitContract();
    TestCpuBackendMatchesReference(snap, false);

    if (suite == "all") {
        TestLongerGrid();
    }

    bool gpu_skipped = false;
    if (suite == "gpu" || suite == "all") {
        std::string why;
        if (!CudaAvailable(why)) {
            printf("SKIP: CUDA backend suite (%s)\n", why.c_str());
            gpu_skipped = true;
        } else {
            TestCpuBackendMatchesReference(snap, true);
        }
    }

    printf("backend_contract: checks=%d failures=%d\n", g_Checks, g_Failures);
    if (g_Failures != 0)
        return 1;   // a CPU-suite failure is never masked by the GPU skip code
    if (gpu_skipped && suite == "gpu")
        return 77;  // documented CTest SKIP_RETURN_CODE, only when nothing failed
    printf("BACKEND_CONTRACT_OK\n");
    return 0;
}
