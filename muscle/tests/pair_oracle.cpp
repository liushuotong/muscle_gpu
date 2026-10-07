// T02 pair oracle: capture, dump and compare the reference pair intermediates.
//
// The tool deliberately calls the frozen R0 routines (through
// muscle_gpu::RunReferencePair) instead of a re-implementation, so a dump is only
// ever produced by the same operation sequence that the real program uses.
//
// CLI (HANDOFF_AI.md section 5):
//   --suite tiny|threshold|lengths|all
//   --case NAME
//   --backend cpu|gpu
//   --dump DIR
//   --compare DIR
//   --seed UINT
//
// Dump layout, one directory per case:
//   meta.json                 schema, identities, dtype/layout/endianness, file list
//   fwd.bin bwd.bin           cell-major float32 (Lx+1)*(Ly+1)*5   (real DP cases)
//   z.bin                     float32 scalar total probability
//   q.bin                     row-major float32 Lx*Ly
//   post.bin                  row-major float32 Lx*Ly
//   ea.bin                    float32 expected accuracy
//   sparse_row_ptr.bin        uint32 LX+1
//   sparse_col_idx.bin        uint32 nnz
//   sparse_values.bin         float32 nnz
// All binary files are little-endian float32/uint32; JSON never carries floats.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "muscle.h"
#include "gpu/backend_registry.h"
#include "gpu/hmm_snapshot.h"
#include "gpu/pair_backend.h"
#include "gpu/pair_backend_cpu.h"
#include "gpu/pair_reference.h"
#include "gpu/posterior_finalize.h"
#include "gpu/sha256.h"
#include "dump_compare.h"

using namespace muscle_gpu;

namespace {

const char *kSourceManifestSha =
    "ba8726a1c5a34823711c0b4465d0887ac4be7e9035f27ec6b4ea0f97c6c4d222";

struct OracleCase {
    std::string name;
    std::string x;
    std::string y;
    bool synthetic_q;            // q supplied directly (threshold boundary cases)
    std::vector<float> q_values; // used when synthetic_q
    uint32_t lx, ly;             // used when synthetic_q

    OracleCase() : synthetic_q(false), lx(0), ly(0) {}
};

uint32_t g_Seed = 0;
bool g_Verbose = false;

// ---------------------------------------------------------------------------
// Deterministic pseudo-random protein sequences (fixed LCG, not MUSCLE's RNG:
// the oracle fixtures must not depend on any program RNG state).
// ---------------------------------------------------------------------------

uint32_t g_RngState = 1;

void RngSeed(uint32_t s) { g_RngState = (s == 0) ? 1u : s; }

uint32_t RngNext() {
    g_RngState = g_RngState * 1664525u + 1013904223u;
    return g_RngState >> 8;
}

std::string RandomProtein(size_t len, uint32_t salt) {
    static const char *aa = "ACDEFGHIKLMNPQRSTVWY";
    RngSeed(12345u + salt);
    std::string s;
    s.reserve(len);
    for (size_t i = 0; i < len; ++i)
        s.push_back(aa[RngNext() % 20]);
    return s;
}

// ---------------------------------------------------------------------------
// Case suites
// ---------------------------------------------------------------------------

void AddTinyCases(std::vector<OracleCase> &cases) {
    const char *names[] = {"L1xL1", "L1xL2", "L2xL1", "L2xL2", "L3xL3",
                           "identical3", "diverse3", "wildcards", "lower", "UandO"};
    const char *xs[] = {"M", "M", "MK", "MK", "MKA", "MKAV", "MKA", "MKBZXJUO",
                      "mka", "MUO"};
    const char *ys[] = {"M", "MK", "M", "MK", "MKV", "MKAV", "WVY", "MKBZXJUO",
                      "MKA", "MOU"};
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        OracleCase c;
        c.name = names[i];
        c.x = xs[i];
        c.y = ys[i];
        cases.push_back(c);
    }
}

// Boundary cases for the two thresholds.  The host threshold values themselves
// are exercised at ULP neighbours of logf(0.01f) and of 0.0f.
void AddThresholdCases(std::vector<OracleCase> &cases) {
    {
        OracleCase c;
        c.name = "dp_short_divergent";
        c.x = "MWWWW";
        c.y = "MCCCC";
        cases.push_back(c);
    }
    {
        OracleCase c;
        c.name = "dp_long_divergent";
        c.x = "MWWWWWWWWWWWWWWWW";
        c.y = "MCCCCCCCCCCCCCCC";
        cases.push_back(c);
    }
    {
        OracleCase c;
        c.name = "dp_gap_rich";
        c.x = "MKAVMKAVMKAV";
        c.y = "MKAWMKAWMKAW";
        cases.push_back(c);
    }

    // Synthetic q grids around the two decision points.  Values are built with
    // nextafterf so the test lands on adjacent ULPs rather than on "roughly" the
    // threshold.
    const float t = MIN_SPARSE_SCORE;
    std::vector<float> qs;
    for (int d = -3; d <= 3; ++d) {
        float v = t;
        for (int k = 0; k < abs(d); ++k)
            v = (d < 0) ? nextafterf(v, -INFINITY) : nextafterf(v, INFINITY);
        qs.push_back(v);
    }
    for (int d = -3; d <= 3; ++d) {
        float v = 0.0f;
        for (int k = 0; k < abs(d); ++k)
            v = (d < 0) ? nextafterf(v, -INFINITY) : nextafterf(v, INFINITY);
        qs.push_back(v);
    }
    qs.push_back(-1.0f);
    qs.push_back(1.0f);
    {
        OracleCase c;
        c.name = "q_threshold_ulps";
        c.synthetic_q = true;
        c.lx = 1;
        c.ly = uint32_t(qs.size());
        c.q_values = qs;
        cases.push_back(c);
    }
}

void AddLengthCases(std::vector<OracleCase> &cases) {
    const uint32_t lens[] = {1, 2, 31, 32, 33, 127, 128, 129};
    for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); ++i) {
        for (size_t j = 0; j < sizeof(lens) / sizeof(lens[0]); ++j) {
            if (i == j && lens[i] > 33)
                continue;  // keep the suite small but keep both orientations
            char name[64];
            snprintf(name, sizeof(name), "len%ux%u", lens[i], lens[j]);
            OracleCase c;
            c.name = name;
            c.x = RandomProtein(lens[i], lens[i] * 7 + lens[j]);
            c.y = RandomProtein(lens[j], lens[j] * 13 + lens[i]);
            cases.push_back(c);
        }
    }
}

std::vector<OracleCase> BuildCases(const std::string &suite) {
    std::vector<OracleCase> cases;
    if (suite == "tiny" || suite == "all")
        AddTinyCases(cases);
    if (suite == "threshold" || suite == "all")
        AddThresholdCases(cases);
    if (suite == "lengths" || suite == "all")
        AddLengthCases(cases);
    if (cases.empty()) {
        fprintf(stderr, "unknown suite >%s<\n", suite.c_str());
        exit(2);
    }
    return cases;
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

bool WriteBinary(const std::string &path, const void *data, size_t bytes, std::string &err) {
    FILE *f = fopen(path.c_str(), "wb");
    if (f == nullptr) {
        err = "cannot create " + path;
        return false;
    }
    const size_t written = (bytes == 0) ? 0 : fwrite(data, 1, bytes, f);
    if (fclose(f) != 0 || written != bytes) {
        err = "short write to " + path;
        return false;
    }
    return true;
}

bool ReadBinary(const std::string &path, std::vector<uint8_t> &data, std::string &err) {
    FILE *f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open " + path;
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    data.resize(size_t(size));
    const size_t got = data.empty() ? 0 : fread(data.data(), 1, data.size(), f);
    fclose(f);
    if (got != data.size()) {
        err = "short read from " + path;
        return false;
    }
    return true;
}

std::string JsonEscape(const std::string &s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '"' || c == '\\')
            out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

std::string ExecutablePath(const char *argv0) {
#if defined(__linux__)
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        return std::string(buf);
    }
#endif
    return std::string(argv0 == nullptr ? "pair_oracle" : argv0);
}

std::string FileSha256(const std::string &path) {
    std::vector<uint8_t> data;
    std::string err;
    if (!ReadBinary(path, data, err))
        return std::string("unavailable");
    return Sha256::OfBuffer(data.empty() ? "" : (const void *)data.data(), data.size());
}

// ---------------------------------------------------------------------------
// One case result
// ---------------------------------------------------------------------------

struct CaseOutput {
    ReferencePairOutput ref;   // used for cpu
    std::vector<float> fwd, bwd, q, post;
    float z;
    float ea;
    uint32_t lx, ly;
    bool synthetic_q;

    CaseOutput() : z(0), ea(0), lx(0), ly(0), synthetic_q(false) {}
};

bool RunCaseCpu(const OracleCase &c, CaseOutput &out, std::string &err) {
    // Synthetic threshold cases carry their own grid size and no sequences.
    out.lx = c.synthetic_q ? c.lx : uint32_t(c.x.size());
    out.ly = c.synthetic_q ? c.ly : uint32_t(c.y.size());
    out.synthetic_q = c.synthetic_q;
    if (c.synthetic_q) {
        out.q = c.q_values;
        FinalizePostFromQ(out.q.data(), c.lx, c.ly, MIN_SPARSE_SCORE, out.post);
        out.z = 0.0f;
        out.ea = ComputeEaFromPost(out.post.data(), c.lx, c.ly);
        return true;
    }
    if (!RunReferencePair((const uint8_t *)c.x.data(), out.lx, (const uint8_t *)c.y.data(),
                          out.ly, true, true, out.ref, err))
        return false;
    out.fwd = out.ref.fwd;
    out.bwd = out.ref.bwd;
    out.q = out.ref.q;
    out.post = out.ref.post;
    out.z = out.ref.z;
    out.ea = out.ref.ea;
    return true;
}

std::string g_DumpDirForDevice;

bool ReadScalarFloat(const std::string &path, float &value) {
    FILE *f = fopen(path.c_str(), "rb");
    if (f == nullptr)
        return false;
    const size_t got = fread(&value, sizeof(float), 1, f);
    fclose(f);
    return got == 1;
}

bool RunCaseGpu(const OracleCase &c, const HmmSnapshot &snap, CaseOutput &out,
                std::string &err) {
    if (c.synthetic_q) {
        // Threshold cases do not need a device; they test the host finalization.
        out.lx = c.lx;
        out.ly = c.ly;
        out.synthetic_q = true;
        out.q = c.q_values;
        FinalizePostFromQ(out.q.data(), c.lx, c.ly, snap.min_sparse_score, out.post);
        out.z = 0.0f;
        out.ea = ComputeEaFromPost(out.post.data(), c.lx, c.ly);
        return true;
    }

    SequencePool pool;
    const uint32_t lx = uint32_t(c.x.size());
    const uint32_t ly = uint32_t(c.y.size());
    pool.offsets.push_back(0);
    pool.bytes.insert(pool.bytes.end(), (const uint8_t *)c.x.data(),
                      (const uint8_t *)c.x.data() + lx);
    pool.lengths.push_back(lx);
    pool.offsets.push_back(pool.bytes.size());
    pool.bytes.insert(pool.bytes.end(), (const uint8_t *)c.y.data(),
                      (const uint8_t *)c.y.data() + ly);
    pool.lengths.push_back(ly);
    pool.offsets.push_back(pool.bytes.size());

    PairJob job;
    job.job_id = 1;
    job.pair_index = 0;
    job.seq_x = 0;
    job.seq_y = 1;
    job.len_x = lx;
    job.len_y = ly;

    BackendConfig config;
    config.kind = BackendKind::Gpu;
    config.device = 0;
    config.device_budget_bytes = 0;   // 0 = documented default (min(16GiB, 0.75*free))
    config.pinned_budget_bytes = 0;
    config.verify = false;
    config.debug_dump_fb = true;
    config.batch_budget_bytes = 0;

    std::string berr;
    const uint64_t budget = ResolveDeviceBudget(config, berr);
    if (budget == 0) {
        err = "cannot resolve the device budget: " + berr;
        return false;
    }
    config.device_budget_bytes = budget;
    config.batch_budget_bytes = budget;

    std::unique_ptr<PairBackend> backend = CreateCudaPairBackend(config, berr);
    if (!backend) {
        err = "cannot create the CUDA backend: " + berr;
        return false;
    }

    std::vector<PairJob> jobs(1, job);
    std::vector<PairResult> results = backend->RunBatch(snap, pool, jobs);
    if (results.size() != 1) {
        err = "CUDA backend returned a wrong result count";
        return false;
    }
    const PairResult &r = results[0];
    if (r.status != PairStatus::Ok) {
        err = std::string("CUDA backend status ") + ToString(r.status) + ": " + r.reason;
        return false;
    }
    out.lx = lx;
    out.ly = ly;
    out.post = r.post;
    out.q = r.q;
    out.ea = r.ea;
    out.z = 0.0f;

    // Pull the raw F/B/Z copies for the dump; the device dump helper writes
    // fwd.bin/bwd.bin/q.bin/z.bin without a job suffix for a single-job batch.
    PairBackendDebug *dbg = dynamic_cast<PairBackendDebug *>(backend.get());
    if (dbg == nullptr) {
        err = "CUDA backend does not implement PairBackendDebug";
        return false;
    }
    if (!g_DumpDirForDevice.empty()) {
        if (!dbg->DumpLastBatch(g_DumpDirForDevice, err))
            return false;
        // Read Z back from the device dump so the oracle writes one consistent
        // scalar for both backends.
        if (!ReadScalarFloat(g_DumpDirForDevice + "/z.bin", out.z)) {
            err = "cannot read back z.bin from the device dump";
            return false;
        }
    }
    return true;
}

} // namespace

int main(int argc, char **argv) {
    std::string suite = "tiny";
    std::string only_case;
    std::string backend = "cpu";
    std::string dump_dir;
    std::string compare_dir;
    uint32_t seed = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--suite" && i + 1 < argc)
            suite = argv[++i];
        else if (arg == "--case" && i + 1 < argc)
            only_case = argv[++i];
        else if (arg == "--backend" && i + 1 < argc)
            backend = argv[++i];
        else if (arg == "--dump" && i + 1 < argc)
            dump_dir = argv[++i];
        else if (arg == "--compare" && i + 1 < argc)
            compare_dir = argv[++i];
        else if (arg == "--seed" && i + 1 < argc)
            seed = StrToUint(argv[++i]);
        else if (arg == "--verbose")
            g_Verbose = true;
        else if (arg == "-h" || arg == "--help") {
            printf("usage: pair_oracle [--suite tiny|threshold|lengths|all] [--case NAME]\n"
                   "                   [--backend cpu|gpu] [--dump DIR] [--compare DIR]\n"
                   "                   [--seed UINT] [--verbose]\n");
            return 0;
        } else {
            fprintf(stderr, "pair_oracle: unknown argument >%s<\n", arg.c_str());
            return 2;
        }
    }
    g_Seed = seed;

    // Same model setup sequence as align.cpp::Align (defaults -> command line ->
    // optional perturbation -> ToPairHMM), minus the replicate machinery.
    SetAlpha(ALPHA_Amino);
    HMMParams HP;
    HP.FromDefaults(false);
    HP.CmdLineUpdate();
    if (seed > 0) {
        ResetRand(seed);
        HP.PerturbProbs(seed);
    }
    HP.ToPairHMM();

    HmmSnapshot snap;
    std::string err;
    if (!CaptureHmmSnapshot(snap, err)) {
        fprintf(stderr, "pair_oracle: %s\n", err.c_str());
        return 1;
    }
    const std::string hmm_digest = HmmSnapshotDigestHex(snap);

    const bool use_gpu = (backend == "gpu");
    if (use_gpu) {
        std::string why;
        if (!CudaAvailable(why)) {
            fprintf(stderr, "pair_oracle: --backend gpu: %s\n", why.c_str());
            return 1;
        }
    } else if (backend != "cpu") {
        fprintf(stderr, "pair_oracle: unknown backend >%s<\n", backend.c_str());
        return 2;
    }

    std::vector<OracleCase> cases = BuildCases(suite);
    if (!only_case.empty()) {
        std::vector<OracleCase> filtered;
        for (size_t i = 0; i < cases.size(); ++i)
            if (cases[i].name == only_case)
                filtered.push_back(cases[i]);
        if (filtered.empty()) {
            fprintf(stderr, "pair_oracle: no case named >%s< in suite %s\n",
                    only_case.c_str(), suite.c_str());
            return 2;
        }
        cases = filtered;
    }

    const std::string exe_path = ExecutablePath(argc > 0 ? argv[0] : nullptr);
    const std::string exe_sha = FileSha256(exe_path);

    for (size_t ci = 0; ci < cases.size(); ++ci) {
        const OracleCase &c = cases[ci];
        const std::string case_dir = dump_dir.empty() ? std::string() : dump_dir + "/" + c.name;
        if (!case_dir.empty()) {
            std::string cmd = "mkdir -p '" + case_dir + "'";
            if (system(cmd.c_str()) != 0) {
                fprintf(stderr, "pair_oracle: cannot create %s\n", case_dir.c_str());
                return 1;
            }
        }
        g_DumpDirForDevice = case_dir;

        CaseOutput out;
        if (use_gpu) {
            if (!RunCaseGpu(c, snap, out, err)) {
                fprintf(stderr, "pair_oracle: case %s: %s\n", c.name.c_str(), err.c_str());
                return 1;
            }
        } else {
            if (!RunCaseCpu(c, out, err)) {
                fprintf(stderr, "pair_oracle: case %s: %s\n", c.name.c_str(), err.c_str());
                return 1;
            }
        }

        if (!case_dir.empty()) {
            std::vector<std::pair<std::string, size_t> > files;
            if (!out.fwd.empty()) {
                if (!WriteBinary(case_dir + "/fwd.bin", out.fwd.data(),
                                 out.fwd.size() * sizeof(float), err) ||
                    !WriteBinary(case_dir + "/bwd.bin", out.bwd.data(),
                                 out.bwd.size() * sizeof(float), err)) {
                    fprintf(stderr, "pair_oracle: %s\n", err.c_str());
                    return 1;
                }
                files.push_back(std::make_pair("fwd.bin", out.fwd.size() * sizeof(float)));
                files.push_back(std::make_pair("bwd.bin", out.bwd.size() * sizeof(float)));
            }
            if (!WriteBinary(case_dir + "/z.bin", &out.z, sizeof(float), err) ||
                !WriteBinary(case_dir + "/q.bin", out.q.data(), out.q.size() * sizeof(float), err) ||
                !WriteBinary(case_dir + "/post.bin", out.post.data(),
                             out.post.size() * sizeof(float), err) ||
                !WriteBinary(case_dir + "/ea.bin", &out.ea, sizeof(float), err)) {
                fprintf(stderr, "pair_oracle: %s\n", err.c_str());
                return 1;
            }
            files.push_back(std::make_pair("z.bin", sizeof(float)));
            files.push_back(std::make_pair("q.bin", out.q.size() * sizeof(float)));
            files.push_back(std::make_pair("post.bin", out.post.size() * sizeof(float)));
            files.push_back(std::make_pair("ea.bin", sizeof(float)));
            // The GPU path obtains fwd/bwd from the device dump, not from the CPU
            // reference, so record whatever those files actually are on disk.
            if (out.fwd.empty()) {
                for (size_t k = 0; k < 2; ++k) {
                    const char *name = (k == 0) ? "fwd.bin" : "bwd.bin";
                    const std::string path = case_dir + "/" + name;
                    FILE *probe = fopen(path.c_str(), "rb");
                    if (probe == nullptr)
                        continue;
                    fseek(probe, 0, SEEK_END);
                    const long size = ftell(probe);
                    fclose(probe);
                    files.push_back(std::make_pair(std::string(name), size_t(size)));
                }
            }

            // Sparse support set, exported with the real MySparseMx so the format
            // (and the row-end offset rule) matches the program.
            if (out.lx == 0 || out.ly == 0) {
                fprintf(stderr, "pair_oracle: case %s has a degenerate grid\n",
                        c.name.c_str());
                return 1;
            }
            MySparseMx mx;
            mx.FromPost(out.post.data(), out.lx, out.ly);
            const uint32_t nnz = mx.m_Offsets[mx.m_LX];
            std::vector<uint32_t> row_ptr(out.lx + 1), col_idx(nnz);
            std::vector<float> values(nnz);
            for (uint32_t i = 0; i <= out.lx; ++i)
                row_ptr[i] = mx.m_Offsets[i];
            for (uint32_t k = 0; k < nnz; ++k) {
                values[k] = mx.GetProb_Offset(k);
                col_idx[k] = mx.GetCol_Offset(k);
            }
            if (!WriteBinary(case_dir + "/sparse_row_ptr.bin", row_ptr.data(),
                             row_ptr.size() * sizeof(uint32_t), err) ||
                !WriteBinary(case_dir + "/sparse_col_idx.bin", col_idx.data(),
                             col_idx.size() * sizeof(uint32_t), err) ||
                !WriteBinary(case_dir + "/sparse_values.bin", values.data(),
                             values.size() * sizeof(float), err)) {
                fprintf(stderr, "pair_oracle: %s\n", err.c_str());
                return 1;
            }
            files.push_back(std::make_pair("sparse_row_ptr.bin",
                                           row_ptr.size() * sizeof(uint32_t)));
            files.push_back(std::make_pair("sparse_col_idx.bin",
                                           col_idx.size() * sizeof(uint32_t)));
            files.push_back(std::make_pair("sparse_values.bin",
                                           values.size() * sizeof(float)));

            // meta.json
            const std::string input_digest =
                Sha256::OfBuffer((c.x + "\x1f" + c.y).data(), c.x.size() + 1 + c.y.size());
            FILE *f = fopen((case_dir + "/meta.json").c_str(), "wb");
            if (f == nullptr) {
                fprintf(stderr, "pair_oracle: cannot create meta.json\n");
                return 1;
            }
            fprintf(f, "{\n");
            fprintf(f, "  \"schema\": 1,\n");
            fprintf(f, "  \"case\": \"%s\",\n", JsonEscape(c.name).c_str());
            fprintf(f, "  \"suite\": \"%s\",\n", JsonEscape(suite).c_str());
            fprintf(f, "  \"source_id\": \"R0\",\n");
            fprintf(f, "  \"source_manifest_sha256\": \"%s\",\n", kSourceManifestSha);
            fprintf(f, "  \"binary_path\": \"%s\",\n", JsonEscape(exe_path).c_str());
            fprintf(f, "  \"binary_sha256\": \"%s\",\n", exe_sha.c_str());
            fprintf(f, "  \"hmm_digest_sha256\": \"%s\",\n", hmm_digest.c_str());
            fprintf(f, "  \"semantics_version\": %u,\n", PAIR_SEMANTICS_VERSION);
            fprintf(f, "  \"input_digest_sha256\": \"%s\",\n", input_digest.c_str());
            fprintf(f, "  \"seed\": %u,\n", seed);
            fprintf(f, "  \"backend\": \"%s\",\n", JsonEscape(backend).c_str());
            fprintf(f, "  \"synthetic_q\": %s,\n", c.synthetic_q ? "true" : "false");
            fprintf(f, "  \"alphabet\": \"amino\",\n");
            fprintf(f, "  \"Lx\": %u,\n  \"Ly\": %u,\n", out.lx, out.ly);
            fprintf(f, "  \"layout\": {\"fwd\": \"cell-major (i*(Ly+1)+j)*5+s\", "
                       "\"q\": \"row-major i*Ly+j\", \"post\": \"row-major i*Ly+j\", "
                       "\"sparse\": \"row_ptr uint32[Lx+1], col_idx/values uint32/float32[nnz]\"},\n");
            fprintf(f, "  \"dtype\": \"float32\",\n");
            fprintf(f, "  \"endianness\": \"little\",\n");
            fprintf(f, "  \"compiler\": \"%s\",\n", JsonEscape(__VERSION__).c_str());
            fprintf(f, "  \"z\": \"see z.bin\",\n");
            fprintf(f, "  \"ea\": \"see ea.bin\",\n");
            fprintf(f, "  \"nnz\": %u,\n", nnz);
            fprintf(f, "  \"x\": \"%s\",\n", JsonEscape(c.x).c_str());
            fprintf(f, "  \"y\": \"%s\",\n", JsonEscape(c.y).c_str());
            fprintf(f, "  \"files\": [\n");
            for (size_t k = 0; k < files.size(); ++k) {
                const std::string sha = FileSha256(case_dir + "/" + files[k].first);
                fprintf(f, "    {\"name\": \"%s\", \"bytes\": %llu, \"sha256\": \"%s\"}%s\n",
                        files[k].first.c_str(), (unsigned long long)files[k].second,
                        sha.c_str(), (k + 1 == files.size()) ? "" : ",");
            }
            fprintf(f, "  ]\n}\n");
            fclose(f);
        }

        if (g_Verbose || dump_dir.empty()) {
            printf("case %-24s Lx=%-4u Ly=%-4u z=%.6g ea=%.9g post[0]=%.6g\n",
                   c.name.c_str(), out.lx, out.ly, double(out.z), double(out.ea),
                   out.post.empty() ? 0.0 : double(out.post[0]));
        }
    }

    printf("pair_oracle: suite=%s cases=%u backend=%s hmm_sha256=%s\n", suite.c_str(),
           unsigned(cases.size()), backend.c_str(), hmm_digest.c_str());

    // Structural comparison against a reference dump (the Python comparator is
    // authoritative for full reports; this keeps the CLI usable standalone).
    if (!compare_dir.empty()) {
        if (dump_dir.empty()) {
            fprintf(stderr, "pair_oracle: --compare requires --dump\n");
            return 2;
        }
        std::string cmp_err;
        const uint32_t bad = CompareDumpTrees(compare_dir, dump_dir, cmp_err);
        if (bad != 0 || !cmp_err.empty()) {
            fprintf(stderr, "pair_oracle: compare failed: %s\n", cmp_err.c_str());
            return 1;
        }
        printf("pair_oracle: dumps match %s\n", compare_dir.c_str());
    }

    printf("PAIR_ORACLE_OK\n");
    return 0;
}
