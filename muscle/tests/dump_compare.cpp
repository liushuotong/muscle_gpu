#include "dump_compare.h"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace muscle_gpu {

namespace {

struct FileKind {
    const char *name;
    int kind;  // 0 = float with F/B/Z/q gate, 1 = float with Post gate,
               // 2 = float scalar EA gate, 3 = exact bytes
};

const FileKind kFiles[] = {
    {"fwd.bin", 0},
    {"bwd.bin", 0},
    {"z.bin", 0},
    {"q.bin", 0},
    {"post.bin", 1},
    {"sparse_values.bin", 1},
    {"ea.bin", 2},
    {"sparse_row_ptr.bin", 3},
    {"sparse_col_idx.bin", 3},
};

bool IsDir(const std::string &path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0)
        return false;
    return S_ISDIR(st.st_mode);
}

std::vector<std::string> ListDirs(const std::string &root) {
    std::vector<std::string> out;
    DIR *d = opendir(root.c_str());
    if (d == nullptr)
        return out;
    for (;;) {
        struct dirent *e = readdir(d);
        if (e == nullptr)
            break;
        const std::string name = e->d_name;
        if (name == "." || name == "..")
            continue;
        if (IsDir(root + "/" + name))
            out.push_back(name);
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

bool ReadAll(const std::string &path, std::vector<uint8_t> &data, std::string &err) {
    FILE *f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "cannot open " + path;
        return false;
    }
    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    data.resize(size_t(size < 0 ? 0 : size));
    const size_t got = data.empty() ? 0 : fread(data.data(), 1, data.size(), f);
    fclose(f);
    if (got != data.size()) {
        err = "short read from " + path;
        return false;
    }
    return true;
}

std::string FormatFloat(double v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.9g", v);
    return std::string(buf);
}

} // namespace

uint32_t CompareDumpTrees(const std::string &reference_dir,
                          const std::string &candidate_dir, std::string &err) {
    err.clear();
    if (!IsDir(reference_dir)) {
        err = "reference dump directory does not exist: " + reference_dir;
        return 0xffffffffu;
    }
    if (!IsDir(candidate_dir)) {
        err = "candidate dump directory does not exist: " + candidate_dir;
        return 0xffffffffu;
    }

    const std::vector<std::string> ref_cases = ListDirs(reference_dir);
    const std::vector<std::string> cand_cases = ListDirs(candidate_dir);
    if (ref_cases.empty()) {
        err = "reference dump directory has no case subdirectories";
        return 0xffffffffu;
    }
    if (ref_cases != cand_cases) {
        err = "case sets differ";
        return 0xffffffffu;
    }

    uint32_t failures = 0;
    for (size_t c = 0; c < ref_cases.size(); ++c) {
        const std::string &name = ref_cases[c];
        const std::string ref_dir = reference_dir + "/" + name;
        const std::string cand_dir = candidate_dir + "/" + name;
        bool case_failed = false;

        for (size_t fi = 0; fi < sizeof(kFiles) / sizeof(kFiles[0]); ++fi) {
            const FileKind &fk = kFiles[fi];
            const std::string ref_path = ref_dir + "/" + fk.name;
            const std::string cand_path = cand_dir + "/" + fk.name;

            std::vector<uint8_t> ref_data, cand_data;
            std::string io_err;
            if (!ReadAll(ref_path, ref_data, io_err)) {
                // Optional files (F/B for synthetic cases) may be absent on both
                // sides; that is only an error when the candidate has it.
                if (ReadAll(cand_path, cand_data, io_err)) {
                    err = "case " + name + ": candidate has " + fk.name +
                          " but the reference does not";
                    return 0xffffffffu;
                }
                continue;
            }
            if (!ReadAll(cand_path, cand_data, io_err)) {
                err = "case " + name + ": " + io_err;
                return 0xffffffffu;
            }
            if (ref_data.size() != cand_data.size()) {
                err = "case " + name + ": size mismatch for " + fk.name;
                return 0xffffffffu;
            }
            if (fk.kind == 3) {
                if (!ref_data.empty() &&
                    memcmp(ref_data.data(), cand_data.data(), ref_data.size()) != 0) {
                    size_t k = 0;
                    while (k < ref_data.size() && ref_data[k] == cand_data[k])
                        ++k;
                    char buf[256];
                    snprintf(buf, sizeof(buf),
                             "case %s: %s differs at byte %zu (structural: support set "
                             "must match exactly)",
                             name.c_str(), fk.name, k);
                    err = buf;
                    case_failed = true;
                    break;
                }
                continue;
            }

            const size_t n = ref_data.size() / sizeof(float);
            const float *ref = (const float *)ref_data.data();
            const float *cand = (const float *)cand_data.data();
            double max_abs = 0.0;
            for (size_t k = 0; k < n; ++k) {
                const float a = cand[k];
                const float b = ref[k];
                if (!std::isfinite(a)) {
                    char buf[256];
                    snprintf(buf, sizeof(buf), "case %s: %s has NaN/Inf at index %zu",
                             name.c_str(), fk.name, k);
                    err = buf;
                    case_failed = true;
                    break;
                }
                const double diff = std::fabs(double(a) - double(b));
                if (diff > max_abs)
                    max_abs = diff;
                double allowed = 0.0;
                if (fk.kind == 0)
                    allowed = 2e-4 + 2e-6 * std::fabs(double(b));
                else if (fk.kind == 1)
                    allowed = 2e-6 + 2e-5 * std::fabs(double(b));
                else
                    allowed = 2e-6;
                if (diff > allowed) {
                    char buf[256];
                    snprintf(buf, sizeof(buf),
                             "case %s: %s[%zu] cand=%s ref=%s diff=%s > gate %s",
                             name.c_str(), fk.name, k, FormatFloat(a).c_str(),
                             FormatFloat(b).c_str(), FormatFloat(diff).c_str(),
                             FormatFloat(allowed).c_str());
                    err = buf;
                    case_failed = true;
                    break;
                }
            }
            if (case_failed)
                break;
        }
        if (case_failed)
            ++failures;
    }
    return failures;
}

} // namespace muscle_gpu
