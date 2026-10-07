#pragma once
// Modified 2026-10-08: GPU posterior batches for Super5; GPLv3, see NOTICE.md.
#pragma push_macro("byte")
#undef byte
#include <functional>
#include <string>
#include <utility>
#include <vector>
#pragma pop_macro("byte")

namespace muscle_gpu {
void PrepareSuper5Backend();
void RecordSuper5Partition(const char *stage, const std::vector<unsigned> &membership,
    const std::vector<std::string> &paths = {});
// Main-thread only. Ordered labels resolve to immutable UNGAPPED global bytes.
// Post is borrowed only during consume(); callers must not retain its address.
// False means no GPU work was performed: execute the original CPU loop.
bool RunSuper5Pairs(const char *stage,
    const std::vector<std::pair<std::string, std::string>> &labels,
    const std::function<void(size_t, const float *, unsigned, unsigned)> &consume);
}
