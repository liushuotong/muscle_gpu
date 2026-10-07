// T00 OpenMP probe.
//
// "It compiled" is not evidence that OpenMP works: this probe enters a real
// parallel region, collects the OpenMP thread ids that actually participate, and
// fails unless at least the requested number of workers is observed.  It also
// reports omp_get_max_threads() and the number of processors so the recorded
// environment is honest about what the runtime saw.

#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

int main(int argc, char **argv) {
    int requested = 2;
    if (argc > 1)
        requested = atoi(argv[1]);
    if (requested < 1)
        requested = 1;

    std::vector<int> ids;
#pragma omp parallel num_threads(requested)
    {
#pragma omp critical
        ids.push_back(omp_get_thread_num());
    }

    std::set<int> unique_ids(ids.begin(), ids.end());
    const int max_threads = omp_get_max_threads();
    const int processors = omp_get_num_procs();

    printf("OMP_PROBE requested=%d max_threads=%d processors=%d regions=%d workers=%d\n",
           requested, max_threads, processors, (int)ids.size(), (int)unique_ids.size());
    printf("OMP_PROBE ids=");
    for (std::set<int>::const_iterator p = unique_ids.begin(); p != unique_ids.end(); ++p)
        printf(" %d", *p);
    printf("\n");

    if ((int)unique_ids.size() < requested) {
        fprintf(stderr,
                "OMP_PROBE_FAIL: observed %d workers but %d were requested "
                "(OpenMP is not really parallel here)\n",
                (int)unique_ids.size(), requested);
        return 1;
    }
    // Print outside any region so the value is the runtime's own default.
    printf("OMP_PROBE_OK workers=%d\n", (int)unique_ids.size());
    return 0;
}
