#pragma once
// Global state of one persistent run: termination counters, abort/watchdog, the wide phase's frontier counts,
// and per-block statistics.

#include <cstdint>

namespace ghuim::persistent {

enum class Abort : uint32_t
{
    kNone     = 0,
    kTimeout  = 1, // the watchdog deadline passed (a hang, or a run longer than allowed)
    kOverflow = 2, // a deque was full when its owner pushed
    kMemory   = 3, // the heap could not hold a database
    kCorrupt  = 4, // a device-side check failed (a bug): the access it guarded was skipped
};

inline const char* abort_name(Abort a)
{
    switch (a) {
    case Abort::kNone:
        return "none";
    case Abort::kTimeout:
        return "timeout";
    case Abort::kOverflow:
        return "deque overflow";
    case Abort::kMemory:
        return "out of device memory";
    case Abort::kCorrupt:
        return "device-side check failed (bug)";
    }
    return "unknown";
}

struct alignas(128) Control
{
    // Tasks seeded or pushed and not yet finished. A task adds its children before it retires itself, so the
    // count reaches 0 only when no task exists anywhere: then every block leaves.
    alignas(128) uint64_t pending;
    alignas(128) uint32_t idle; // blocks looking for work (REFIM's "hungry" signal)
    uint32_t abort;             // Abort, set once by the first block that fails
    uint64_t deadline_ns;       // %globaltimer value at which the run aborts; 0 = never
    // Wide phase: tasks in each frontier buffer and the largest task size among them.
    uint32_t frontier_size[2];
    uint64_t largest[2];
    uint32_t wide_levels; // levels run by the whole grid before the work-stealing phase
    uint32_t failed_line; // Abort::kCorrupt: source line of the first failed device-side check
    uint64_t debug[6];    // ... and values the check saw (check-specific)
};

struct BlockStats
{
    uint64_t tasks;        // run by this block in the work-stealing phase
    uint64_t wide_tasks;   // run by the whole grid, credited to the block that emitted their children
    uint64_t steals_dsmem; // from the shared-memory deque of a block in the same cluster
    uint64_t steals_near;  // from the global deque of a block in the same cluster
    uint64_t steals_far;   // from the global deque of a block in another cluster
    uint64_t steal_misses; // empty victim or lost race
    uint64_t exports;      // tasks moved from the own shared-memory deque to the own global deque
    uint64_t busy_ns;      // running tasks (work-stealing phase)
    uint64_t idle_ns;      // looking for work
    uint64_t max_local;    // largest shared-memory deque seen after a push
    uint64_t max_global;   // largest global deque seen after a push
    uint64_t work[4];      // Work-defined counters, summed by the host (e.g. patterns found)
};

} // namespace ghuim::persistent
