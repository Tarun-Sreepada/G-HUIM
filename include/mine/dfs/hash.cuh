#pragma once
// Open-addressing table keyed by item (key 0 = empty) accumulating per-item utilities. Probes are
// bounded by the table size so a full table is reported instead of looping forever.

#include "mine/dfs/types.cuh"

namespace ghuim::dfs {

__device__ inline uint64_t fasthash(uint64_t i)
{
    i += 1ULL;
    i ^= i >> 33ULL;
    i *= 0xff51afd7ed558ccdULL;
    i ^= i >> 33ULL;
    i *= 0xc4ceb9fe1a85ec53ULL;
    i ^= i >> 33ULL;
    return i;
}

// Slot holding `key`, or -1 if the table does not contain it.
__device__ inline int find_slot(const Utilities* table, uint32_t slots, uint32_t key)
{
    uint32_t slot = fasthash(key) % slots;
    for (uint32_t probes = 0; probes < slots; ++probes) {
        if (table[slot].key == key)
            return static_cast<int>(slot);
        if (table[slot].key == 0)
            return -1;
        slot = slot + 1 == slots ? 0 : slot + 1;
    }
    return -1;
}

enum class Utility
{
    Local,
    Subtree
};

// Adds `utility` to the entry of `key`, claiming a free slot if the key is new. Many threads may call it.
template <Utility Kind>
__device__ inline void accumulate(Control& control, Utilities* table, uint32_t slots, uint32_t key, uint32_t utility)
{
    uint32_t slot = fasthash(key) % slots;
    for (uint32_t probes = 0; probes < slots; ++probes) {
        const uint32_t seen = compare_swap(table[slot].key, 0u, key);
        if (seen == 0 || seen == key) {
            add(Kind == Utility::Local ? table[slot].local_utility : table[slot].subtree_utility, utility);
            return;
        }
        slot = slot + 1 == slots ? 0 : slot + 1;
    }
    control.fail(kTableFull);
}

// The same for a table owned by one thread.
template <Utility Kind>
__device__ inline void accumulate_private(Control& control, Utilities* table, uint32_t slots, uint32_t key,
                                          uint32_t utility)
{
    uint32_t slot = fasthash(key) % slots;
    for (uint32_t probes = 0; probes < slots; ++probes) {
        if (table[slot].key == 0)
            table[slot].key = key;
        if (table[slot].key == key) {
            (Kind == Utility::Local ? table[slot].local_utility : table[slot].subtree_utility) += utility;
            return;
        }
        slot = slot + 1 == slots ? 0 : slot + 1;
    }
    control.fail(kTableFull);
}

} // namespace ghuim::dfs
