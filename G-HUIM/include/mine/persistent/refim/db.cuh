#pragma once
// Projected databases of the persistent REFIM search, and the per-group scratch used to build them.
//
// A database is one position-independent block of memory (so it can be copied between a block's stack and the
// heap with memcpy): a header, then its arrays. Every item of a database is secondary at its node (children are
// compacted to the items that survive select), so no per-depth item marks are needed. Transactions keep their
// items ascending; their order in the database is arbitrary.

#include "mine/widths.hpp"

#include <cstdint>

namespace ghuim::inline GHUIM_WIDTHS::refim {

constexpr uint32_t kStackDb  = ~0u;     // DbHeader::cls of a database on a block's stack
constexpr uint32_t kStaticDb = ~0u - 1; // ... of the root, which lives for the whole run

struct alignas(16) DbHeader
{
    uint32_t n_tx;
    uint32_t n_primary;
    uint32_t n_occ;
    uint32_t top; // largest item; an extension by it has nothing after it (a leaf)
    uint64_t n_items;
    uint64_t bytes;
    uint32_t refs;      // heap databases: tasks and frames still using it
    uint32_t cls;       // heap size class, kStackDb or kStaticDb
    uint32_t occ_shift; // an occurrence is tx << occ_shift | pos (the same for every database of a run)
    uint32_t pad[5];
};

__host__ __device__ inline uint64_t align16(uint64_t x)
{
    return (x + 15) & ~uint64_t{15};
}

struct DbLayout
{
    uint64_t tx_off, occ, tx_len, primary, occ_begin, prefix, utils, items, bytes;
};

__host__ __device__ inline DbLayout db_layout(uint64_t n_tx, uint64_t n_items, uint64_t n_primary, uint64_t n_occ)
{
    DbLayout L;
    uint64_t at = sizeof(DbHeader);
    L.tx_off    = at;
    at          = align16(at + 4 * n_tx);
    L.occ       = at;
    at          = align16(at + 4 * n_occ);
    L.tx_len    = at;
    at          = align16(at + 4 * n_tx);
    L.primary   = at;
    at          = align16(at + 4 * n_primary);
    L.occ_begin = at;
    at          = align16(at + 4 * (n_primary + 1));
    L.prefix    = at;
    at          = align16(at + sizeof(Util) * n_tx);
    L.utils     = at;
    at          = align16(at + sizeof(Util) * n_items);
    L.items     = at;
    at          = align16(at + sizeof(Key) * n_items);
    L.bytes     = at;
    return L;
}

// Occurrence of a primary item in transaction `tx` at position `pos`, packed into 32 bits as REFIM does: the position
// takes the low `shift` bits (sized from the longest transaction of the input).
__host__ __device__ inline uint32_t make_occ(uint32_t tx, uint32_t pos, uint32_t shift)
{
    return tx << shift | pos;
}

struct Db
{
    DbHeader* h;
    uint32_t* tx_off; // first entry of each transaction in items/utils
    uint32_t* occ;    // primary[k] occurs at occ[occ_begin[k], occ_begin[k + 1])
    uint32_t* tx_len;
    uint32_t* primary; // items to extend with
    uint32_t* occ_begin;
    Util*     prefix; // utility of the node's itemset in each transaction
    Util*     utils;
    Key*      items;

    __host__ __device__ static Db view(char* base)
    {
        Db d;
        d.h              = reinterpret_cast<DbHeader*>(base);
        const DbLayout L = db_layout(d.h->n_tx, d.h->n_items, d.h->n_primary, d.h->n_occ);
        d.tx_off         = reinterpret_cast<uint32_t*>(base + L.tx_off);
        d.occ            = reinterpret_cast<uint32_t*>(base + L.occ);
        d.tx_len         = reinterpret_cast<uint32_t*>(base + L.tx_len);
        d.primary        = reinterpret_cast<uint32_t*>(base + L.primary);
        d.occ_begin      = reinterpret_cast<uint32_t*>(base + L.occ_begin);
        d.prefix         = reinterpret_cast<Util*>(base + L.prefix);
        d.utils          = reinterpret_cast<Util*>(base + L.utils);
        d.items          = reinterpret_cast<Key*>(base + L.items);
        return d;
    }
};

struct Bound
{
    unsigned long long local, subtree;
};

// Scratch of one expanding group (a block, or the grid in the wide phase), indexed by item. Zero (slot: -1) between
// expansions: each expansion resets what it touched.
struct Scratch
{
    Bound*    bound;
    int32_t*  slot;    // child primary index of an item
    uint32_t* mark;    // 1: touched by the projection, 2: kept (secondary in the child)
    uint32_t* touched; // items with mark != 0
    uint32_t* count;   // per child primary: occurrences, then the write cursor
    uint32_t* chosen;  // child primary items, in slot order
};

} // namespace ghuim::inline GHUIM_WIDTHS::refim
