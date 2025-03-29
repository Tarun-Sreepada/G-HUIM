#pragma once

#include <cstdint>

// __device__ __host__ uint32_t murmur_hash3(const void *key, uint32_t len, uint32_t seed = 42) {
//     uint32_t h = seed;

//     if (len == sizeof(uint32_t)) {
//         // Integer case (direct MurmurHash3 integer hash)
//         uint32_t k = *reinterpret_cast<const uint32_t*>(key);
//         k ^= k >> 16;
//         k *= 0x85ebca6b;
//         k ^= k >> 13;
//         k *= 0xc2b2ae35;
//         k ^= k >> 16;
//         return k ^ seed;
//     } else {
//         // String case (byte-by-byte MurmurHash3 hash)
//         const char* str_key = reinterpret_cast<const char*>(key);
//         for (uint32_t i = 0; i < len; i++) {
//             h ^= str_key[i];
//             h *= 0x5bd1e995;
//             h ^= h >> 15;
//         }
//         return h;
//     }
// }
