#pragma once
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include "work.cuh"
#include <vector>   

// This header structure is stored in the first two integers of our pattern array.
struct PatternHeader {
    int count;      // Total number of patterns inserted.
    int next_index; // Next free index (starting from HEADER_SIZE).
};

constexpr int HEADER_SIZE = 2;             // Reserve the first two ints.
constexpr int DEFAULT_CAPACITY = 128 * 1024 * 1024; // Total capacity (number of ints)

struct HighUtilityPatterns {
    int32_t* d_patterns;                           // Unified memory for patterns.
    int capacity;                                  // Total number of ints allocated.
    std::unordered_map<int, std::string> rename;   // Map to convert item IDs to strings.

    // Constructor: allocates unified memory and initializes the header.
    HighUtilityPatterns(int cap = DEFAULT_CAPACITY) : capacity(cap) {
        cudaMallocManaged(&d_patterns, capacity * sizeof(int32_t));
        memset(d_patterns, 0, capacity * sizeof(int32_t));
        // Initialize header: count = 0, next_index = HEADER_SIZE.
        d_patterns[0] = 0;             // count
        d_patterns[1] = HEADER_SIZE;   // next free index
    }

    // Destructor: frees the allocated memory.
    ~HighUtilityPatterns() {
        cudaFree(d_patterns);
    }

    // Device-side function to add a new pattern.
    // This function must be called from a kernel.
    __device__ void add_pattern(const WorkItem* wi) {
        // Reinterpret the first two ints as a header.
        PatternHeader* header = reinterpret_cast<PatternHeader*>(d_patterns);
        // Atomically update the pattern count.
        atomicAdd(&header->count, 1);
        // Reserve space for the new pattern: items + 1 utility.
        int spaceNeeded = wi->pattern_length + 1;
        int idx = atomicAdd(&header->next_index, spaceNeeded);

        // Check for capacity overflow.
        if (idx + spaceNeeded > capacity) {
            printf("Exceeded capacity while adding pattern.\n");
            return;
        }

        // Copy the pattern items.
        for (int i = 0; i < wi->pattern_length; i++) {
            d_patterns[idx + i] = wi->pattern[i];
        }
        // Store the utility immediately after the pattern.
        d_patterns[idx + wi->pattern_length] = wi->utility;
    }

    // Host-side function to parse the unified memory array into a map.
    // Each pattern is stored sequentially (starting at index HEADER_SIZE),
    // with each pattern taking (pattern_length items followed by 1 utility).
    std::map<std::string, int> parse_patterns() const {
        std::map<std::string, int> patterns;
        // The header's next_index indicates the total number of ints written.
        int totalElements = d_patterns[1];
        int i = HEADER_SIZE;  // Start after the header.
        while (i < totalElements) {
            std::vector<int> patternItems;
            // Read until we encounter a zero delimiter.
            // (Assumes a 0 delimiter separates patterns.)
            while (i < totalElements && d_patterns[i] != 0) {
                patternItems.push_back(d_patterns[i]);
                i++;
            }
            // Skip over the delimiter.
            i++;

            if (patternItems.empty())
                continue;

            // The last integer is the utility.
            int utility = patternItems.back();
            patternItems.pop_back();

            // Build a string representation using the rename map.
            std::string patternStr;
            for (int id : patternItems) {
                auto it = rename.find(id);
                if (it != rename.end())
                    patternStr += it->second + " ";
                else
                    patternStr += std::to_string(id) + " ";
            }
            if (!patternStr.empty())
                patternStr.pop_back(); // Remove trailing space.

            patterns[patternStr] = utility;
        }
        return patterns;
    }
};
