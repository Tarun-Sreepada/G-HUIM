/**
 * @file timer.hpp
 * @brief Simple timer and memory profiler for labeled code segments.
 */

#pragma once

#include "memory/memory.cuh"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric> // Required for std::accumulate
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @struct TimeRecord
 * @brief Stores timestamp and memory usage for a labeled point in time.
 */
struct TimeRecord
{
    std::string                                    label;     ///< Label for the time point.
    std::chrono::high_resolution_clock::time_point timestamp; ///< Time recorded.
    size_t                                         rss;       ///< Resident Set Size (memory in KB).
    unsigned long long    gpu_mem_offset_before;              ///< GPU allocator offset before the operation.
    unsigned long long    gpu_mem_offset_after;               ///< GPU allocator offset after the operation.
    std::vector<uint32_t> distributions;
};

/**
 * @brief Reads the current memory RSS from /proc/self/status.
 * @return Memory usage in kilobytes.
 */
inline size_t getMemoryRSS()
{
    std::ifstream status_file("/proc/self/status");
    if (!status_file.is_open()) {
        return 0;
    }
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line.substr(7)); // Extract value part
            size_t             value;
            if (iss >> value)
                return value;
        }
    }
    return 0;
}

/**
 * @class Timer
 * @brief Tracks execution time and memory usage between labeled checkpoints.
 */
class Timer
{
  public:
    /**
     * @brief Records a timestamp and current memory with a label.
     * @param label Label to identify the checkpoint.
     */
    void recordPoint(const std::string& label, unsigned long long before = 0, unsigned long long after = 0,
                     const std::vector<uint32_t>& distributions = {})
    {
        TimeRecord rec;
        rec.label                 = label;
        rec.timestamp             = std::chrono::high_resolution_clock::now();
        rec.rss                   = getMemoryRSS();
        rec.gpu_mem_offset_before = before;
        rec.gpu_mem_offset_after  = after;
        rec.distributions         = distributions;
        records.push_back(rec);
    }

    /**
     * @brief Prints all recorded time points and memory usage with dynamically adjusted columns.
     *
     * Shows differences between points and total elapsed time.
     */
    void printRecords() const
    {
        if (records.empty()) {
            std::cout << "No records to display." << std::endl;
            return;
        }

        // --- Pre-computation for column widths ---
        size_t max_label_len      = 10; // "Label"
        size_t max_time_len       = 12; // "Time (ms)"
        size_t max_before_len     = 12; // "Before (MB)"
        size_t max_after_len      = 11; // "After (MB)"
        size_t max_diff_len       = 10; // "Diff (MB)"
        size_t max_dist_count_len = 5;  // "Dists"

        for (const auto& rec : records) {
            max_label_len = std::max(max_label_len, rec.label.length());
            max_before_len =
                    std::max(max_before_len, std::to_string(rec.gpu_mem_offset_before / (1024 * 1024)).length());
            max_after_len = std::max(max_after_len, std::to_string(rec.gpu_mem_offset_after / (1024 * 1024)).length());
            max_dist_count_len = std::max(max_dist_count_len, std::to_string(rec.distributions.size()).length());

            long long diff_bytes = rec.gpu_mem_offset_after - rec.gpu_mem_offset_before;
            if (rec.gpu_mem_offset_after < rec.gpu_mem_offset_before) { // Wrapped
                diff_bytes =
                        (allocators::atomic::allocator_total - rec.gpu_mem_offset_before) + rec.gpu_mem_offset_after;
            }
            max_diff_len = std::max(max_diff_len, std::to_string(diff_bytes / (1024 * 1024)).length());
        }

        for (size_t i = 1; i < records.size(); ++i) {
            auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(records[i].timestamp -
                                                                                   records[i - 1].timestamp)
                                     .count();
            max_time_len = std::max(max_time_len, std::to_string(time_diff).length());
        }

        // --- Print Header ---
        const int padding = 2;
        std::cout << std::left << std::setw(static_cast<int>(max_label_len) + padding) << "Label" << std::right
                  << std::setw(static_cast<int>(max_time_len) + padding) << "Time (ms)" << std::right
                  << std::setw(static_cast<int>(max_before_len) + padding) << "Before (MB)" << std::right
                  << std::setw(static_cast<int>(max_after_len) + padding) << "After (MB)" << std::right
                  << std::setw(static_cast<int>(max_diff_len) + padding) << "Diff (MB)" << std::right
                  << std::setw(static_cast<int>(max_dist_count_len) + padding) << "Dists"
                  << "  Distributions" << std::endl;

        size_t total_width = max_label_len + max_time_len + max_before_len + max_after_len + max_diff_len +
                             max_dist_count_len + (padding * 6) + 18;
        std::cout << std::string(total_width, '-') << std::endl;

        // --- Print Records ---
        unsigned long long total_gpu_mem_allocated = 0;
        for (size_t i = 0; i < records.size(); ++i) {
            auto time_diff = (i == 0) ? 0
                                      : std::chrono::duration_cast<std::chrono::milliseconds>(records[i].timestamp -
                                                                                              records[i - 1].timestamp)
                                                .count();

            unsigned long long before_mb  = records[i].gpu_mem_offset_before / (1024 * 1024);
            unsigned long long after_mb   = records[i].gpu_mem_offset_after / (1024 * 1024);
            long long          diff_bytes = records[i].gpu_mem_offset_after - records[i].gpu_mem_offset_before;

            if (records[i].gpu_mem_offset_after < records[i].gpu_mem_offset_before) { // Wrapped
                diff_bytes = (allocators::atomic::allocator_total - records[i].gpu_mem_offset_before) +
                             records[i].gpu_mem_offset_after;
            }

            long long diff_mb = diff_bytes / (1024 * 1024);
            total_gpu_mem_allocated += diff_bytes;

            std::cout << std::left << std::setw(static_cast<int>(max_label_len) + padding) << records[i].label
                      << std::right << std::setw(static_cast<int>(max_time_len) + padding) << time_diff << std::right
                      << std::setw(static_cast<int>(max_before_len) + padding) << before_mb << std::right
                      << std::setw(static_cast<int>(max_after_len) + padding) << after_mb << std::right
                      << std::setw(static_cast<int>(max_diff_len) + padding) << diff_mb << std::right
                      << std::setw(static_cast<int>(max_dist_count_len) + padding) << records[i].distributions.size();

            // Correctly print distributions
            std::stringstream ss;
            ss << "  [";
            for (size_t j = 0; j < records[i].distributions.size(); ++j) {
                ss << records[i].distributions[j] << (j < records[i].distributions.size() - 1 ? ", " : "");
            }
            ss << "]";
            std::cout << ss.str() << std::endl;
        }

        // --- Print Footer ---
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(records.back().timestamp -
                                                                                records.front().timestamp)
                                  .count();
        std::cout << std::string(total_width, '-') << std::endl;
        std::cout << std::left << std::setw(static_cast<int>(max_label_len) + padding) << "Total time" << std::right
                  << std::setw(static_cast<int>(max_time_len) + padding) << total_time << " ms" << std::endl;
        std::cout << std::left << std::setw(static_cast<int>(max_label_len) + padding) << "Total GPU Memory Allocated"
                  << std::right << std::setw(static_cast<int>(max_time_len) + padding)
                  << (total_gpu_mem_allocated / (1024 * 1024)) << " MB" << std::endl;
    }

    int getTimeDifference(const std::string& label1, const std::string& label2) const
    {
        auto it1 = std::find_if(records.begin(), records.end(),
                                [&label1](const TimeRecord& rec) { return rec.label == label1; });
        auto it2 = std::find_if(records.begin(), records.end(),
                                [&label2](const TimeRecord& rec) { return rec.label == label2; });

        if (it1 == records.end() || it2 == records.end())
            throw std::runtime_error("One or both labels not found.");

        return std::chrono::duration_cast<std::chrono::milliseconds>(it2->timestamp - it1->timestamp).count();
    }

  private:
    std::vector<TimeRecord> records; ///< List of recorded points.
};