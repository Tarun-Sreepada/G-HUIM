/**
 * @file timer.hpp
 * @brief Simple timer and memory profiler for labeled code segments.
 */

 #pragma once

 #include <chrono>
 #include <fstream>
 #include <iomanip>
 #include <iostream>
 #include <sstream>
 #include <stdexcept>
 #include <string>
 #include <vector>
 #include <algorithm>
 #include "memory/memory.cuh"
 
 /**
  * @struct TimeRecord
  * @brief Stores timestamp and memory usage for a labeled point in time.
  */
 struct TimeRecord
{
    std::string label;                                     ///< Label for the time point.
    std::chrono::high_resolution_clock::time_point timestamp;  ///< Time recorded.
    size_t rss;                                            ///< Resident Set Size (memory in KB).
    unsigned long long gpu_mem_offset_before;              ///< GPU allocator offset before the operation.
    unsigned long long gpu_mem_offset_after;               ///< GPU allocator offset after the operation.
};
 
 /**
  * @brief Reads the current memory RSS from /proc/self/status.
  * @return Memory usage in kilobytes.
  */
 inline size_t getMemoryRSS()
 {
     std::ifstream status_file("/proc/self/status");
     std::string line;
     while (std::getline(status_file, line))
     {
         if (line.substr(0, 6) == "VmRSS:")
         {
             std::istringstream iss(line);
             std::string key, unit;
             size_t value;
             if (iss >> key >> value >> unit)
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
     void recordPoint(const std::string &label, unsigned long long before = 0, unsigned long long after = 0)
    {
        TimeRecord rec;
        rec.label = label;
        rec.timestamp = std::chrono::high_resolution_clock::now();
        rec.rss = getMemoryRSS();
        rec.gpu_mem_offset_before = before;
        rec.gpu_mem_offset_after = after;
        records.push_back(rec);
    }
 
     /**
      * @brief Prints all recorded time points and memory usage.
      *
      * Shows differences between points and total elapsed time.
      */
     void printRecords() const
    {
        if (records.empty())
        {
            std::cout << "No records to display." << std::endl;
            return;
        }

        size_t max_label_len = 0;
        for (const auto &rec : records)
            max_label_len = std::max(max_label_len, rec.label.length());

        // IMPROVEMENT: Adjust headers for clarity.
        std::cout << std::left << std::setw(static_cast<int>(max_label_len) + 4) << "Label"
                  << std::right << std::setw(12) << "Time (ms)"
                  << std::right << std::setw(20) << "Before (MB)"
                  << std::right << std::setw(20) << "After (MB)"
                  << std::right << std::setw(20) << "Diff (MB)" << std::endl;
        std::cout << std::string(max_label_len + 76, '-') << std::endl;

        unsigned long long total_gpu_mem_allocated = 0;

        for (size_t i = 0; i < records.size(); ++i)
        {
            auto time_diff = (i == 0) ? 0 : std::chrono::duration_cast<std::chrono::milliseconds>(
                                                  records[i].timestamp - records[i - 1].timestamp)
                                                  .count();

            unsigned long long before_mb = records[i].gpu_mem_offset_before / (1024 * 1024);
            unsigned long long after_mb = records[i].gpu_mem_offset_after / (1024 * 1024);
            long long diff_bytes = records[i].gpu_mem_offset_after - records[i].gpu_mem_offset_before;
            if (records[i].gpu_mem_offset_after < records[i].gpu_mem_offset_before) { // Wrapped
                diff_bytes = (allocators::atomic::allocator_total - records[i].gpu_mem_offset_before) + records[i].gpu_mem_offset_after;
            }
            
            long long diff_mb = diff_bytes / (1024 * 1024);
            total_gpu_mem_allocated += diff_bytes;

            std::cout << std::left << std::setw(static_cast<int>(max_label_len) + 4) << records[i].label
                      << std::right << std::setw(12) << time_diff
                      << std::right << std::setw(20) << before_mb
                      << std::right << std::setw(20) << after_mb
                      << std::right << std::setw(20) << diff_mb
                      << std::endl;
        }

        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                          records.back().timestamp - records.front().timestamp)
                          .count();
        std::cout << std::string(max_label_len + 76, '-') << std::endl;
        std::cout << std::left << std::setw(static_cast<int>(max_label_len) + 4) << "Total time"
                  << std::right << std::setw(12) << total_time << " ms" << std::endl;
        // IMPROVEMENT: Add a final summary for total GPU memory allocated across all iterations.
        std::cout << std::left << std::setw(static_cast<int>(max_label_len) + 4) << "Total GPU Memory Allocated"
                  << std::right << std::setw(12) << (total_gpu_mem_allocated / (1024 * 1024)) << " MB" << std::endl;
    }
 
     int getTimeDifference(const std::string &label1, const std::string &label2) const
     {
         auto it1 = std::find_if(records.begin(), records.end(), [&label1](const TimeRecord &rec)
                                 { return rec.label == label1; });
         auto it2 = std::find_if(records.begin(), records.end(), [&label2](const TimeRecord &rec)
                                 { return rec.label == label2; });
 
         if (it1 == records.end() || it2 == records.end())
             throw std::runtime_error("One or both labels not found.");
 
         return std::chrono::duration_cast<std::chrono::milliseconds>(
                    it2->timestamp - it1->timestamp)
             .count();
     }
 
 private:
     std::vector<TimeRecord> records; ///< List of recorded points.
 };