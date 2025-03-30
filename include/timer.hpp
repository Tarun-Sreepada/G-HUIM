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
 
 /**
  * @struct TimeRecord
  * @brief Stores timestamp and memory usage for a labeled point in time.
  */
 struct TimeRecord
 {
     std::string label;  ///< Label for the time point.
     std::chrono::high_resolution_clock::time_point timestamp;  ///< Time recorded.
     size_t rss;  ///< Resident Set Size (memory in KB).
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
     void recordPoint(const std::string &label)
     {
         TimeRecord rec;
         rec.label = label;
         rec.timestamp = std::chrono::high_resolution_clock::now();
         rec.rss = getMemoryRSS();
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
 
         std::cout << std::left << std::setw(static_cast<int>(max_label_len) + 4) << "Label"
                   << std::right << std::setw(12) << "Time (ms)"
                   << std::right << std::setw(16) << "Memory (KB)" << std::endl;
         std::cout << std::string(max_label_len + 32, '-') << std::endl;
 
         std::cout << std::left << std::setw(static_cast<int>(max_label_len) + 4) << records[0].label
                   << std::right << std::setw(12) << 0
                   << std::right << std::setw(16) << records[0].rss << std::endl;
 
         for (size_t i = 1; i < records.size(); ++i)
         {
             auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  records[i].timestamp - records[i - 1].timestamp)
                                  .count();
             auto mem_diff = static_cast<int>(records[i].rss) - static_cast<int>(records[i - 1].rss);
             std::cout << std::left << std::setw(static_cast<int>(max_label_len) + 4) << records[i].label
                       << std::right << std::setw(12) << time_diff
                       << std::right << std::setw(16) << records[i].rss
                       << " (" << (mem_diff >= 0 ? "+" : "") << mem_diff << ")" << std::endl;
         }
 
         auto total = std::chrono::duration_cast<std::chrono::milliseconds>(
                          records.back().timestamp - records.front().timestamp)
                          .count();
         std::cout << std::string(max_label_len + 32, '-') << std::endl;
         std::cout << std::left << std::setw(static_cast<int>(max_label_len) + 4) << "Total time"
                   << std::right << std::setw(12) << total << " ms" << std::endl;
     }
 
     /**
      * @brief Gets time difference between two labeled points.
      * @param label1 First label.
      * @param label2 Second label.
      * @return Time in milliseconds.
      * @throw std::runtime_error if label(s) not found.
      */
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
 