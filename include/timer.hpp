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

// Structure to record time and memory usage.
struct TimeRecord
{
    std::string label;
    std::chrono::high_resolution_clock::time_point timestamp;
    size_t rss; // in kilobytes
};

// Helper function to read current memory RSS (Resident Set Size) in kilobytes.
// This implementation works on Linux by reading /proc/self/status.
inline size_t getMemoryRSS()
{
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line))
    {
        if (line.substr(0, 6) == "VmRSS:")
        {
            std::istringstream iss(line);
            std::string key;
            size_t value;
            std::string unit;
            if (iss >> key >> value >> unit)
            {
                return value; // Value is typically in kB
            }
        }
    }
    return 0;
}

class Timer
{
public:
    // Record a new time point with the provided label and the current RSS memory.
    void recordPoint(const std::string &label)
    {
        TimeRecord rec;
        rec.label = label;
        rec.timestamp = std::chrono::high_resolution_clock::now();
        rec.rss = getMemoryRSS();
        records.push_back(rec);
    }

    // Print the recorded points along with memory usage.
    // The first record is printed as the base (0 ms, base memory),
    // then each subsequent record shows the difference in time and memory usage from the previous one.
    // Finally, prints the total time between the first and the last record.
    void printRecords() const
    {
        if (records.empty())
        {
            std::cout << "No records to display." << std::endl;
            return;
        }

        // Calculate max label length for formatting.
        size_t max_label_len = 0;
        for (const auto &rec : records)
        {
            if (rec.label.length() > max_label_len)
            {
                max_label_len = rec.label.length();
            }
        }

        // Header
        std::cout << std::left << std::setw(static_cast<int>(max_label_len) + 4) << "Label"
                  << std::right << std::setw(12) << "Time (ms)"
                  << std::right << std::setw(16) << "Memory (KB)" << std::endl;
        std::cout << std::string(max_label_len + 32, '-') << std::endl;

        // First point: base values.
        std::cout << std::left << std::setw(static_cast<int>(max_label_len) + 4) << records[0].label
                  << std::right << std::setw(12) << 0
                  << std::right << std::setw(16) << records[0].rss << std::endl;

        // Intermediate points.
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

        // Total time
        auto total = std::chrono::duration_cast<std::chrono::milliseconds>(
                         records.back().timestamp - records.front().timestamp)
                         .count();
        std::cout << std::string(max_label_len + 32, '-') << std::endl;
        std::cout << std::left << std::setw(static_cast<int>(max_label_len) + 4) << "Total time"
                  << std::right << std::setw(12) << total << " ms" << std::endl;
    }

    // Get the time difference between two records identified by their labels.
    // Throws an exception if one or both labels are not found.
    int getTimeDifference(const std::string &label1, const std::string &label2) const
    {
        auto it1 = std::find_if(records.begin(), records.end(), [&label1](const TimeRecord &rec)
                                { return rec.label == label1; });
        auto it2 = std::find_if(records.begin(), records.end(), [&label2](const TimeRecord &rec)
                                { return rec.label == label2; });

        if (it1 == records.end() || it2 == records.end())
        {
            throw std::runtime_error("One or both labels not found.");
        }

        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   it2->timestamp - it1->timestamp)
            .count();
    }

private:
    std::vector<TimeRecord> records;
};
