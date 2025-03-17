
#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

struct TimeRecord {
    std::string label;
    std::chrono::high_resolution_clock::time_point timestamp;
};

class Timer {
public:
    // Record a new time point with the provided label.
    void recordPoint(const std::string& label) {
        TimeRecord rec;
        rec.label = label;
        rec.timestamp = std::chrono::high_resolution_clock::now();
        records.push_back(rec);
    }

    // Print the recorded points.
    // The first record is printed as the base (0 ms),
    // then each subsequent record shows the difference from the previous one.
    // Finally, prints the total time between the first and the last record.
    void printRecords() const {
        if (records.empty()) {
            std::cout << "No records to display." << std::endl;
            return;
        }

        // Print the first record as base
        std::cout << records[0].label << ": " << 0 << " ms" << std::endl;
        for (size_t i = 1; i < records.size(); ++i) {
            auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                records[i].timestamp - records[i-1].timestamp
            ).count();
            std::cout << records[i].label << ": " << diff << " ms" << std::endl;
        }

        // Compute total time from first to last record.
        auto total = std::chrono::duration_cast<std::chrono::milliseconds>(
            records.back().timestamp - records.front().timestamp
        ).count();
        std::cout << "Total time: " << total << " ms" << std::endl;
    }

    // get 2 records and return the time difference between them
    int getTimeDifference(const std::string& label1, const std::string& label2) const {
        auto it1 = std::find_if(records.begin(), records.end(), [&label1](const TimeRecord& rec) {
            return rec.label == label1;
        });
        auto it2 = std::find_if(records.begin(), records.end(), [&label2](const TimeRecord& rec) {
            return rec.label == label2;
        });

        if (it1 == records.end() || it2 == records.end()) {
            throw std::runtime_error("One or both labels not found.");
        }

        return std::chrono::duration_cast<std::chrono::milliseconds>(
            it2->timestamp - it1->timestamp
        ).count();
    }

private:
    std::vector<TimeRecord> records;
};

