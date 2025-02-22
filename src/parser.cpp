#include "parser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <charconv>

// Implementation of the split function
std::vector<std::string> split(const std::string& s, const std::string& delimiter) {
    std::vector<std::string> tokens;
    size_t pos_start = 0, pos_end;
    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        if (pos_end > pos_start) {
            tokens.emplace_back(s.substr(pos_start, pos_end - pos_start));
        }
        pos_start = pos_end + delimiter.length();
    }
    if (pos_start < s.length()) {
        tokens.emplace_back(s.substr(pos_start));
    }
    return tokens;
}

inline bool convert_to_int(const std::string &str, int &num) {
    // Use std::from_chars which avoids exception overhead.
    auto first = str.data();
    auto last = str.data() + str.size();
    auto [ptr, ec] = std::from_chars(first, last, num);
    return ec == std::errc() && ptr == last;
}

ReadFileResult read_file(const std::string& inputFile, const std::string& sep, int minUtil, Timer& timer)
{
    ReadFileResult result;
    // Reserve an estimate for fileData if possible.
    std::vector<std::pair<std::vector<std::string>, std::vector<int>>> fileData;
    fileData.reserve(1024);

    // TWU dictionary: item string -> total weight. Reserve some buckets.
    std::unordered_map<std::string, int> twu;
    twu.reserve(1024);

    std::ifstream infile(inputFile);
    if (!infile) {
        std::cerr << "Error opening file: " << inputFile << "\n";
        std::exit(1);
    }
    std::string line;
    while (std::getline(infile, line)) {
        // Expected format: items : weight : utility_list
        // Items and utility_list are further separated by sep.
        std::vector<std::string> parts = split(line, ":");
        if (parts.size() < 3) {
            std::cerr << "Invalid line: " << line << "\n";
            continue;
        }

        std::vector<std::string> items = split(parts[0], sep);
        int weight = 0;
        if (!convert_to_int(parts[1], weight)) {
            std::cerr << "Invalid weight in line: " << line << "\n";
            continue;
        }

        std::vector<std::string> utilStrs = split(parts[2], sep);
        std::vector<int> utils;
        utils.reserve(utilStrs.size());
        bool valid = true;
        for (const auto &s : utilStrs) {
            int utilVal = 0;
            if (!convert_to_int(s, utilVal)) {
                std::cerr << "Invalid utility value '" << s << "' in line: " << line << "\n";
                valid = false;
                break;
            }
            utils.push_back(utilVal);
        }
        if (!valid)
            continue;

        if (items.size() != utils.size()) {
            std::cerr << "Mismatch between number of items and utilities in line: " << line << "\n";
            continue;
        }

        // Move the transaction data into fileData.
        fileData.emplace_back(std::move(items), std::move(utils));

        // Update TWU: add weight for each item.
        const auto& curItems = fileData.back().first;
        for (const auto &item : curItems) {
            twu[item] += weight;
        }
    }
    infile.close();

    timer.recordPoint("Read file");

    // Filter TWU based on minUtil threshold.
    for (auto it = twu.begin(); it != twu.end();) {
        if (it->second < minUtil)
            it = twu.erase(it);
        else
            ++it;
    }

    // Create a sorted vector of (item, total_weight) in descending order.
    std::vector<std::pair<std::string, int>> sortedTWU(twu.begin(), twu.end());
    std::sort(sortedTWU.begin(), sortedTWU.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Map each item (string) to an integer.
    std::unordered_map<std::string, int> strToInt;
    int t = static_cast<int>(sortedTWU.size());
    result.max_item = t;
    for (const auto &p : sortedTWU) {
        strToInt[p.first] = t;
        result.rename[t] = p.first;
        t--;
    }

    // Build filtered transactions and compute subtree utility.
    std::unordered_map<int, int> subtree;
    // Reserve for transactions if possible.
    std::map<std::vector<int>, std::vector<int>> filteredTransactions;

    for (const auto &entry : fileData) {
        const auto &itemsStr = entry.first;
        const auto &utilsOrig = entry.second;
        std::vector<std::pair<int, int>> transaction;
        transaction.reserve(itemsStr.size());
        for (size_t i = 0; i < itemsStr.size(); ++i) {
            auto it = strToInt.find(itemsStr[i]);
            if (it != strToInt.end()) {
                transaction.emplace_back(it->second, utilsOrig[i]);
            }
        }
        if (transaction.empty())
            continue;

        // Sort transaction by item id.
        std::sort(transaction.begin(), transaction.end(),
                  [](const std::pair<int, int> &a, const std::pair<int, int> &b) { return a.first < b.first; });

        // Build key and value vectors.
        std::vector<int> key;
        std::vector<int> val;
        key.reserve(transaction.size());
        val.reserve(transaction.size());
        for (const auto &p : transaction) {
            key.push_back(p.first);
            val.push_back(p.second);
        }

        // Use key (sorted vector) as key for filteredTransactions.
        auto it = filteredTransactions.find(key);
        if (it == filteredTransactions.end()) {
            filteredTransactions.emplace(std::move(key), std::move(val));
        } else {
            // If the transaction already exists, add corresponding utilities.
            for (size_t i = 0; i < it->second.size(); ++i) {
                it->second[i] += val[i];
            }
        }

        // Compute subtree utility for this transaction.
        int subUtil = std::accumulate(val.begin(), val.end(), 0);
        int temp = 0;
        for (size_t i = 0; i < key.size(); ++i) {
            subtree[key[i]] += subUtil - temp;
            temp += val[i];
        }
    }
    timer.recordPoint("Filter/Merge transactions");

    // Determine primary items: those with subtree utility >= minUtil.
    for (const auto &p : subtree) {
        if (p.second >= minUtil) {
            result.primary.push_back(static_cast<uint32_t>(p.first));
        }
    }

    result.filteredTransactions = std::move(filteredTransactions);
    return result;
}


