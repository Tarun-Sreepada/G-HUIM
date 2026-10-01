#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace ghuim {

// Writes itemsets as "item item ... UTIL u", translating internal names to the input's items.
class ItemsetWriter
{
  public:
    // Internal name n is original_item[n - first_name].
    ItemsetWriter(const std::string& path, const std::vector<uint32_t>& original_item, uint32_t first_name);
    ~ItemsetWriter();
    ItemsetWriter(const ItemsetWriter&)            = delete;
    ItemsetWriter& operator=(const ItemsetWriter&) = delete;

    void write(const uint32_t* names, size_t count, uint64_t utility);
    // Records "name... utility 0" as the depth-first miners collect them.
    void write_records(const uint32_t* words, size_t count);

  private:
    std::FILE*                   file_;
    const std::vector<uint32_t>& original_item_;
    uint32_t                     first_name_;
    std::string                  line_;
};

} // namespace ghuim
