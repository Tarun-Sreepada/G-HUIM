#include "mine/output.hpp"

#include <stdexcept>

namespace ghuim {

ItemsetWriter::ItemsetWriter(const std::string& path, const std::vector<uint32_t>& original_item, uint32_t first_name)
    : file_(std::fopen(path.c_str(), "w")), original_item_(original_item), first_name_(first_name)
{
    if (file_ == nullptr)
        throw std::runtime_error("cannot open output file: " + path);
}

ItemsetWriter::~ItemsetWriter()
{
    if (file_ != nullptr)
        std::fclose(file_);
}

void ItemsetWriter::write(const uint32_t* names, size_t count, uint64_t utility)
{
    line_.clear();
    for (size_t i = 0; i < count; ++i) {
        line_ += std::to_string(original_item_.at(names[i] - first_name_));
        line_.push_back(' ');
    }
    line_ += "UTIL ";
    line_ += std::to_string(utility);
    line_.push_back('\n');
    std::fwrite(line_.data(), 1, line_.size(), file_);
}

} // namespace ghuim
