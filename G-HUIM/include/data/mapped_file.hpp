#pragma once

#include <cstddef>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ghuim {

class MappedFile
{
public:
    explicit MappedFile(const std::string& path)
    {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0)
            throw std::runtime_error("could not open input: " + path);
        struct stat info{};
        if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) {
            ::close(fd);
            throw std::runtime_error("input is not a regular file: " + path);
        }
        size_ = static_cast<size_t>(info.st_size);
        if (size_ != 0) {
            void* mapping = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
            if (mapping == MAP_FAILED) {
                ::close(fd);
                throw std::runtime_error("could not map input: " + path);
            }
            ::madvise(mapping, size_, MADV_SEQUENTIAL);
            data_ = static_cast<const char*>(mapping);
        }
        ::close(fd);
    }
    ~MappedFile()
    {
        if (data_ != nullptr)
            ::munmap(const_cast<char*>(data_), size_);
    }
    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    std::string_view view() const
    {
        return {data_, size_};
    }

private:
    const char* data_ = nullptr;
    size_t      size_ = 0;
};

} // namespace ghuim
