#pragma once

#include <iostream>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <liburing.h>
#include <vector>

struct raw_file
{
    char *data;
    size_t size_bytes;
    size_t submitted_bytes;
    size_t retrieved_bytes;

    int fd;

    raw_file() : data(nullptr), size_bytes(0), submitted_bytes(0),
                 retrieved_bytes(0), fd(-1) {}

    void close()
    {
        if (data)
        {
            free(data);
            data = nullptr;
        }
        if (fd != -1)
        {
            ::close(fd);
            fd = -1;
        }
    }
};

raw_file load_file(std::string filename, int threads);