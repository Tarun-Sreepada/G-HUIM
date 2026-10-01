#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace ghuim {

// Threads worth using for `count` units of work: bounded by the hardware and by a minimum grain.
inline size_t worker_count(size_t count, size_t grain)
{
    const size_t hardware = std::max(1u, std::thread::hardware_concurrency());
    return std::clamp<size_t>(count / std::max<size_t>(grain, 1), 1, hardware);
}

// Runs body(worker, begin, end) over [0, count) split into `workers` contiguous ranges, one per
// thread. The first exception thrown by any worker is rethrown after all have finished.
template <typename Body> void parallel_for(size_t workers, size_t count, Body&& body)
{
    if (workers <= 1) {
        body(size_t{0}, size_t{0}, count);
        return;
    }
    std::exception_ptr failure;
    std::mutex         failure_lock;
    auto               run = [&](size_t w) {
        try {
            body(w, count * w / workers, count * (w + 1) / workers);
        }
        catch (...) {
            std::lock_guard guard(failure_lock);
            if (!failure)
                failure = std::current_exception();
        }
    };
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    for (size_t w = 1; w < workers; ++w)
        threads.emplace_back(run, w);
    run(0);
    for (auto& t : threads)
        t.join();
    if (failure)
        std::rethrow_exception(failure);
}

} // namespace ghuim
