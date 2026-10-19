#pragma once
// Named ranges for Nsight Systems. The NVTX v3 header loads its injection library lazily, so a
// normal run pays only a null check; there is nothing to link.

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define GHUIM_NVTX 1
#endif

namespace ghuim {

#ifdef GHUIM_NVTX
// A range that lasts for the enclosing scope.
class ScopedRange
{
  public:
    explicit ScopedRange(const char* name) { nvtxRangePushA(name); }
    ~ScopedRange() { nvtxRangePop(); }
    ScopedRange(const ScopedRange&)            = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
};

// Consecutive ranges: next() closes the current one and opens another.
class PhaseRanges
{
  public:
    PhaseRanges() = default;
    ~PhaseRanges() { end(); }
    void next(const char* name)
    {
        end();
        nvtxRangePushA(name);
        open_ = true;
    }
    void end()
    {
        if (open_)
            nvtxRangePop();
        open_ = false;
    }
    PhaseRanges(const PhaseRanges&)            = delete;
    PhaseRanges& operator=(const PhaseRanges&) = delete;

  private:
    bool open_ = false;
};
#else
class ScopedRange
{
  public:
    explicit ScopedRange(const char*) {}
};

class PhaseRanges
{
  public:
    void next(const char*) {}
    void end() {}
};
#endif

} // namespace ghuim

#define GHUIM_CONCAT_(a, b) a##b
#define GHUIM_CONCAT(a, b) GHUIM_CONCAT_(a, b)
#define GHUIM_RANGE(name) ::ghuim::ScopedRange GHUIM_CONCAT(ghuim_range_, __LINE__)(name)
