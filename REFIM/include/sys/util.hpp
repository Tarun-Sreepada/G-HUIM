#pragma once

#include <cstddef>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#else
#include <sys/resource.h>
#endif

inline std::size_t peak_rss_bytes() {
#if defined(_WIN32)
	PROCESS_MEMORY_COUNTERS pmc;
	if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return pmc.PeakWorkingSetSize;
	return 0;
#else
	rusage usage{};
	getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
	return static_cast<std::size_t>(usage.ru_maxrss);  // bytes on macOS
#else
	return static_cast<std::size_t>(usage.ru_maxrss) * 1024;  // KiB on Linux
#endif
#endif
}
