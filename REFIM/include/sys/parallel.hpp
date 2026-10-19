#pragma once
// Tiny fork-join helper: runs fn(i) for i in [0, n) on n threads (the caller runs i = 0).

#include <exception>
#include <thread>
#include <vector>

template <class F>
void parallel_run(int n, F&& fn) {
	if (n <= 1) {
		fn(0);
		return;
	}
	std::vector<std::exception_ptr> errors(static_cast<std::size_t>(n));
	const auto guarded = [&](int i) {
		try {
			fn(i);
		} catch (...) {
			errors[static_cast<std::size_t>(i)] = std::current_exception();
		}
	};
	std::vector<std::thread> pool;
	for (int i = 1; i < n; ++i) pool.emplace_back(guarded, i);
	guarded(0);
	for (auto& t : pool) t.join();
	for (auto& e : errors)
		if (e) std::rethrow_exception(e);
}
