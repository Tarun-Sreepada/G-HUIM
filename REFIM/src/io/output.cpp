#include "io/output.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace {

constexpr std::size_t kChunk = std::size_t{4} << 20;  // bytes formatted before each pwrite

std::size_t digits(Utility v) {
	std::size_t n = v < 0 ? 2 : 1;
	std::uint64_t u = v < 0 ? ~static_cast<std::uint64_t>(v) + 1 : static_cast<std::uint64_t>(v);
	while (u >= 10) u /= 10, ++n;
	return n;
}

std::size_t size_of(const Patterns& r, const std::vector<std::string>& names) {
	std::size_t total = 0;
	for (std::size_t p = 0; p < r.size(); ++p) {
		const std::size_t b = r.begin[p], e = r.begin[p + 1];
		for (std::size_t i = b; i < e; ++i) total += names[r.items[i]].size();
		total += (e - b - 1) + 1 + digits(r.utils[p]) + 1;  // tabs, ':', utility, '\n'
	}
	return total;
}

void write_all(int fd, const char* data, std::size_t n, std::size_t offset) {
	while (n) {
		const ssize_t w = ::pwrite(fd, data, n, static_cast<off_t>(offset));
		if (w < 0) {
			if (errno == EINTR) continue;
			throw std::runtime_error(std::string("write failed: ") + std::strerror(errno));
		}
		data += w;
		n -= static_cast<std::size_t>(w);
		offset += static_cast<std::size_t>(w);
	}
}

void emit(int fd, const Patterns& r, const std::vector<std::string>& names, std::size_t offset) {
	std::vector<char> buf(kChunk + 4096);
	char* out = buf.data();
	const auto flush = [&] {
		write_all(fd, buf.data(), static_cast<std::size_t>(out - buf.data()), offset);
		offset += static_cast<std::size_t>(out - buf.data());
		out = buf.data();
	};
	for (std::size_t p = 0; p < r.size(); ++p) {
		// One line can exceed the slack only with enormous item names; grow the flush check.
		std::size_t line = 24;
		for (std::size_t i = r.begin[p]; i < r.begin[p + 1]; ++i) line += names[r.items[i]].size() + 1;
		if (static_cast<std::size_t>(out - buf.data()) + line > buf.size()) {
			flush();
			if (line > buf.size()) buf.resize(line);
			out = buf.data();
		}
		for (std::size_t i = r.begin[p]; i < r.begin[p + 1]; ++i) {
			if (i != r.begin[p]) *out++ = '\t';
			const std::string& s = names[r.items[i]];
			std::memcpy(out, s.data(), s.size());
			out += s.size();
		}
		*out++ = ':';
		out = std::to_chars(out, out + 24, r.utils[p]).ptr;
		*out++ = '\n';
	}
	flush();
}

}  // namespace

std::size_t write_patterns(const std::string& path, const std::vector<Patterns>& results,
                           const std::vector<std::string>& names) {
	const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
	if (fd < 0) throw std::runtime_error("cannot open output file: " + path);

	std::vector<std::size_t> bytes(results.size(), 0), offset(results.size(), 0);
	std::vector<std::thread> pool;
	std::vector<std::exception_ptr> errors(results.size());
	const auto run_all = [&](auto&& fn) {
		pool.clear();
		for (std::size_t w = 0; w < results.size(); ++w)
			pool.emplace_back([&, w] {
				try {
					fn(w);
				} catch (...) {
					errors[w] = std::current_exception();
				}
			});
		for (auto& t : pool) t.join();
		for (auto& e : errors)
			if (e) std::rethrow_exception(e);
	};

	try {
		run_all([&](std::size_t w) { bytes[w] = size_of(results[w], names); });
		std::size_t total = 0;
		for (std::size_t w = 0; w < results.size(); ++w) offset[w] = total, total += bytes[w];

		struct stat st{};
		if (total && ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode)) {
			if (::posix_fallocate(fd, 0, static_cast<off_t>(total)) != 0) (void)!::ftruncate(fd, static_cast<off_t>(total));
		}
		run_all([&](std::size_t w) {
			if (bytes[w]) emit(fd, results[w], names, offset[w]);
		});
		::close(fd);
		return total;
	} catch (...) {
		::close(fd);
		throw;
	}
}
