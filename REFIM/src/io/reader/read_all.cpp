#include "io/reader/steps.hpp"

#include "sys/parallel.hpp"

#include <cerrno>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

std::string read_all(const std::string& path, int threads) {
	const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
	if (fd < 0) throw std::runtime_error("cannot open input file: " + path);
	struct stat st{};
	if (::fstat(fd, &st) != 0) {
		::close(fd);
		throw std::runtime_error("cannot stat input file: " + path);
	}
	std::string buf(static_cast<std::size_t>(st.st_size), '\0');
	const std::size_t size = buf.size();
	const int n = size < (std::size_t{8} << 20) ? 1 : threads;
	try {
		parallel_run(n, [&](int i) {
			std::size_t pos = size / n * i;
			const std::size_t end = i + 1 == n ? size : size / n * (i + 1);
			while (pos < end) {
				const ssize_t r = ::pread(fd, buf.data() + pos, end - pos, static_cast<off_t>(pos));
				if (r < 0 && errno == EINTR) continue;
				if (r <= 0) throw std::runtime_error("read failed: " + path);
				pos += static_cast<std::size_t>(r);
			}
		});
	} catch (...) {
		::close(fd);
		throw;
	}
	::close(fd);
	return buf;
}
