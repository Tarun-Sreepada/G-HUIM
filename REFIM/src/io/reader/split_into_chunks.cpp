#include "io/reader/steps.hpp"

#include <algorithm>

std::vector<Chunk> split_into_chunks(std::string_view text, int threads) {
	const int n = text.size() < (std::size_t{1} << 20) ? 1 : threads;
	std::vector<Chunk> chunks(static_cast<std::size_t>(n));
	{
		std::size_t at = 0;
		for (int i = 0; i < n; ++i) {
			std::size_t end = text.size();
			if (i + 1 < n) {
				const std::size_t nl = text.find('\n', std::max(at, text.size() / n * (i + 1)));
				end = nl == std::string_view::npos ? text.size() : nl + 1;
			}
			chunks[i].begin = at;
			chunks[i].end = end;
			at = end;
		}
	}
	return chunks;
}
