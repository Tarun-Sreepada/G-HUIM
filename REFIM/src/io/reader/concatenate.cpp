#include "io/reader/steps.hpp"

#include "sys/parallel.hpp"

#include <algorithm>

Flat concatenate(std::vector<Chunk>& chunks) {
	const int n = static_cast<int>(chunks.size());
	Flat flat;
	auto& keys = flat.keys;
	auto& vals = flat.vals;
	auto& begin = flat.begin;
	std::vector<std::size_t> key_off(chunks.size() + 1, 0), tx_off(chunks.size() + 1, 0);
	for (std::size_t i = 0; i < chunks.size(); ++i) {
		key_off[i + 1] = key_off[i] + chunks[i].keys.size();
		tx_off[i + 1] = tx_off[i] + chunks[i].tx_begin.size() - 1;
	}
	keys.reset(new Item[key_off.back()]);  // uninitialised: filled in parallel below
	vals.reset(new Util[key_off.back()]);
	begin.reset(new std::size_t[tx_off.back() + 1]);
	begin[0] = 0;
	parallel_run(n, [&](int i) {
		const Chunk& c = chunks[i];
		std::copy(c.keys.begin(), c.keys.end(), keys.get() + key_off[i]);
		std::copy(c.vals.begin(), c.vals.end(), vals.get() + key_off[i]);
		for (std::size_t t = 1; t < c.tx_begin.size(); ++t) begin[tx_off[i] + t] = key_off[i] + c.tx_begin[t];
	});
	chunks.clear();
	chunks.shrink_to_fit();
	flat.num_tx = tx_off.back();
	return flat;
}
