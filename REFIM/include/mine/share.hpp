#pragma once
// Root tasks shared between REFIM's workers and other engines (GREFIM's GPUs). With
// SearchOptions::make_share, the workers are not seeded with every root task: each takes the next one from
// the share whenever it has nothing to pop or steal, and the share hands the rest to the other engines.

#include "mine/scheduler.hpp"
#include "mine/search.hpp"

#include <cstdint>
#include <memory>
#include <vector>

inline namespace REFIM_WIDTHS {

class RootShare : public RootShareBase {
public:
	// Once, before any next(): the root database (slot k extends with root->primary[k]) and the item count.
	virtual void start(std::shared_ptr<const Level> root, std::size_t num_items) = 0;
	// The next root slot for CPU worker `worker`, which has finished the last one it took (or is starting);
	// false once the CPU should take no more (and from then on).
	virtual bool next(int worker, std::uint32_t& slot) = 0;
	// After the workers finished: waits for the other engines and returns their pattern count; with
	// collect_patterns, appends their patterns.
	virtual std::size_t finish(std::vector<Patterns>& patterns) = 0;
};

}  // namespace REFIM_WIDTHS
