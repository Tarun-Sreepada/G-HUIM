#include "mine/bfs/host/itemsets.hpp"

#include <algorithm>

namespace ghuim::bfs {

// Rebuilds each itemset by following parent links back through the levels.
void write_itemsets(const std::vector<std::vector<Node>>& levels, uint64_t minimum_utility, ItemsetWriter& writer)
{
    std::vector<uint32_t> names;
    for (size_t level = 0; level < levels.size(); ++level) {
        for (const Node& node : levels[level]) {
            if (node.utility < minimum_utility)
                continue;
            names.assign(1, node.item);
            uint32_t parent = node.parent;
            for (size_t up = level; up-- > 0;) {
                const Node& ancestor = levels[up][parent];
                names.push_back(ancestor.item);
                parent = ancestor.parent;
            }
            std::reverse(names.begin(), names.end());
            writer.write(names.data(), names.size(), node.utility);
        }
    }
}

} // namespace ghuim::bfs
