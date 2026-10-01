#include "efim.hpp"

#include "reader.hpp"
#include "util.hpp"
#include "worker.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

void Efim::mine() {
	const auto start = std::chrono::steady_clock::now();
	results_.clear();
	rename_.clear();

	auto root = std::make_shared<Level>();
	read_database(input_file_, sep_, min_util_, *root, rename_);

	Scheduler scheduler;
	std::vector<Task> tasks;
	for (std::uint32_t k = 0; k < root->primary.size(); ++k) tasks.push_back({root, 0, k, {}});
	scheduler.push(std::move(tasks));
	root.reset();

	std::vector<std::unique_ptr<Worker>> workers;
	for (int i = 0; i < threads_; ++i) workers.push_back(std::make_unique<Worker>(rename_.size(), min_util_, threads_, scheduler));
	if (threads_ == 1) {
		workers[0]->run();
	} else {
		std::vector<std::thread> pool;
		for (auto& w : workers) pool.emplace_back(&Worker::run, w.get());
		for (auto& t : pool) t.join();
	}
	for (auto& w : workers) results_.push_back(std::move(w->patterns));

	runtime_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	memory_rss_ = peak_rss_bytes();
}

void Efim::save(const std::string& out_file) const {
	std::ofstream out(out_file, std::ios::binary);
	if (!out) throw std::runtime_error("cannot open output file: " + out_file);
	std::string line;
	for (const auto& r : results_) {
		for (std::size_t p = 0; p < r.size(); ++p) {
			line.clear();
			for (std::size_t i = r.begin[p]; i < r.begin[p + 1]; ++i) {
				if (i != r.begin[p]) line += '\t';
				line += rename_[r.items[i]];
			}
			line += ':';
			line += std::to_string(r.utils[p]);
			line += '\n';
			out << line;
		}
	}
}

std::size_t Efim::pattern_count() const {
	std::size_t n = 0;
	for (const auto& r : results_) n += r.size();
	return n;
}

void Efim::print_results() const {
	std::cout << "Total number of High Utility Patterns: " << pattern_count() << '\n';
	std::cout << "Peak Memory in RSS: " << memory_rss_ << '\n';
	std::cout << "Total ExecutionTime in seconds: " << runtime_ << '\n';
}
