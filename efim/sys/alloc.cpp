#include "sys/alloc.hpp"

#include <map>
#include <memory>
#include <mutex>

namespace {

int size_class(std::size_t total) { return __builtin_ctzll(total); }

struct Lock {
	explicit Lock(std::atomic_flag& f) : flag(f) {
		while (flag.test_and_set(std::memory_order_acquire))
			while (flag.test(std::memory_order_relaxed)) {}
	}
	~Lock() { flag.clear(std::memory_order_release); }
	std::atomic_flag& flag;
};

std::mutex g_registry_mutex;
std::map<int, std::unique_ptr<Arena>>& registry() {
	static std::map<int, std::unique_ptr<Arena>>* r = new std::map<int, std::unique_ptr<Arena>>();  // never destroyed
	return *r;
}

}  // namespace

Arena* Arena::get(int id) {
	std::lock_guard<std::mutex> guard(g_registry_mutex);
	auto& slot = registry()[id];
	if (!slot) slot.reset(new Arena());
	return slot.get();
}

Arena::Stats Arena::stats() {
	std::lock_guard<std::mutex> guard(g_registry_mutex);
	Stats s;
	for (auto& [id, a] : registry()) {
		s.reused += a->reused_.load();
		s.mapped += a->mapped_.load();
		s.bytes_mapped += a->bytes_mapped_.load();
	}
	return s;
}

void* Arena::map_chunk(std::size_t total, Arena* owner) {
	// Over-map so the start can be aligned to a huge page, then trim the ends.
	void* raw = ::mmap(nullptr, total + kHuge, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (raw == MAP_FAILED) throw std::bad_alloc();
	const auto start = reinterpret_cast<std::uintptr_t>(raw);
	const std::uintptr_t at = (start + kHuge - 1) & ~(kHuge - 1);
	if (at != start) ::munmap(raw, at - start);
	const std::uintptr_t end = start + total + kHuge;
	if (end != at + total) ::munmap(reinterpret_cast<void*>(at + total), end - (at + total));
	::madvise(reinterpret_cast<void*>(at), total, MADV_HUGEPAGE);
	void* base = reinterpret_cast<void*>(at);
	*static_cast<Arena**>(base) = owner;
	return base;
}

void Arena::unmap_chunk(void* base, std::size_t total) { ::munmap(base, total); }

void* Arena::allocate_chunk(std::size_t total) {
	Pool& pool = pools_[size_class(total)];
	{
		Lock guard(pool.lock);
		if (!pool.free.empty()) {
			void* base = pool.free.back();
			pool.free.pop_back();
			reused_.fetch_add(1, std::memory_order_relaxed);
			return base;
		}
	}
	mapped_.fetch_add(1, std::memory_order_relaxed);
	bytes_mapped_.fetch_add(total, std::memory_order_relaxed);
	return map_chunk(total, this);
}

void Arena::release_chunk(void* base, std::size_t total) {
	Pool& pool = pools_[size_class(total)];
	Lock guard(pool.lock);
	pool.free.push_back(base);
}
