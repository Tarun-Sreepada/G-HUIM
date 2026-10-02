#pragma once
// Allocator for the big arrays. Blocks of 2 MiB or more come from per-L3-group (CCD) arenas: a
// worker thread bound to an arena takes power-of-two chunks from that arena's free lists, so
// chunks that were first touched by a worker of this CCD (node-local, warm in its L3 / TLB) are
// recycled by the same CCD instead of going back to the kernel (mmap/munmap + page faults).
// Chunks are mmap'd with MADV_HUGEPAGE. A chunk freed by another thread returns to the arena it
// came from. Threads without an arena (the reader, main) use plain mmap/munmap. Smaller blocks use
// a 64-byte aligned malloc. Arenas live until process exit.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <sys/mman.h>
#include <utility>
#include <vector>

class Arena {
public:
	static constexpr std::size_t kHuge = std::size_t{2} << 20;
	static constexpr std::size_t kHeader = 64;  // keeps user data 64-byte aligned

	// The arena for L3 group `id` (created on first use).
	static Arena* get(int id);
	// Makes `arena` the calling thread's arena (nullptr = none).
	static void bind(Arena* arena) { current_ = arena; }
	static Arena* current() { return current_; }

	// Cumulative counters over all arenas, for the debug log.
	struct Stats {
		std::size_t reused = 0, mapped = 0, bytes_mapped = 0;
	};
	static Stats stats();

	void* allocate_chunk(std::size_t total);  // `total` is a power of two >= kHuge
	void release_chunk(void* base, std::size_t total);

	static void* map_chunk(std::size_t total, Arena* owner);
	static void unmap_chunk(void* base, std::size_t total);

private:
	struct alignas(64) Pool {
		std::atomic_flag lock = ATOMIC_FLAG_INIT;
		std::vector<void*> free;
	};
	static constexpr int kClasses = 48;
	Pool pools_[kClasses];
	std::atomic<std::size_t> reused_{0}, mapped_{0}, bytes_mapped_{0};
	static inline thread_local Arena* current_ = nullptr;
};

template <class T>
struct HugeAllocator {
	using value_type = T;

	HugeAllocator() = default;
	template <class U>
	HugeAllocator(const HugeAllocator<U>&) noexcept {}

	T* allocate(std::size_t n) {
		const std::size_t bytes = n * sizeof(T);
		if (bytes >= Arena::kHuge - Arena::kHeader) {
			std::size_t total = Arena::kHuge;
			while (total < bytes + Arena::kHeader) total <<= 1;
			Arena* arena = Arena::current();
			void* base = arena ? arena->allocate_chunk(total) : Arena::map_chunk(total, nullptr);
			return reinterpret_cast<T*>(static_cast<char*>(base) + Arena::kHeader);
		}
		void* p = std::aligned_alloc(64, (bytes + 63) & ~std::size_t{63});
		if (!p) throw std::bad_alloc();
		return static_cast<T*>(p);
	}

	void deallocate(T* p, std::size_t n) noexcept {
		const std::size_t bytes = n * sizeof(T);
		if (bytes >= Arena::kHuge - Arena::kHeader) {
			std::size_t total = Arena::kHuge;
			while (total < bytes + Arena::kHeader) total <<= 1;
			void* base = reinterpret_cast<char*>(p) - Arena::kHeader;
			Arena* owner = *static_cast<Arena**>(base);
			if (owner)
				owner->release_chunk(base, total);
			else
				Arena::unmap_chunk(base, total);
		} else {
			std::free(p);
		}
	}

	// Default-initialise instead of value-initialise: resize() leaves trivial elements
	// uninitialised (no serial zero-fill of huge arrays; the writer touches the pages first).
	template <class U>
	void construct(U* p) noexcept(noexcept(::new (static_cast<void*>(p)) U)) {
		::new (static_cast<void*>(p)) U;
	}
	template <class U, class... Args>
	void construct(U* p, Args&&... args) {
		::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
	}

	template <class U>
	bool operator==(const HugeAllocator<U>&) const noexcept { return true; }
};

template <class T>
using HugeVector = std::vector<T, HugeAllocator<T>>;

// RAII: binds the calling thread to an arena for its lifetime.
struct ArenaBinding {
	explicit ArenaBinding(Arena* arena) : previous_(Arena::current()) { Arena::bind(arena); }
	~ArenaBinding() { Arena::bind(previous_); }
	ArenaBinding(const ArenaBinding&) = delete;
	ArenaBinding& operator=(const ArenaBinding&) = delete;

private:
	Arena* previous_;
};
