// 14: compute data compression of tile padding.
// A 1 GiB tiled buffer (tiles of 32 lanes x 32 rows of u32, so one row is one 128 B line) holds hashed values
// with a fraction of zero padding, in two shapes:
//   rows   the last rows of every tile are zero (length spread: shorter transactions end early),
//   lanes  the last lanes of every tile are zero in every row (a partly filled tile).
// The buffer comes from cudaMalloc (never compressed) or from cuMemCreate with CU_MEM_ALLOCATION_COMP_GENERIC,
// where the L2 may store lines that compress (zeros) in fewer DRAM bytes. Timed per memory and padding:
//   fill      coalesced write of the whole buffer (padding written as zeros),
//   read      coalesced read of the whole buffer (uint4 loads),
//   scatter   tile_write's pattern: lane l of a warp writes its column into its own tile, 4 B per store,
//             only the real (non-padding) values; the padding stays zero from a memset.
// Rates count the buffer's bytes (scatter: the bytes it writes). ok = the read sum matches the expected one.

#include "bench.cuh"

#include <cuda.h>

constexpr uint32_t kLanes = 32, kRows = 32, kTileWords = kLanes * kRows;
constexpr size_t   kWords = size_t{1} << 28; // 1 GiB of u32
constexpr uint32_t kTiles = static_cast<uint32_t>(kWords / kTileWords);

#define CU_CHECK(call)                                                                                         \
    do {                                                                                                       \
        CUresult r_ = (call);                                                                                  \
        if (r_ != CUDA_SUCCESS) {                                                                              \
            const char* s_ = nullptr;                                                                          \
            cuGetErrorString(r_, &s_);                                                                         \
            std::fprintf(stderr, "CU error: %s at %s:%d: %s\n", #call, __FILE__, __LINE__, s_ ? s_ : "?");     \
            std::exit(1);                                                                                      \
        }                                                                                                      \
    } while (0)

// Device memory from the VMM API, optionally compressible. `compressed` reports what the driver granted.
struct VmmBuffer
{
    CUdeviceptr                  ptr  = 0;
    size_t                       size = 0;
    CUmemGenericAllocationHandle handle{};
    bool                         compressed = false;

    VmmBuffer(size_t bytes, bool compressible)
    {
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUmemAllocationProp prop{};
        prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id   = dev;
        if (compressible)
            prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
        size_t granularity = 0;
        CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        size = (bytes + granularity - 1) / granularity * granularity;
        CU_CHECK(cuMemCreate(&handle, size, &prop, 0));
        CUmemAllocationProp got{};
        CU_CHECK(cuMemGetAllocationPropertiesFromHandle(&got, handle));
        compressed = got.allocFlags.compressionType == CU_MEM_ALLOCATION_COMP_GENERIC;
        CU_CHECK(cuMemAddressReserve(&ptr, size, granularity, 0, 0));
        CU_CHECK(cuMemMap(ptr, size, 0, handle, 0));
        CUmemAccessDesc access{};
        access.location = prop.location;
        access.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU_CHECK(cuMemSetAccess(ptr, size, &access, 1));
    }
    ~VmmBuffer()
    {
        cuMemUnmap(ptr, size);
        cuMemAddressFree(ptr, size);
        cuMemRelease(handle);
    }
    uint32_t* get() const { return reinterpret_cast<uint32_t*>(ptr); }
};

// Padding shape: kRowsShape zeroes the last `pad` rows of a tile, kLanesShape its last `pad` lanes.
enum Shape : uint32_t { kRowsShape, kLanesShape };

__host__ __device__ inline bool is_pad(uint32_t shape, uint32_t pad, uint32_t row, uint32_t lane)
{
    return shape == kRowsShape ? row >= kRows - pad : lane >= kLanes - pad;
}

__host__ __device__ inline uint32_t value_at(size_t i)
{
    return hash32(static_cast<uint32_t>(i) ^ static_cast<uint32_t>(i >> 32)) | 1u; // never zero
}

__global__ void fill_kernel(uint32_t* v, uint32_t shape, uint32_t pad)
{
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < kWords; i += (size_t)gridDim.x * blockDim.x) {
        const uint32_t w = static_cast<uint32_t>(i % kTileWords);
        v[i]             = is_pad(shape, pad, w / kLanes, w % kLanes) ? 0u : value_at(i);
    }
}

__global__ void read_kernel(const uint4* v, u64* total)
{
    u64 acc = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < kWords / 4; i += (size_t)gridDim.x * blockDim.x) {
        const uint4 q = v[i];
        acc += (u64)q.x + q.y + q.z + q.w;
    }
    atomicAdd(total, acc);
}

// Thread t owns column (lane t / kTiles) of tile t % kTiles: neighbouring threads write to neighbouring tiles.
__global__ void scatter_kernel(uint32_t* v, uint32_t shape, uint32_t pad)
{
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (t >= size_t{kTiles} * kLanes)
        return;
    const uint32_t tile = static_cast<uint32_t>(t % kTiles), lane = static_cast<uint32_t>(t / kTiles);
    const size_t   base = size_t{tile} * kTileWords + lane;
    for (uint32_t row = 0; row < kRows; ++row)
        if (!is_pad(shape, pad, row, lane))
            v[base + size_t{row} * kLanes] = value_at(base + size_t{row} * kLanes);
}

u64 expected_sum(uint32_t shape, uint32_t pad)
{
    // Every tile has the same padding, but the values differ per word: sum them on the host once per config.
    u64 sum = 0;
    for (size_t i = 0; i < kWords; ++i) {
        const uint32_t w = static_cast<uint32_t>(i % kTileWords);
        if (!is_pad(shape, pad, w / kLanes, w % kLanes))
            sum += value_at(i);
    }
    return sum;
}

u64 read_sum(uint32_t* v)
{
    DeviceBuffer<u64> total(1);
    total.zero();
    read_kernel<<<sm_count() * 8, 256>>>(reinterpret_cast<const uint4*>(v), total.get());
    CUDA_CHECK(cudaGetLastError());
    return read_scalar(total.get());
}

void run(const char* memory, uint32_t* v, uint32_t shape, uint32_t pad, u64 expect)
{
    const char*  shape_name = shape == kRowsShape ? "rows" : "lanes";
    const double padding    = static_cast<double>(pad) / (shape == kRowsShape ? kRows : kLanes);
    const double bytes      = kWords * 4.0;
    const int    grid       = sm_count() * 8;

    const float fill_ms = time_ms([&] { fill_kernel<<<grid, 256>>>(v, shape, pad); });
    const bool  fill_ok = read_sum(v) == expect;
    std::printf("%s,%s,%.3f,fill,%.4f,%.1f,%d\n", memory, shape_name, padding, fill_ms, bytes / fill_ms / 1e6, fill_ok);

    DeviceBuffer<u64> total(1);
    u64               got = 0;
    const float       read_ms = time_ms([&] {
        CUDA_CHECK(cudaMemsetAsync(total.get(), 0, sizeof(u64)));
        read_kernel<<<grid, 256>>>(reinterpret_cast<const uint4*>(v), total.get());
    });
    got = read_scalar(total.get());
    std::printf("%s,%s,%.3f,read,%.4f,%.1f,%d\n", memory, shape_name, padding, read_ms, bytes / read_ms / 1e6, got == expect);

    const uint32_t threads   = 256;
    const uint32_t blocks    = static_cast<uint32_t>((size_t{kTiles} * kLanes + threads - 1) / threads);
    const float    scatter_ms = time_ms([&] {
        CUDA_CHECK(cudaMemsetAsync(v, 0, kWords * 4));
        scatter_kernel<<<blocks, threads>>>(v, shape, pad);
    });
    // The memset is part of every timed run; time it alone and subtract it.
    const float memset_ms = time_ms([&] { CUDA_CHECK(cudaMemsetAsync(v, 0, kWords * 4)); });
    scatter_kernel<<<blocks, threads>>>(v, shape, pad);
    CUDA_CHECK(cudaGetLastError());
    const bool scatter_ok = read_sum(v) == expect;
    std::printf("%s,%s,%.3f,scatter,%.4f,%.1f,%d\n", memory, shape_name, padding, scatter_ms - memset_ms,
                bytes * (1.0 - padding) / (scatter_ms - memset_ms) / 1e6, scatter_ok);
}

int main()
{
    print_device();
    CUDA_CHECK(cudaFree(nullptr)); // the runtime's primary context serves the driver calls too
    int dev = 0, supported = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CU_CHECK(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, dev));
    std::fprintf(stderr, "# generic compression supported: %d\n", supported);

    DeviceBuffer<uint32_t> plain(kWords);
    VmmBuffer              vmm(kWords * 4, false);
    VmmBuffer              comp(kWords * 4, supported != 0);
    std::fprintf(stderr, "# compressible allocation granted: %d\n", comp.compressed);

    std::printf("memory,shape,padding,op,ms,gb_per_s,ok\n");
    for (uint32_t shape : {kRowsShape, kLanesShape})
        for (uint32_t pad : {0u, 4u, 8u, 16u, 24u, 32u}) {
            const u64 expect = expected_sum(shape, pad);
            run("cudaMalloc", plain.get(), shape, pad, expect);
            run("vmm", vmm.get(), shape, pad, expect);
            if (comp.compressed)
                run("vmm_compressible", comp.get(), shape, pad, expect);
        }
    return 0;
}
