// Phase 1 toolchain check: prints the device, radix-sorts 1M random keys with
// CUB, and compares against std::sort. Exit code 0 on success.

#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>

#include "gpu_index/common.cuh"
#include "random_gen.hpp"

using gpu_index::DeviceBuffer;

int main() {
  try {
    int device = 0;
    cudaDeviceProp prop{};
    GPU_INDEX_CUDA_CHECK(cudaGetDevice(&device));
    GPU_INDEX_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int runtime = 0, driver = 0;
    GPU_INDEX_CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    GPU_INDEX_CUDA_CHECK(cudaDriverGetVersion(&driver));
    std::printf("device %d: %s, cc %d.%d, %.1f GiB, %d SMs\n", device, prop.name, prop.major,
                prop.minor, prop.totalGlobalMem / double(1 << 30), prop.multiProcessorCount);
    std::printf("cuda runtime %d, driver %d, cub %d\n", runtime, driver, CUB_VERSION);

    const std::size_t n = 1 << 20;
    std::vector<std::uint32_t> host =
        gen::generate_keys<std::uint32_t>(n, gen::Distribution::Uniform, 42);

    cudaStream_t stream;
    GPU_INDEX_CUDA_CHECK(cudaStreamCreate(&stream));

    DeviceBuffer<std::uint32_t> in, out(n);
    in.copy_from_host(host, stream);

    std::size_t temp_bytes = 0;
    GPU_INDEX_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, in.data(), out.data(),
                                                        static_cast<int>(n), 0, 32, stream));
    DeviceBuffer<unsigned char> temp(temp_bytes);

    cudaEvent_t start, stop;
    GPU_INDEX_CUDA_CHECK(cudaEventCreate(&start));
    GPU_INDEX_CUDA_CHECK(cudaEventCreate(&stop));
    GPU_INDEX_CUDA_CHECK(cudaEventRecord(start, stream));
    GPU_INDEX_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(temp.data(), temp_bytes, in.data(),
                                                        out.data(), static_cast<int>(n), 0, 32,
                                                        stream));
    GPU_INDEX_CUDA_CHECK(cudaEventRecord(stop, stream));
    std::vector<std::uint32_t> gpu_sorted = out.to_host(stream);
    float gpu_ms = 0;
    GPU_INDEX_CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    auto t0 = std::chrono::steady_clock::now();
    std::sort(host.begin(), host.end());
    auto t1 = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    GPU_INDEX_CUDA_CHECK(cudaEventDestroy(start));
    GPU_INDEX_CUDA_CHECK(cudaEventDestroy(stop));
    GPU_INDEX_CUDA_CHECK(cudaStreamDestroy(stream));

    const bool ok = gpu_sorted == host;
    // Single cold run: a toolchain sanity check, not a benchmark.
    std::printf("sorted %zu keys: %s (gpu kernel %.3f ms, std::sort %.3f ms, single cold run)\n",
                n, ok ? "OK" : "MISMATCH", gpu_ms, cpu_ms);
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "smoke failed: %s\n", e.what());
    return 1;
  }
}
