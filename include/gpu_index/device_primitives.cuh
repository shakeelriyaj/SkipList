#pragma once

// Device building blocks shared by FlatSortedArray and GpuOrderedIndex, so
// the two implementations run identical pipelines and benchmarks measure
// only the representation difference.
//
// sort_unique_keys(): radix sort + adjacent-unique of an arbitrary device
// key array (readme sec. 5.1 steps 1-3). The input is never modified.
//
// merge_unique_keys(): set union of two sorted, duplicate-free device arrays
// (readme sec. 5.2 steps 3-4). Used by bulk_insert.
//
// fill_node_max(): routing-key kernel for the packed node layout
// (node_layout.hpp). Reused after every mutation in Phase 4.

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <thrust/set_operations.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/system_error.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "gpu_index/common.cuh"

namespace gpu_index {
namespace detail {

// CUB's sort/select entry points take `int` item counts.
inline int checked_cub_count(std::size_t n, const char* what) {
  if (n > static_cast<std::size_t>(INT_MAX)) {
    throw std::length_error(std::string(what) + ": input exceeds INT_MAX keys");
  }
  return static_cast<int>(n);
}

// Sorts `in` and removes duplicates. On return, the first `m` elements of
// `unique_out` hold the strictly increasing distinct keys, where `m` is the
// return value. unique_out is resized to in.size(); slots past m are
// unspecified. (Callers copy the first m keys into an exactly sized buffer.)
//
// Synchronizes `stream` once, to read back m: the caller needs it on the host
// to size the final allocation. All temporaries are RAII-owned and released
// on exception.
template <typename Key>
std::size_t sort_unique_keys(DeviceSpan<const Key> in, DeviceBuffer<Key>& unique_out,
                             cudaStream_t stream) {
  static_assert(std::is_integral<Key>::value && std::is_unsigned<Key>::value,
                "v1 supports unsigned integer keys only");
  const std::size_t n = in.size();
  if (n == 0) {
    unique_out.resize_discard(0);
    return 0;
  }
  const int count = checked_cub_count(n, "sort_unique_keys");
  constexpr int end_bit = static_cast<int>(sizeof(Key) * 8);

  DeviceBuffer<Key> sorted(n);
  unique_out.resize_discard(n);
  DeviceBuffer<int> num_unique(1);

  std::size_t sort_bytes = 0, unique_bytes = 0;
  GPU_INDEX_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, in.data(),
                                                      sorted.data(), count, 0, end_bit, stream));
  GPU_INDEX_CUDA_CHECK(cub::DeviceSelect::Unique(nullptr, unique_bytes, sorted.data(),
                                                 unique_out.data(), num_unique.data(), count,
                                                 stream));
  DeviceBuffer<unsigned char> temp(std::max<std::size_t>(1, std::max(sort_bytes, unique_bytes)));

  GPU_INDEX_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(temp.data(), sort_bytes, in.data(),
                                                      sorted.data(), count, 0, end_bit, stream));
  GPU_INDEX_CUDA_CHECK(cub::DeviceSelect::Unique(temp.data(), unique_bytes, sorted.data(),
                                                 unique_out.data(), num_unique.data(), count,
                                                 stream));

  int m = 0;
  GPU_INDEX_CUDA_CHECK(
      cudaMemcpyAsync(&m, num_unique.data(), sizeof(int), cudaMemcpyDeviceToHost, stream));
  GPU_INDEX_CUDA_CHECK(cudaStreamSynchronize(stream));
  if (m < 0 || static_cast<std::size_t>(m) > n) {
    throw std::runtime_error("sort_unique_keys: device returned an invalid unique count");
  }
  return static_cast<std::size_t>(m);
}

// Set union of `a` and `b`, which must each be strictly increasing (sorted,
// duplicate-free). On return, the first `m` elements of `out` hold the
// strictly increasing union, where `m` is the return value. out is resized
// to a.size() + b.size() (the worst case, no overlap); slots past m are
// unspecified. Neither input is modified. `out` must not alias either input.
//
// Because each input is duplicate-free on its own, every key present in both
// is emitted exactly once, so the result needs no separate unique pass.
//
// Runs on `stream` and synchronizes it at most once, because
// thrust::set_union returns the output end on the host (no sync if either
// input is empty). Thrust failures are rethrown as cuda_error so callers see
// one error type.
template <typename Key>
std::size_t merge_unique_keys(DeviceSpan<const Key> a, DeviceSpan<const Key> b,
                              DeviceBuffer<Key>& out, cudaStream_t stream) {
  const std::size_t total = a.size() + b.size();
  out.resize_discard(total);
  if (total == 0) return 0;

  // One side empty: the union is the other side. Skips the Thrust call (and
  // its sync and scratch allocation).
  if (a.empty() || b.empty()) {
    const DeviceSpan<const Key>& src = a.empty() ? b : a;
    GPU_INDEX_CUDA_CHECK(cudaMemcpyAsync(out.data(), src.data(), src.size() * sizeof(Key),
                                         cudaMemcpyDeviceToDevice, stream));
    return src.size();
  }

  Key* end = nullptr;
  try {
    end = thrust::set_union(thrust::cuda::par.on(stream), a.data(), a.data() + a.size(), b.data(),
                            b.data() + b.size(), out.data());
  } catch (const thrust::system_error& e) {
    (void)cudaGetLastError();  // clear non-sticky error state, as cuda_check does
    const cudaError_t code = e.code().category() == thrust::cuda_category()
                                 ? static_cast<cudaError_t>(e.code().value())
                                 : cudaErrorUnknown;
    throw cuda_error(code, "thrust::set_union", __FILE__, __LINE__);
  } catch (const std::bad_alloc&) {
    (void)cudaGetLastError();
    throw cuda_error(cudaErrorMemoryAllocation, "thrust::set_union (scratch allocation)",
                     __FILE__, __LINE__);
  }

  const std::ptrdiff_t m = end - out.data();
  if (m < 0 || static_cast<std::size_t>(m) > total) {
    throw std::runtime_error("merge_unique_keys: thrust returned an invalid output end");
  }
  return static_cast<std::size_t>(m);
}

// One thread per node: node_max[i] = last valid key of node i in the packed
// layout (node i covers slots [i*cap, min((i+1)*cap, n))).
template <typename Key>
__global__ void fill_node_max_kernel(const Key* __restrict__ keys, std::size_t n, std::size_t cap,
                                     std::size_t num_nodes, Key* __restrict__ node_max) {
  const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i >= num_nodes) return;
  const std::size_t end = (i + 1) * cap;
  node_max[i] = keys[(end < n ? end : n) - 1];
}

template <typename Key>
void fill_node_max(const Key* keys, std::size_t n, std::size_t cap, std::size_t num_nodes,
                   Key* node_max, cudaStream_t stream) {
  if (num_nodes == 0) return;
  constexpr unsigned block = 256;
  const std::size_t grid = (num_nodes + block - 1) / block;
  fill_node_max_kernel<Key><<<static_cast<unsigned>(grid), block, 0, stream>>>(
      keys, n, cap, num_nodes, node_max);
  GPU_INDEX_CHECK_LAUNCH();
}

}  // namespace detail
}  // namespace gpu_index
