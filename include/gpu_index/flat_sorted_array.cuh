#pragma once

// FlatSortedArray<Key>: the GPU baseline from readme sec. 9.1 item 3. A
// single sorted, duplicate-free device array with no node structure or
// routing layer. It runs the same pipeline as GpuOrderedIndex so the
// benchmark comparison isolates the representation.
//
// build contract:
//   - Input is a device span (or a host vector, via build_from_host). It is
//     never modified and may be released as soon as build returns.
//   - Replaces the current contents. Unsorted input and duplicates are fine;
//     duplicates collapse (set semantics).
//   - Synchronizes `stream` once to learn the unique count.
//   - Strong exception guarantee: on failure, *this is unchanged.

#include <cstddef>
#include <vector>

#include "gpu_index/common.cuh"
#include "gpu_index/device_primitives.cuh"

namespace gpu_index {

template <typename Key>
class FlatSortedArray {
 public:
  using key_type = Key;

  FlatSortedArray() = default;
  FlatSortedArray(FlatSortedArray&&) noexcept = default;
  FlatSortedArray& operator=(FlatSortedArray&&) noexcept = default;

  void build(DeviceSpan<const Key> input, cudaStream_t stream = 0) {
    DeviceBuffer<Key> unique;
    const std::size_t m = detail::sort_unique_keys(input, unique, stream);
    DeviceBuffer<Key> keys(m);
    if (m > 0) {
      GPU_INDEX_CUDA_CHECK(cudaMemcpyAsync(keys.data(), unique.data(), m * sizeof(Key),
                                           cudaMemcpyDeviceToDevice, stream));
    }
    // The copy is stream-ordered, so `unique` can be freed now: cudaFree
    // waits for outstanding work that uses the allocation.
    keys_.swap(keys);
  }

  void build_from_host(const std::vector<Key>& input, cudaStream_t stream = 0) {
    DeviceBuffer<Key> staged;
    staged.copy_from_host(input, stream);
    build(staged.span(), stream);
  }

  std::size_t size() const { return keys_.size(); }
  bool empty() const { return keys_.empty(); }
  DeviceSpan<const Key> keys() const { return keys_.span(); }

  std::vector<Key> to_host(cudaStream_t stream = 0) const { return keys_.to_host(stream); }

 private:
  DeviceBuffer<Key> keys_;
};

}  // namespace gpu_index
