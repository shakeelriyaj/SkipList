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
//
// bulk_insert contract (readme sec. 5.2):
//   - Batch is a device span (or a host vector, via bulk_insert_from_host).
//     It is never modified, may be unsorted, may contain duplicates, and may
//     alias this array's own keys.
//   - Result is the set union of the current keys and the batch. Keys already
//     present are no-ops. An empty batch, or a batch whose keys are all
//     already present, leaves the storage untouched (no reallocation).
//   - Full rebuild: the batch is sorted and deduplicated, then merged with
//     the existing keys into fresh storage. O(b log b + n + b) work.
//   - Synchronizes `stream` at most twice: once for the batch's unique count,
//     once for the merged size.
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
    assign_sorted(unique.data(), m, stream);
  }

  void build_from_host(const std::vector<Key>& input, cudaStream_t stream = 0) {
    DeviceBuffer<Key> staged;
    staged.copy_from_host(input, stream);
    build(staged.span(), stream);
  }

  void bulk_insert(DeviceSpan<const Key> batch, cudaStream_t stream = 0) {
    if (batch.empty()) return;
    DeviceBuffer<Key> incoming;
    const std::size_t b = detail::sort_unique_keys(batch, incoming, stream);
    DeviceBuffer<Key> merged;
    const std::size_t m = detail::merge_unique_keys<Key>(
        keys_.span(), DeviceSpan<const Key>(incoming.data(), b), merged, stream);
    if (m == keys_.size()) return;  // every batch key was already present
    assign_sorted(merged.data(), m, stream);
  }

  void bulk_insert_from_host(const std::vector<Key>& batch, cudaStream_t stream = 0) {
    if (batch.empty()) return;
    DeviceBuffer<Key> staged;
    staged.copy_from_host(batch, stream);
    bulk_insert(staged.span(), stream);
  }

  std::size_t size() const { return keys_.size(); }
  bool empty() const { return keys_.empty(); }
  DeviceSpan<const Key> keys() const { return keys_.span(); }

  std::vector<Key> to_host(cudaStream_t stream = 0) const { return keys_.to_host(stream); }

 private:
  // Replaces the contents with the first m keys at `src`, which must be
  // strictly increasing. Allocates exactly m slots and commits only after
  // every throwing call has succeeded.
  void assign_sorted(const Key* src, std::size_t m, cudaStream_t stream) {
    DeviceBuffer<Key> keys(m);
    if (m > 0) {
      GPU_INDEX_CUDA_CHECK(
          cudaMemcpyAsync(keys.data(), src, m * sizeof(Key), cudaMemcpyDeviceToDevice, stream));
    }
    // The copy is stream-ordered, so the caller's source buffer can be freed
    // once this returns: cudaFree waits for outstanding work that uses it.
    keys_.swap(keys);
  }

  DeviceBuffer<Key> keys_;
};

}  // namespace gpu_index
