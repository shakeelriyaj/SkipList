#pragma once

// GpuOrderedIndex<Key, NodeCap>: the multi-key node index (readme sec. 3)
// in the packed v1 layout described in node_layout.hpp:
//
//   keys_      num_nodes * NodeCap slots; node i owns [i*NodeCap, (i+1)*NodeCap)
//   node_max_  num_nodes routing keys (largest key of each node)
//   size_      logical number of keys
//
// Every node except the last is full, so node counts are implied by size_
// and are not stored on the device.
//
// build contract (readme sec. 5.1):
//   - Input is a device span (or a host vector, via build_from_host). It is
//     never modified and may be released as soon as build returns.
//   - Replaces the current contents. Unsorted input and duplicates are fine;
//     duplicates collapse (set semantics). Empty input gives zero nodes and
//     no allocation.
//   - Storage is sized from the unique count: ceil(m / NodeCap) nodes.
//   - Synchronizes `stream` once to learn the unique count; everything else
//     is stream-ordered.
//   - Strong exception guarantee: on failure, *this is unchanged.
//
// Padding slots are zero-filled so host copies are deterministic, but the
// contract still treats them as unspecified.
//
// bulk_insert contract (readme sec. 5.2):
//   - Batch is a device span (or a host vector, via bulk_insert_from_host).
//     It is never modified, may be unsorted, may contain duplicates, and may
//     alias this index's own key slots.
//   - Result is the set union of the current keys and the batch. Keys already
//     present are no-ops. An empty batch, or a batch whose keys are all
//     already present, leaves the storage untouched (no reallocation).
//   - Full rebuild: in the packed layout one new key shifts every later key,
//     so the batch is sorted and deduplicated, merged with the existing
//     logical keys (padding is never read), then re-laid-out into
//     ceil(m / NodeCap) nodes with node_max recomputed. Same pipeline as
//     FlatSortedArray::bulk_insert plus the layout step.
//   - Synchronizes `stream` at most twice: once for the batch's unique count,
//     once for the merged size.
//   - Strong exception guarantee: on failure, *this is unchanged.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gpu_index/common.cuh"
#include "gpu_index/device_primitives.cuh"
#include "gpu_index/device_validation.cuh"
#include "gpu_index/node_layout.hpp"
#include "gpu_index/validation.hpp"

namespace gpu_index {

template <typename Key, std::size_t NodeCap = 256>
class GpuOrderedIndex {
  static_assert(NodeCap > 0 && NodeCap <= UINT32_MAX, "NodeCap must be in [1, 2^32)");

 public:
  using key_type = Key;
  static constexpr std::size_t node_capacity = NodeCap;

  GpuOrderedIndex() = default;
  GpuOrderedIndex(GpuOrderedIndex&&) noexcept = default;
  GpuOrderedIndex& operator=(GpuOrderedIndex&&) noexcept = default;

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
    // Only the first size_ slots are keys; the rest of the last node is padding.
    const DeviceSpan<const Key> current(keys_.data(), size_);
    DeviceBuffer<Key> merged;
    const std::size_t m = detail::merge_unique_keys<Key>(
        current, DeviceSpan<const Key>(incoming.data(), b), merged, stream);
    if (m == size_) return;  // every batch key was already present
    assign_sorted(merged.data(), m, stream);
  }

  void bulk_insert_from_host(const std::vector<Key>& batch, cudaStream_t stream = 0) {
    if (batch.empty()) return;
    DeviceBuffer<Key> staged;
    staged.copy_from_host(batch, stream);
    bulk_insert(staged.span(), stream);
  }

  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  std::size_t num_nodes() const { return node_max_.size(); }

  DeviceSpan<const Key> key_slots() const { return keys_.span(); }
  DeviceSpan<const Key> node_max() const { return node_max_.span(); }

  // Copies the device state into the host layout used by validate_layout and
  // by the CPU reference (CpuOrderedSet::to_layout). Synchronizes `stream`.
  HostNodeLayout<Key> to_host_layout(cudaStream_t stream = 0) const {
    HostNodeLayout<Key> layout;
    layout.node_capacity = NodeCap;
    layout.logical_size = size_;
    layout.keys = keys_.to_host(stream);
    layout.node_max = node_max_.to_host(stream);
    const std::size_t nodes = node_max_.size();
    layout.node_counts.resize(nodes);
    for (std::size_t i = 0; i < nodes; ++i) {
      const std::size_t begin = i * NodeCap;
      layout.node_counts[i] =
          static_cast<std::uint32_t>(begin + NodeCap <= size_ ? NodeCap : size_ - begin);
    }
    return layout;
  }

  // Logical key sequence (padding removed). Synchronizes `stream`.
  std::vector<Key> to_host(cudaStream_t stream = 0) const {
    return flatten(to_host_layout(stream));
  }

  // Checks the readme sec. 8.2 invariants on the device (device_validation.cuh).
  // Returns the first violation in validate_layout order. Synchronizes
  // `stream` once; the only device-to-host traffic is one 8-byte result.
  ValidationResult validate(cudaStream_t stream = 0) const {
    return detail::validate_packed_layout_device<Key>(keys_.span(), node_max_.span(), size_,
                                                      NodeCap, stream);
  }

  // Reference path: copies the whole index to the host and runs
  // validate_layout. Slow; meant for tests and debugging.
  ValidationResult validate_on_host(cudaStream_t stream = 0) const {
    using VE = ValidationError;
    const std::size_t nodes = node_max_.size();
    if (nodes != num_nodes_for(size_, NodeCap)) {
      return ValidationResult::failure(VE::NodeCountMismatch, nodes);
    }
    if (keys_.size() / NodeCap < nodes) {
      return ValidationResult::failure(VE::BufferTooSmall, keys_.size());
    }
    return validate_layout(to_host_layout(stream), /*require_packed=*/true);
  }

 private:
  // Replaces the contents with the first m keys at `src`, which must be
  // strictly increasing: lays them out as ceil(m / NodeCap) packed nodes,
  // zero-fills the padding, and recomputes node_max. Commits only after every
  // throwing call has succeeded. `src` may be freed once this returns (the
  // copy is stream-ordered and cudaFree waits for it).
  void assign_sorted(const Key* src, std::size_t m, cudaStream_t stream) {
    const std::size_t nodes = num_nodes_for(m, NodeCap);
    const std::size_t slots = nodes * NodeCap;

    DeviceBuffer<Key> keys(slots);
    DeviceBuffer<Key> node_max(nodes);
    if (m > 0) {
      GPU_INDEX_CUDA_CHECK(
          cudaMemcpyAsync(keys.data(), src, m * sizeof(Key), cudaMemcpyDeviceToDevice, stream));
      if (slots > m) {
        GPU_INDEX_CUDA_CHECK(
            cudaMemsetAsync(keys.data() + m, 0, (slots - m) * sizeof(Key), stream));
      }
      detail::fill_node_max(keys.data(), m, NodeCap, nodes, node_max.data(), stream);
    }

    // Commit only after every call that can throw has succeeded.
    keys_.swap(keys);
    node_max_.swap(node_max);
    size_ = m;
  }

  DeviceBuffer<Key> keys_;
  DeviceBuffer<Key> node_max_;
  std::size_t size_ = 0;
};

}  // namespace gpu_index
