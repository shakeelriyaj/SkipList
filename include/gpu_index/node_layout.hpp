#pragma once

// Host-side description of the multi-key node layout and its invariant
// checker. The GPU index copies its device state into a HostNodeLayout for
// validation until a device-side validator exists; the CPU reference uses it
// to produce the layout the GPU build must match.
//
// Layout (v1, packed): node i owns key slots [i * cap, (i + 1) * cap).
// Every node except the last is full; the last holds 1..cap keys. Slots past
// node_counts[i] are padding and must never be read as keys. node_max[i] is
// the routing key: the largest key in node i.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "gpu_index/validation.hpp"

namespace gpu_index {

template <typename Key>
struct HostNodeLayout {
  std::size_t node_capacity = 0;
  std::size_t logical_size = 0;
  std::vector<Key> keys;                  // num_nodes() * node_capacity slots
  std::vector<std::uint32_t> node_counts; // valid keys per node
  std::vector<Key> node_max;              // routing key per node

  std::size_t num_nodes() const { return node_counts.size(); }
};

inline std::size_t num_nodes_for(std::size_t num_keys, std::size_t capacity) {
  return capacity == 0 ? 0 : (num_keys + capacity - 1) / capacity;
}

// Packs strictly increasing keys into fixed-capacity nodes.
template <typename Key>
HostNodeLayout<Key> partition_into_nodes(const Key* sorted_keys, std::size_t n,
                                         std::size_t capacity) {
  if (capacity == 0 || capacity > UINT32_MAX) {
    throw std::invalid_argument("partition_into_nodes: capacity must be in [1, 2^32)");
  }
  HostNodeLayout<Key> layout;
  layout.node_capacity = capacity;
  layout.logical_size = n;
  const std::size_t nodes = num_nodes_for(n, capacity);
  layout.keys.assign(nodes * capacity, Key{});
  layout.node_counts.resize(nodes);
  layout.node_max.resize(nodes);
  for (std::size_t i = 0; i < n; ++i) layout.keys[i] = sorted_keys[i];
  for (std::size_t node = 0; node < nodes; ++node) {
    const std::size_t begin = node * capacity;
    const std::size_t count = (begin + capacity <= n) ? capacity : n - begin;
    layout.node_counts[node] = static_cast<std::uint32_t>(count);
    layout.node_max[node] = sorted_keys[begin + count - 1];
  }
  return layout;
}

template <typename Key>
HostNodeLayout<Key> partition_into_nodes(const std::vector<Key>& sorted_keys,
                                         std::size_t capacity) {
  return partition_into_nodes(sorted_keys.data(), sorted_keys.size(), capacity);
}

// Checks the invariants from readme sec. 8.2. With require_packed = false,
// partially filled interior nodes are allowed (reserved for a later layout
// with slack); empty nodes are always rejected. Returns the first violation.
template <typename Key>
ValidationResult validate_layout(const HostNodeLayout<Key>& layout, bool require_packed = true) {
  using VE = ValidationError;
  const std::size_t cap = layout.node_capacity;
  const std::size_t nodes = layout.num_nodes();

  if (cap == 0) return ValidationResult::failure(VE::InvalidCapacity, 0);
  if (layout.node_max.size() != nodes) {
    return ValidationResult::failure(VE::NodeCountMismatch, layout.node_max.size());
  }
  if (layout.keys.size() / cap < nodes) {
    return ValidationResult::failure(VE::BufferTooSmall, layout.keys.size());
  }

  std::size_t total = 0;
  for (std::size_t node = 0; node < nodes; ++node) {
    const std::size_t count = layout.node_counts[node];
    const std::size_t base = node * cap;
    if (count == 0) return ValidationResult::failure(VE::EmptyNode, node);
    if (count > cap) return ValidationResult::failure(VE::NodeOverCapacity, node);
    if (require_packed && node + 1 < nodes && count != cap) {
      return ValidationResult::failure(VE::NodeNotFull, node);
    }
    for (std::size_t j = 1; j < count; ++j) {
      const Key prev = layout.keys[base + j - 1];
      const Key cur = layout.keys[base + j];
      if (cur == prev) return ValidationResult::failure(VE::DuplicateKey, base + j);
      if (cur < prev) return ValidationResult::failure(VE::KeysOutOfOrder, base + j);
    }
    if (layout.node_max[node] != layout.keys[base + count - 1]) {
      return ValidationResult::failure(VE::NodeMaxMismatch, node);
    }
    if (node > 0) {
      const Key prev_max = layout.keys[(node - 1) * cap + layout.node_counts[node - 1] - 1];
      const Key first = layout.keys[base];
      if (first == prev_max) return ValidationResult::failure(VE::DuplicateKey, base);
      if (first < prev_max) return ValidationResult::failure(VE::NodeOrderViolated, node);
    }
    total += count;
  }
  if (total != layout.logical_size) return ValidationResult::failure(VE::SizeMismatch, total);
  return ValidationResult::success();
}

// Logical key sequence of a layout (padding removed).
template <typename Key>
std::vector<Key> flatten(const HostNodeLayout<Key>& layout) {
  std::vector<Key> out;
  out.reserve(layout.logical_size);
  for (std::size_t node = 0; node < layout.num_nodes(); ++node) {
    const std::size_t base = node * layout.node_capacity;
    for (std::size_t j = 0; j < layout.node_counts[node]; ++j) out.push_back(layout.keys[base + j]);
  }
  return out;
}

}  // namespace gpu_index
