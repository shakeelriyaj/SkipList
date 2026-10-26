#pragma once

// CpuOrderedSet: the trusted reference (oracle) for every index operation.
// A sorted, duplicate-free std::vector manipulated only with standard
// algorithms. It must be obviously correct, not fast; the optimized CPU
// baseline for benchmarks lives elsewhere.
//
// Contract (shared with the GPU implementations):
//   - Set semantics, ordering by unsigned integer '<'.
//   - build / bulk_insert / bulk_delete mutate *this; union_with, split and
//     range_query return new objects. Inputs are never modified.
//   - Duplicate keys in build/insert batches collapse; inserting an existing
//     key is a no-op. Deleting a missing key is ignored.
//   - split(p): left = {k < p}, right = {k >= p}.
//   - range_query / range_reduce use the half-open interval [lo, hi);
//     lo >= hi is an empty range.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "gpu_index/node_layout.hpp"
#include "gpu_index/reduce_ops.hpp"
#include "gpu_index/validation.hpp"

namespace gpu_index {

template <typename Key>
class CpuOrderedSet {
  static_assert(std::is_integral<Key>::value && std::is_unsigned<Key>::value,
                "v1 supports unsigned integer keys only");

 public:
  using key_type = Key;

  CpuOrderedSet() = default;
  explicit CpuOrderedSet(std::vector<Key> keys) { build(std::move(keys)); }

  void build(std::vector<Key> keys) { keys_ = normalize(std::move(keys)); }

  void bulk_insert(const std::vector<Key>& batch) {
    const std::vector<Key> incoming = normalize(batch);
    std::vector<Key> merged;
    merged.reserve(keys_.size() + incoming.size());
    std::set_union(keys_.begin(), keys_.end(), incoming.begin(), incoming.end(),
                   std::back_inserter(merged));
    keys_ = std::move(merged);
  }

  void bulk_delete(const std::vector<Key>& batch) {
    const std::vector<Key> removing = normalize(batch);
    std::vector<Key> kept;
    kept.reserve(keys_.size());
    std::set_difference(keys_.begin(), keys_.end(), removing.begin(), removing.end(),
                        std::back_inserter(kept));
    keys_ = std::move(kept);
  }

  CpuOrderedSet union_with(const CpuOrderedSet& other) const {
    CpuOrderedSet out;
    out.keys_.reserve(keys_.size() + other.keys_.size());
    std::set_union(keys_.begin(), keys_.end(), other.keys_.begin(), other.keys_.end(),
                   std::back_inserter(out.keys_));
    return out;
  }

  std::pair<CpuOrderedSet, CpuOrderedSet> split(Key pivot) const {
    const auto mid = std::lower_bound(keys_.begin(), keys_.end(), pivot);
    CpuOrderedSet left, right;
    left.keys_.assign(keys_.begin(), mid);
    right.keys_.assign(mid, keys_.end());
    return {std::move(left), std::move(right)};
  }

  std::vector<Key> range_query(Key lo, Key hi) const {
    const auto r = locate(lo, hi);
    return std::vector<Key>(r.first, r.second);
  }

  Key range_reduce(Key lo, Key hi, ReduceOp op) const {
    const auto r = locate(lo, hi);
    return std::accumulate(r.first, r.second, reduce_identity<Key>(op),
                           [op](Key acc, Key k) { return reduce_apply(op, acc, k); });
  }

  // One byte per query (1 = present) to mirror the device output buffer.
  std::vector<std::uint8_t> contains(const std::vector<Key>& queries) const {
    std::vector<std::uint8_t> out(queries.size());
    for (std::size_t i = 0; i < queries.size(); ++i) {
      out[i] = std::binary_search(keys_.begin(), keys_.end(), queries[i]) ? 1 : 0;
    }
    return out;
  }

  std::size_t size() const { return keys_.size(); }
  bool empty() const { return keys_.empty(); }
  const std::vector<Key>& keys() const { return keys_; }

  // The packed node layout a GPU index with this capacity must reproduce.
  HostNodeLayout<Key> to_layout(std::size_t node_capacity) const {
    return partition_into_nodes(keys_, node_capacity);
  }

  ValidationResult validate() const {
    for (std::size_t i = 1; i < keys_.size(); ++i) {
      if (keys_[i] == keys_[i - 1]) return ValidationResult::failure(ValidationError::DuplicateKey, i);
      if (keys_[i] < keys_[i - 1]) return ValidationResult::failure(ValidationError::KeysOutOfOrder, i);
    }
    return ValidationResult::success();
  }

 private:
  using const_iter = typename std::vector<Key>::const_iterator;

  static std::vector<Key> normalize(std::vector<Key> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
  }

  std::pair<const_iter, const_iter> locate(Key lo, Key hi) const {
    if (!(lo < hi)) return {keys_.end(), keys_.end()};
    return {std::lower_bound(keys_.begin(), keys_.end(), lo),
            std::lower_bound(keys_.begin(), keys_.end(), hi)};
  }

  std::vector<Key> keys_;
};

}  // namespace gpu_index
