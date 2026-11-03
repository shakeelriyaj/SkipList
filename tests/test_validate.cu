// Device validator tests. The host validate_layout() is the oracle: for any
// packed layout, valid or corrupted, the device validator must return the
// same first violation (error code and index).

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "check.hpp"
#include "gpu_index/common.cuh"
#include "gpu_index/cpu_ordered_set.hpp"
#include "gpu_index/device_validation.cuh"
#include "gpu_index/gpu_ordered_index.cuh"
#include "gpu_index/node_layout.hpp"
#include "random_gen.hpp"

using gpu_index::CpuOrderedSet;
using gpu_index::DeviceBuffer;
using gpu_index::DeviceSpan;
using gpu_index::GpuOrderedIndex;
using gpu_index::HostNodeLayout;
using gpu_index::ValidationError;
using gpu_index::ValidationResult;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using VE = ValidationError;

namespace {

std::string describe(const ValidationResult& r) {
  return std::string(gpu_index::to_string(r.error)) + "@" + std::to_string(r.index);
}

template <typename Key>
ValidationResult validate_on_device(const HostNodeLayout<Key>& layout) {
  DeviceBuffer<Key> keys, node_max;
  keys.copy_from_host(layout.keys);
  node_max.copy_from_host(layout.node_max);
  return gpu_index::detail::validate_packed_layout_device<Key>(
      keys.span(), node_max.span(), layout.logical_size, layout.node_capacity, 0);
}

// Device and host must agree exactly on `layout`.
template <typename Key>
void check_agrees(const HostNodeLayout<Key>& layout, const std::string& label) {
  const ValidationResult host = gpu_index::validate_layout(layout, /*require_packed=*/true);
  const ValidationResult dev = validate_on_device(layout);
  if (host.error != dev.error || host.index != dev.index) {
    check::report(__FILE__, __LINE__,
                  label + ": device " + describe(dev) + " vs host " + describe(host));
  }
}

template <typename Key>
HostNodeLayout<Key> layout_of(std::vector<Key> keys, std::size_t cap) {
  return CpuOrderedSet<Key>(std::move(keys)).to_layout(cap);
}

std::vector<u32> iota_keys(u32 n, u32 step = 10) {
  std::vector<u32> v(n);
  for (u32 i = 0; i < n; ++i) v[i] = (i + 1) * step;
  return v;
}

// Applies one random corruption to the logical keys or the routing layer.
template <typename Key>
std::string corrupt_once(HostNodeLayout<Key>& l, std::mt19937_64& rng) {
  const std::size_t n = l.logical_size;
  std::uniform_int_distribution<std::size_t> pos(0, n - 1);
  const std::size_t p = pos(rng);
  switch (rng() % 6) {
    case 0:  // swap with the next key (may cross a node boundary)
      if (p + 1 < n) std::swap(l.keys[p], l.keys[p + 1]);
      return "swap@" + std::to_string(p);
    case 1:  // duplicate the previous key
      if (p > 0) l.keys[p] = l.keys[p - 1];
      return "dup@" + std::to_string(p);
    case 2:  // arbitrary value
      l.keys[p] = static_cast<Key>(rng());
      return "rand@" + std::to_string(p);
    case 3: {  // stale routing key
      const std::size_t node = p / l.node_capacity;
      l.node_max[node] = static_cast<Key>(l.node_max[node] + 1);
      return "node_max@" + std::to_string(node);
    }
    case 4:  // first key of a node not greater than the previous node's max
      if (p >= l.node_capacity) {
        const std::size_t first = (p / l.node_capacity) * l.node_capacity;
        const Key prev_max = l.keys[first - 1];
        // Equal (cross-node duplicate) or smaller (cross-node order violation).
        l.keys[first] = (rng() % 2 || prev_max == 0) ? prev_max : static_cast<Key>(prev_max - 1);
      }
      return "boundary@" + std::to_string(p);
    default:  // extreme values
      l.keys[p] = (rng() % 2) ? Key{0} : static_cast<Key>(~Key{0});
      return "extreme@" + std::to_string(p);
  }
}

}  // namespace

// ---------------------------------------------------------------- valid layouts

TEST_CASE(validate_empty_index) {
  GpuOrderedIndex<u32, 4> index;
  CHECK(index.validate().ok());
  index.build_from_host({});
  CHECK(index.validate().ok());
}

TEST_CASE(validate_built_indexes_pass) {
  const gen::Distribution dists[] = {gen::Distribution::Uniform,
                                     gen::Distribution::DuplicateHeavy,
                                     gen::Distribution::Clustered, gen::Distribution::Skewed};
  for (gen::Distribution dist : dists) {
    for (std::size_t n : {1u, 3u, 4u, 5u, 1000u, 65537u}) {
      const auto k32 = gen::generate_keys<u32>(n, dist, n);
      const auto k64 = gen::generate_keys<u64>(n, dist, n);
      GpuOrderedIndex<u32, 1> a;
      GpuOrderedIndex<u32, 4> b;
      GpuOrderedIndex<u64, 256> c;
      a.build_from_host(k32);
      b.build_from_host(k32);
      c.build_from_host(k64);
      CHECK(a.validate().ok());
      CHECK(b.validate().ok());
      CHECK(c.validate().ok());
      CHECK(c.validate_on_host().ok());
    }
  }
}

// ---------------------------------------------------------------- hand cases
// Layout for iota_keys(10), cap 4: [10 20 30 40 | 50 60 70 80 | 90 100 -- --]

TEST_CASE(validate_detects_intra_node_order) {
  auto l = layout_of(iota_keys(10), 4);
  std::swap(l.keys[5], l.keys[6]);  // 70 60 inside node 1; node_max still 80
  CHECK(l.node_max[1] == 80u);
  const auto r = validate_on_device(l);
  CHECK(r.error == VE::KeysOutOfOrder);
  CHECK_EQ(r.index, 6u);
  check_agrees(l, "intra order");
}

TEST_CASE(validate_detects_intra_node_duplicate) {
  auto l = layout_of(iota_keys(10), 4);
  l.keys[2] = l.keys[1];
  const auto r = validate_on_device(l);
  CHECK(r.error == VE::DuplicateKey);
  CHECK_EQ(r.index, 2u);
  check_agrees(l, "intra dup");
}

TEST_CASE(validate_detects_cross_node_order) {
  auto l = layout_of(iota_keys(10), 4);
  l.keys[4] = 35;  // node 1 starts below node 0's max (40), still < 60
  const auto r = validate_on_device(l);
  CHECK(r.error == VE::NodeOrderViolated);
  CHECK_EQ(r.index, 1u);
  check_agrees(l, "cross order");
}

TEST_CASE(validate_detects_cross_node_duplicate) {
  auto l = layout_of(iota_keys(10), 4);
  l.keys[8] = 80;  // first key of node 2 equals node 1's max
  const auto r = validate_on_device(l);
  CHECK(r.error == VE::DuplicateKey);
  CHECK_EQ(r.index, 8u);
  check_agrees(l, "cross dup");
}

TEST_CASE(validate_detects_node_max_mismatch) {
  auto l = layout_of(iota_keys(10), 4);
  l.node_max[2] = 99;  // last node (partial): real max is 100
  const auto r = validate_on_device(l);
  CHECK(r.error == VE::NodeMaxMismatch);
  CHECK_EQ(r.index, 2u);
  check_agrees(l, "node_max");
}

TEST_CASE(validate_reports_first_violation_in_host_order) {
  // Errors in node 1 (intra order at slot 7, node_max) and node 2 (dup):
  // node 1's intra-node error must win.
  auto l = layout_of(iota_keys(12), 4);
  l.keys[9] = l.keys[8];  // node 2 intra dup
  l.node_max[1] = 1;      // node 1 node_max
  l.keys[7] = 65;         // node 1 intra order (slot 7) -> reported before node_max
  const auto r = validate_on_device(l);
  CHECK(r.error == VE::KeysOutOfOrder);
  CHECK_EQ(r.index, 7u);
  check_agrees(l, "first violation");

  // Within one node: intra error beats node_max, node_max beats the boundary.
  auto m = layout_of(iota_keys(12), 4);
  m.keys[4] = 5;       // boundary violation for node 1 (and intra ok: 5 < 60)
  m.node_max[1] = 1;   // node_max violation for node 1
  check_agrees(m, "node_max before boundary");
  CHECK(validate_on_device(m).error == VE::NodeMaxMismatch);
}

TEST_CASE(validate_bookkeeping_errors) {
  auto l = layout_of(iota_keys(10), 4);
  DeviceBuffer<u32> keys, node_max;
  keys.copy_from_host(l.keys);
  node_max.copy_from_host(l.node_max);
  using gpu_index::detail::validate_packed_layout_device;

  // node_max shorter than the node count implied by the logical size
  auto r = validate_packed_layout_device<u32>(keys.span(), DeviceSpan<const u32>(node_max.data(), 2),
                                              10, 4, 0);
  CHECK(r.error == VE::NodeCountMismatch);
  CHECK_EQ(r.index, 2u);

  // key storage smaller than num_nodes * cap
  r = validate_packed_layout_device<u32>(DeviceSpan<const u32>(keys.data(), 11), node_max.span(), 10,
                                         4, 0);
  CHECK(r.error == VE::BufferTooSmall);
  CHECK_EQ(r.index, 11u);

  r = validate_packed_layout_device<u32>(keys.span(), node_max.span(), 10, 0, 0);
  CHECK(r.error == VE::InvalidCapacity);
}

// ---------------------------------------------------------------- differential

template <typename Key, std::size_t Cap>
void randomized_differential(std::uint64_t seeds) {
  for (std::uint64_t seed = 1; seed <= seeds; ++seed) {
    std::mt19937_64 rng(seed * 7919 + Cap);
    const std::size_t n = 1 + rng() % 3000;
    auto base = layout_of(gen::generate_keys<Key>(n, gen::Distribution::Uniform, seed), Cap);
    if (base.logical_size == 0) continue;
    check_agrees(base, "valid seed=" + std::to_string(seed));

    auto l = base;
    std::string what;
    const int corruptions = 1 + static_cast<int>(rng() % 4);
    for (int c = 0; c < corruptions; ++c) what += corrupt_once(l, rng) + " ";
    check_agrees(l, "cap=" + std::to_string(Cap) + " seed=" + std::to_string(seed) + " " + what);
  }
}

TEST_CASE(validate_matches_host_on_random_corruptions) {
  randomized_differential<u32, 1>(300);
  randomized_differential<u32, 4>(300);
  randomized_differential<u32, 64>(300);
  randomized_differential<u64, 256>(300);
}

TEST_CASE(validate_large_index) {
  const std::size_t n = 10'000'000;
  auto l = layout_of(gen::generate_keys<u32>(n, gen::Distribution::Uniform, 5), 256);
  check_agrees(l, "10M valid");
  const std::size_t last = l.logical_size - 1;
  l.keys[last] = l.keys[last - 1];  // single defect at the very end
  const auto r = validate_on_device(l);
  CHECK(r.error == VE::DuplicateKey);
  CHECK_EQ(r.index, last);
  check_agrees(l, "10M dup at end");
}

CHECK_MAIN()
