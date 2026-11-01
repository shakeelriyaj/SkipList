// Phase 3 build tests: GpuOrderedIndex and FlatSortedArray must reproduce
// the CPU reference exactly (same logical keys, same packed node layout) and
// pass validate() for every edge case in readme sec. 8.1 plus seeded random
// inputs, for both uint32_t and uint64_t keys.

#include <cstdint>
#include <string>
#include <vector>

#include "check.hpp"
#include "gpu_index/common.cuh"
#include "gpu_index/cpu_ordered_set.hpp"
#include "gpu_index/flat_sorted_array.cuh"
#include "gpu_index/gpu_ordered_index.cuh"
#include "random_gen.hpp"

using gpu_index::CpuOrderedSet;
using gpu_index::DeviceBuffer;
using gpu_index::FlatSortedArray;
using gpu_index::GpuOrderedIndex;
using gpu_index::HostNodeLayout;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

namespace {

template <typename Key>
bool same_layout(const HostNodeLayout<Key>& a, const HostNodeLayout<Key>& b) {
  // Padding is zero-filled on the GPU and Key{} in partition_into_nodes, so
  // the full slot arrays are comparable.
  return a.node_capacity == b.node_capacity && a.logical_size == b.logical_size &&
         a.keys == b.keys && a.node_counts == b.node_counts && a.node_max == b.node_max;
}

// Builds both GPU structures from `input` and checks them against the oracle.
template <typename Key, std::size_t Cap>
void check_build(const std::vector<Key>& input, const std::string& label) {
  const CpuOrderedSet<Key> ref(input);

  GpuOrderedIndex<Key, Cap> index;
  index.build_from_host(input);
  const auto v = index.validate();
  if (!v.ok()) {
    check::report(__FILE__, __LINE__,
                  label + ": validate failed: " + gpu_index::to_string(v.error) + " at " +
                      std::to_string(v.index));
  }
  CHECK_EQ(index.size(), ref.size());
  CHECK_EQ(index.num_nodes(), gpu_index::num_nodes_for(ref.size(), Cap));
  if (!same_layout(index.to_host_layout(), ref.to_layout(Cap))) {
    check::report(__FILE__, __LINE__, label + ": node layout differs from CPU reference");
  }

  FlatSortedArray<Key> flat;
  flat.build_from_host(input);
  CHECK_EQ(flat.size(), ref.size());
  if (flat.to_host() != ref.keys()) {
    check::report(__FILE__, __LINE__, label + ": flat array differs from CPU reference");
  }
}

template <typename Key>
void check_all_caps(const std::vector<Key>& input, const std::string& label) {
  check_build<Key, 1>(input, label + " cap=1");
  check_build<Key, 4>(input, label + " cap=4");
  check_build<Key, 256>(input, label + " cap=256");
}

}  // namespace

// ---------------------------------------------------------------- edge cases

TEST_CASE(build_empty) {
  check_all_caps<u32>({}, "empty u32");
  check_all_caps<u64>({}, "empty u64");
  GpuOrderedIndex<u32, 4> index;
  index.build_from_host({});
  CHECK_EQ(index.size(), 0u);
  CHECK_EQ(index.num_nodes(), 0u);
  CHECK(index.key_slots().empty());
  CHECK(index.validate().ok());
}

TEST_CASE(build_single) {
  check_all_caps<u32>({42}, "single u32");
  check_all_caps<u64>({42}, "single u64");
}

TEST_CASE(build_sorted_and_unsorted_agree) {
  std::vector<u32> sorted, reversed;
  for (u32 i = 0; i < 1000; ++i) sorted.push_back(i * 3);
  reversed.assign(sorted.rbegin(), sorted.rend());
  check_all_caps(sorted, "sorted");
  check_all_caps(reversed, "reversed");

  GpuOrderedIndex<u32, 4> a, b;
  a.build_from_host(sorted);
  b.build_from_host(reversed);
  CHECK(same_layout(a.to_host_layout(), b.to_host_layout()));
}

TEST_CASE(build_all_duplicates) {
  check_all_caps(std::vector<u32>(10000, 7u), "all-equal u32");
  check_all_caps(std::vector<u64>(10000, 7u), "all-equal u64");
}

TEST_CASE(build_node_boundaries) {
  // Exactly full, one short, one over, and two full nodes for cap = 4.
  for (u32 n : {3u, 4u, 5u, 8u, 9u}) {
    std::vector<u32> keys;
    for (u32 i = 0; i < n; ++i) keys.push_back(n - i);  // unsorted
    check_build<u32, 4>(keys, "boundary n=" + std::to_string(n));
  }
  // Same for cap = 256 around 256 and 512.
  for (u32 n : {255u, 256u, 257u, 512u, 513u}) {
    std::vector<u32> keys;
    for (u32 i = 0; i < n; ++i) keys.push_back((i * 2654435761u) ^ 0x5bd1e995u);
    check_build<u32, 256>(keys, "boundary n=" + std::to_string(n));
  }
}

TEST_CASE(build_extreme_keys) {
  // Radix sort must handle the top bit and the full key range.
  check_all_caps<u32>({UINT32_MAX, 0u, 1u << 31, UINT32_MAX, 0u, (1u << 31) - 1}, "extreme u32");
  check_all_caps<u64>({UINT64_MAX, 0u, u64(1) << 63, u64(1) << 32, UINT64_MAX, (u64(1) << 32) - 1},
                      "extreme u64");
}

TEST_CASE(build_replaces_previous_contents) {
  GpuOrderedIndex<u32, 4> index;
  index.build_from_host({1, 2, 3, 4, 5, 6, 7, 8, 9});
  index.build_from_host({100, 50});
  CHECK(index.to_host() == (std::vector<u32>{50, 100}));
  CHECK_EQ(index.num_nodes(), 1u);
  CHECK(index.validate().ok());
  index.build_from_host({});
  CHECK_EQ(index.size(), 0u);
  CHECK_EQ(index.num_nodes(), 0u);
}

TEST_CASE(build_from_device_span_leaves_input_untouched) {
  const std::vector<u32> input = {9, 3, 3, 7, 1, 9, 0};
  DeviceBuffer<u32> d;
  d.copy_from_host(input);

  GpuOrderedIndex<u32, 4> index;
  index.build(d.span());
  FlatSortedArray<u32> flat;
  flat.build(d.span());

  CHECK(d.to_host() == input);
  CHECK(index.to_host() == (std::vector<u32>{0, 1, 3, 7, 9}));
  CHECK(flat.to_host() == (std::vector<u32>{0, 1, 3, 7, 9}));
}

TEST_CASE(build_on_nondefault_stream) {
  cudaStream_t stream;
  GPU_INDEX_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const auto keys = gen::generate_keys<u32>(100000, gen::Distribution::Uniform, 7);
  GpuOrderedIndex<u32, 256> index;
  index.build_from_host(keys, stream);
  CHECK(index.validate(stream).ok());
  CHECK(index.to_host(stream) == CpuOrderedSet<u32>(keys).keys());
  GPU_INDEX_CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE(build_move_transfers_ownership) {
  GpuOrderedIndex<u32, 4> a;
  a.build_from_host({5, 1, 3});
  GpuOrderedIndex<u32, 4> b(std::move(a));
  CHECK(b.to_host() == (std::vector<u32>{1, 3, 5}));
  CHECK(b.validate().ok());
}

// ---------------------------------------------------------------- randomized

TEST_CASE(build_randomized) {
  const gen::Distribution dists[] = {gen::Distribution::Uniform,
                                     gen::Distribution::DuplicateHeavy,
                                     gen::Distribution::Clustered, gen::Distribution::Skewed};
  const std::size_t sizes[] = {1, 17, 1000, 65537, 300000};
  for (gen::Distribution dist : dists) {
    for (std::size_t n : sizes) {
      for (std::uint64_t seed = 1; seed <= 3; ++seed) {
        const std::string label = std::string(gen::to_string(dist)) + " n=" + std::to_string(n) +
                                  " seed=" + std::to_string(seed);
        check_build<u32, 64>(gen::generate_keys<u32>(n, dist, seed), label + " u32");
        check_build<u64, 256>(gen::generate_keys<u64>(n, dist, seed), label + " u64");
      }
    }
  }
}

TEST_CASE(build_large) {
  const std::size_t n = 10'000'000;
  check_build<u32, 256>(gen::generate_keys<u32>(n, gen::Distribution::Uniform, 99), "10M u32");
  check_build<u64, 256>(gen::generate_keys<u64>(n, gen::Distribution::DuplicateHeavy, 99),
                        "10M dup-heavy u64");
}

CHECK_MAIN()
