// Phase 4 bulk_insert tests: after every insert, GpuOrderedIndex and
// FlatSortedArray must match CpuOrderedSet::bulk_insert exactly (same logical
// keys, same packed node layout) and pass validate(), for every edge case in
// readme sec. 8.1 plus seeded random insert sequences, for uint32_t and
// uint64_t keys.

#include <algorithm>
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
using gpu_index::DeviceSpan;
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

// The oracle and both GPU structures, kept in lockstep.
template <typename Key, std::size_t Cap>
struct Trio {
  CpuOrderedSet<Key> ref;
  GpuOrderedIndex<Key, Cap> index;
  FlatSortedArray<Key> flat;

  explicit Trio(const std::vector<Key>& initial) : ref(initial) {
    index.build_from_host(initial);
    flat.build_from_host(initial);
  }

  void insert(const std::vector<Key>& batch) {
    ref.bulk_insert(batch);
    index.bulk_insert_from_host(batch);
    flat.bulk_insert_from_host(batch);
  }

  void check(const std::string& label) {
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
    CHECK_EQ(flat.size(), ref.size());
    if (flat.to_host() != ref.keys()) {
      check::report(__FILE__, __LINE__, label + ": flat array differs from CPU reference");
    }
  }
};

// Builds from `initial`, applies each batch in order, checks after each one.
template <typename Key, std::size_t Cap>
void check_inserts(const std::vector<Key>& initial, const std::vector<std::vector<Key>>& batches,
                   const std::string& label) {
  Trio<Key, Cap> t(initial);
  t.check(label + " after build");
  for (std::size_t i = 0; i < batches.size(); ++i) {
    t.insert(batches[i]);
    t.check(label + " after insert " + std::to_string(i));
  }
}

template <typename Key>
void check_all_caps(const std::vector<Key>& initial, const std::vector<std::vector<Key>>& batches,
                    const std::string& label) {
  check_inserts<Key, 1>(initial, batches, label + " cap=1");
  check_inserts<Key, 4>(initial, batches, label + " cap=4");
  check_inserts<Key, 256>(initial, batches, label + " cap=256");
}

std::vector<u32> range_u32(u32 begin, u32 end, u32 step = 1) {
  std::vector<u32> out;
  for (u32 k = begin; k < end; k += step) out.push_back(k);
  return out;
}

}  // namespace

// ---------------------------------------------------------------- edge cases

TEST_CASE(insert_into_empty) {
  check_all_caps<u32>({}, {{5, 1, 3}}, "into empty u32");
  check_all_caps<u64>({}, {{5, 1, 3}}, "into empty u64");
}

TEST_CASE(insert_empty_batch) {
  check_all_caps<u32>({}, {{}}, "both empty");
  check_all_caps<u32>({4, 2, 9}, {{}}, "empty batch u32");
  check_all_caps<u64>({4, 2, 9}, {{}}, "empty batch u64");
}

TEST_CASE(insert_batch_with_internal_duplicates) {
  check_all_caps<u32>({10, 20}, {{5, 5, 15, 15, 15, 25, 5}}, "batch dups u32");
  check_all_caps<u32>({10, 20}, {std::vector<u32>(10000, 7u)}, "batch all-equal u32");
  check_all_caps<u64>({10, 20}, {{5, 5, 15, 15, 15, 25, 5}}, "batch dups u64");
}

TEST_CASE(insert_overlap_with_existing) {
  // Partial overlap: the running example from the design notes.
  check_all_caps<u32>({3, 7, 9, 12}, {{9, 1, 7, 15, 1}}, "partial overlap");
  // Full overlap: every batch key already present.
  check_all_caps<u32>({3, 7, 9, 12}, {{12, 3, 9}}, "full overlap");
}

TEST_CASE(insert_existing_keys_leaves_storage_untouched) {
  GpuOrderedIndex<u32, 4> index;
  index.build_from_host({1, 2, 3, 4, 5});
  FlatSortedArray<u32> flat;
  flat.build_from_host({1, 2, 3, 4, 5});
  const u32* index_slots = index.key_slots().data();
  const u32* flat_keys = flat.keys().data();

  index.bulk_insert_from_host({5, 1, 3, 3});
  flat.bulk_insert_from_host({5, 1, 3, 3});
  index.bulk_insert_from_host({});
  flat.bulk_insert_from_host({});

  CHECK(index.key_slots().data() == index_slots);
  CHECK(flat.keys().data() == flat_keys);
  CHECK(index.to_host() == (std::vector<u32>{1, 2, 3, 4, 5}));
  CHECK(index.validate().ok());
}

TEST_CASE(insert_before_after_and_between) {
  const std::vector<u32> base = range_u32(100, 200, 10);  // 100, 110, ..., 190
  check_all_caps<u32>(base, {{1, 2, 3}}, "all below");
  check_all_caps<u32>(base, {{500, 400, 300}}, "all above");
  check_all_caps<u32>(base, {{105, 155, 185}}, "interleaved");
  check_all_caps<u32>(base, {{0, 145, 1000}}, "both ends and middle");
}

TEST_CASE(insert_extreme_keys) {
  check_all_caps<u32>({1u << 31, 5u}, {{UINT32_MAX, 0u, (1u << 31) - 1, UINT32_MAX}},
                      "extreme u32");
  check_all_caps<u64>({u64(1) << 63, 5u},
                      {{UINT64_MAX, 0u, (u64(1) << 63) - 1, u64(1) << 32, UINT64_MAX}},
                      "extreme u64");
  // Max key already present, then inserted again.
  check_all_caps<u32>({UINT32_MAX}, {{UINT32_MAX, 0u}}, "max present u32");
}

TEST_CASE(insert_node_boundaries) {
  // cap = 4: results of 3, 4, 5, 8 and 9 keys, starting from both partial
  // and exactly full last nodes.
  for (u32 before : {2u, 4u}) {
    for (u32 after : {3u, 4u, 5u, 8u, 9u}) {
      if (after <= before) continue;
      const std::vector<u32> initial = range_u32(0, 2 * before, 2);  // evens
      const std::vector<u32> batch = range_u32(1, 2 * (after - before), 2);  // odds
      check_inserts<u32, 4>(initial, {batch},
                            "boundary " + std::to_string(before) + "->" + std::to_string(after));
    }
  }
  // cap = 256: grow across 256 and 512.
  for (u32 after : {255u, 256u, 257u, 512u, 513u}) {
    const std::vector<u32> initial = range_u32(0, 400, 2);  // 200 evens
    const std::vector<u32> batch = range_u32(1, 2 * (after - 200), 2);
    check_inserts<u32, 256>(initial, {batch}, "boundary 200->" + std::to_string(after));
  }
}

TEST_CASE(insert_repeated_small_batches) {
  std::vector<std::vector<u32>> batches;
  for (u32 i = 0; i < 50; ++i) batches.push_back({i * 7 % 61, i * 13 % 61, i});
  check_all_caps<u32>({}, batches, "repeated small");
}

TEST_CASE(insert_after_build_replaces_contents) {
  GpuOrderedIndex<u32, 4> index;
  index.build_from_host({1, 2, 3, 4, 5, 6, 7, 8, 9});
  index.bulk_insert_from_host({10});
  index.build_from_host({100, 50});
  index.bulk_insert_from_host({75, 50});
  CHECK(index.to_host() == (std::vector<u32>{50, 75, 100}));
  CHECK_EQ(index.num_nodes(), 1u);
  CHECK(index.validate().ok());
}

TEST_CASE(insert_from_device_span_leaves_input_untouched) {
  const std::vector<u32> batch = {9, 3, 3, 7, 1, 9, 0};
  DeviceBuffer<u32> d;
  d.copy_from_host(batch);

  GpuOrderedIndex<u32, 4> index;
  index.build_from_host({2, 4, 6});
  index.bulk_insert(d.span());
  FlatSortedArray<u32> flat;
  flat.build_from_host({2, 4, 6});
  flat.bulk_insert(d.span());

  CHECK(d.to_host() == batch);
  CHECK(index.to_host() == (std::vector<u32>{0, 1, 2, 3, 4, 6, 7, 9}));
  CHECK(flat.to_host() == (std::vector<u32>{0, 1, 2, 3, 4, 6, 7, 9}));
  CHECK(index.validate().ok());
}

TEST_CASE(insert_own_keys) {
  // The batch aliases the index's own storage: must be a no-op.
  GpuOrderedIndex<u32, 4> index;
  index.build_from_host({8, 6, 4, 2, 0});
  index.bulk_insert(DeviceSpan<const u32>(index.key_slots().data(), index.size()));
  CHECK(index.to_host() == (std::vector<u32>{0, 2, 4, 6, 8}));
  CHECK(index.validate().ok());

  FlatSortedArray<u32> flat;
  flat.build_from_host({8, 6, 4, 2, 0});
  flat.bulk_insert(flat.keys());
  CHECK(flat.to_host() == (std::vector<u32>{0, 2, 4, 6, 8}));
}

TEST_CASE(insert_on_nondefault_stream) {
  cudaStream_t stream;
  GPU_INDEX_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const auto initial = gen::generate_keys<u32>(100000, gen::Distribution::Uniform, 7);
  const auto batch = gen::generate_keys<u32>(50000, gen::Distribution::Uniform, 8);
  CpuOrderedSet<u32> ref(initial);
  ref.bulk_insert(batch);

  GpuOrderedIndex<u32, 256> index;
  index.build_from_host(initial, stream);
  index.bulk_insert_from_host(batch, stream);
  CHECK(index.validate(stream).ok());
  CHECK(index.to_host(stream) == ref.keys());

  FlatSortedArray<u32> flat;
  flat.build_from_host(initial, stream);
  flat.bulk_insert_from_host(batch, stream);
  CHECK(flat.to_host(stream) == ref.keys());
  GPU_INDEX_CUDA_CHECK(cudaStreamDestroy(stream));
}

// ---------------------------------------------------------------- randomized

// Each batch mixes keys sampled from the current contents (to control the
// overlap rate) with freshly generated keys from the same distribution.
template <typename Key, std::size_t Cap>
void random_sequence(gen::Distribution dist, std::size_t initial_size, std::size_t batch_size,
                     double overlap, std::uint64_t seed, const std::string& label) {
  Trio<Key, Cap> t(gen::generate_keys<Key>(initial_size, dist, seed));
  t.check(label + " after build");
  for (std::uint64_t step = 0; step < 4; ++step) {
    const std::uint64_t s = seed * 1000 + step;
    std::vector<Key> batch = gen::sample(t.ref.keys(), overlap * batch_size /
                                                           std::max<std::size_t>(1, t.ref.size()),
                                         s);
    if (batch.size() > batch_size) batch.resize(batch_size);
    const auto fresh = gen::generate_keys<Key>(batch_size - batch.size(), dist, s + 500);
    batch.insert(batch.end(), fresh.begin(), fresh.end());
    t.insert(batch);
    t.check(label + " step " + std::to_string(step));
  }
}

TEST_CASE(insert_randomized) {
  const gen::Distribution dists[] = {gen::Distribution::Uniform,
                                     gen::Distribution::DuplicateHeavy,
                                     gen::Distribution::Clustered, gen::Distribution::Skewed};
  const std::size_t initial_sizes[] = {0, 17, 1000, 65537};
  const std::size_t batch_sizes[] = {1, 100, 20000};
  const double overlaps[] = {0.0, 0.5, 1.0};
  for (gen::Distribution dist : dists) {
    for (std::size_t n : initial_sizes) {
      for (std::size_t b : batch_sizes) {
        for (double overlap : overlaps) {
          const std::string label = std::string(gen::to_string(dist)) + " n=" +
                                    std::to_string(n) + " b=" + std::to_string(b) +
                                    " overlap=" + std::to_string(overlap);
          random_sequence<u32, 64>(dist, n, b, overlap, 1 + n + b, label + " u32");
          random_sequence<u64, 256>(dist, n, b, overlap, 2 + n + b, label + " u64");
        }
      }
    }
  }
}

TEST_CASE(insert_large) {
  // Small batch into a big index, and a batch as big as the index.
  random_sequence<u32, 256>(gen::Distribution::Uniform, 10'000'000, 1000, 0.5, 99,
                            "10M + 1k u32");
  random_sequence<u64, 256>(gen::Distribution::Uniform, 1'000'000, 1'000'000, 0.5, 98,
                            "1M + 1M u64");
}

CHECK_MAIN()
