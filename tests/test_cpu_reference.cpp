// Tests for the CPU reference set, the packed node layout, and the layout
// validator. Hand-computed cases cover readme sec. 8.1; the randomized
// section cross-checks CpuOrderedSet against an independent std::set model.

#include <cstdint>
#include <iterator>
#include <set>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "gpu_index/cpu_ordered_set.hpp"
#include "gpu_index/node_layout.hpp"
#include "random_gen.hpp"

using gpu_index::CpuOrderedSet;
using gpu_index::HostNodeLayout;
using gpu_index::ReduceOp;
using gpu_index::ValidationError;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Set32 = CpuOrderedSet<u32>;
using Vec32 = std::vector<u32>;

// ---------------------------------------------------------------- build

TEST_CASE(build_empty) {
  Set32 s(Vec32{});
  CHECK_EQ(s.size(), 0u);
  CHECK(s.empty());
  CHECK(s.validate().ok());
}

TEST_CASE(build_single) {
  Set32 s(Vec32{42});
  CHECK_EQ(s.size(), 1u);
  CHECK(s.keys() == Vec32{42});
}

TEST_CASE(build_sorted_and_unsorted_agree) {
  Set32 a(Vec32{1, 2, 3, 5, 8});
  Set32 b(Vec32{8, 3, 1, 5, 2});
  CHECK(a.keys() == b.keys());
  CHECK(a.validate().ok());
}

TEST_CASE(build_collapses_duplicates) {
  Set32 s(Vec32{7, 7, 7, 1, 1, 9, 7});
  CHECK(s.keys() == (Vec32{1, 7, 9}));
}

TEST_CASE(build_extreme_keys) {
  const u32 mx = UINT32_MAX;
  Set32 s(Vec32{mx, 0, mx, 0});
  CHECK(s.keys() == (Vec32{0, mx}));
}

TEST_CASE(build_replaces_previous_contents) {
  Set32 s(Vec32{1, 2, 3});
  s.build(Vec32{9});
  CHECK(s.keys() == Vec32{9});
}

// ---------------------------------------------------------------- insert / delete

TEST_CASE(insert_new_and_existing) {
  Set32 s(Vec32{10, 20, 30});
  s.bulk_insert(Vec32{20, 25, 5, 25, 40});
  CHECK(s.keys() == (Vec32{5, 10, 20, 25, 30, 40}));
}

TEST_CASE(insert_existing_only_keeps_size) {
  Set32 s(Vec32{1, 2, 3});
  s.bulk_insert(Vec32{3, 2, 1, 1});
  CHECK_EQ(s.size(), 3u);
}

TEST_CASE(insert_empty_batch_and_into_empty) {
  Set32 s;
  s.bulk_insert(Vec32{});
  CHECK(s.empty());
  s.bulk_insert(Vec32{4, 4, 2});
  CHECK(s.keys() == (Vec32{2, 4}));
}

TEST_CASE(delete_present_missing_and_repeated) {
  Set32 s(Vec32{1, 2, 3, 4, 5});
  s.bulk_delete(Vec32{2, 2, 99, 4, 0});
  CHECK(s.keys() == (Vec32{1, 3, 5}));
}

TEST_CASE(delete_everything_and_from_empty) {
  Set32 s(Vec32{1, 2, 3});
  s.bulk_delete(Vec32{3, 1, 2});
  CHECK(s.empty());
  s.bulk_delete(Vec32{1});
  CHECK(s.empty());
  s.bulk_delete(Vec32{});
  CHECK(s.empty());
}

// ---------------------------------------------------------------- union

TEST_CASE(union_disjoint) {
  Set32 a(Vec32{1, 3, 5}), b(Vec32{2, 4, 6});
  CHECK(a.union_with(b).keys() == (Vec32{1, 2, 3, 4, 5, 6}));
}

TEST_CASE(union_identical) {
  Set32 a(Vec32{1, 3, 5});
  CHECK(a.union_with(a).keys() == a.keys());
}

TEST_CASE(union_overlapping_and_empty) {
  Set32 a(Vec32{1, 2, 3, 4}), b(Vec32{3, 4, 5}), e;
  CHECK(a.union_with(b).keys() == (Vec32{1, 2, 3, 4, 5}));
  CHECK(a.union_with(e).keys() == a.keys());
  CHECK(e.union_with(a).keys() == a.keys());
  CHECK(e.union_with(e).empty());
}

TEST_CASE(union_does_not_modify_inputs) {
  Set32 a(Vec32{1, 2}), b(Vec32{3});
  (void)a.union_with(b);
  CHECK(a.keys() == (Vec32{1, 2}));
  CHECK(b.keys() == Vec32{3});
}

// ---------------------------------------------------------------- split

TEST_CASE(split_at_existing_key_goes_right) {
  Set32 s(Vec32{10, 20, 30, 40});
  auto parts = s.split(30);
  CHECK(parts.first.keys() == (Vec32{10, 20}));
  CHECK(parts.second.keys() == (Vec32{30, 40}));
}

TEST_CASE(split_at_absent_key) {
  Set32 s(Vec32{10, 20, 30, 40});
  auto parts = s.split(25);
  CHECK(parts.first.keys() == (Vec32{10, 20}));
  CHECK(parts.second.keys() == (Vec32{30, 40}));
}

TEST_CASE(split_below_min_and_above_max) {
  Set32 s(Vec32{10, 20});
  auto lo = s.split(0);
  CHECK(lo.first.empty());
  CHECK(lo.second.keys() == s.keys());
  auto hi = s.split(UINT32_MAX);
  CHECK(hi.first.keys() == s.keys());
  CHECK(hi.second.empty());
  auto e = Set32().split(5);
  CHECK(e.first.empty() && e.second.empty());
  CHECK(s.keys() == (Vec32{10, 20}));  // original untouched
}

// ---------------------------------------------------------------- range query

TEST_CASE(range_query_half_open) {
  Set32 s(Vec32{1, 2, 3, 4, 5});
  CHECK(s.range_query(2, 4) == (Vec32{2, 3}));
  CHECK(s.range_query(0, UINT32_MAX) == s.keys());
  CHECK(s.range_query(3, 3).empty());
  CHECK(s.range_query(4, 2).empty());
  CHECK(s.range_query(6, 100).empty());
  CHECK(Set32().range_query(0, 10).empty());
}

TEST_CASE(range_query_cannot_include_max_key) {
  // Half-open bounds: UINT32_MAX is unreachable as an included key.
  Set32 s(Vec32{1, UINT32_MAX});
  CHECK(s.range_query(0, UINT32_MAX) == Vec32{1});
}

// ---------------------------------------------------------------- range reduce

TEST_CASE(range_reduce_all_ops) {
  Set32 s(Vec32{3, 5, 6, 12});
  CHECK_EQ(s.range_reduce(0, 100, ReduceOp::Sum), 26u);
  CHECK_EQ(s.range_reduce(0, 100, ReduceOp::Min), 3u);
  CHECK_EQ(s.range_reduce(0, 100, ReduceOp::Max), 12u);
  CHECK_EQ(s.range_reduce(0, 100, ReduceOp::BitAnd), 3u & 5u & 6u & 12u);
  CHECK_EQ(s.range_reduce(0, 100, ReduceOp::BitOr), 3u | 5u | 6u | 12u);
  CHECK_EQ(s.range_reduce(0, 100, ReduceOp::BitXor), 3u ^ 5u ^ 6u ^ 12u);
  CHECK_EQ(s.range_reduce(5, 12, ReduceOp::Sum), 11u);
}

TEST_CASE(range_reduce_empty_returns_identity) {
  Set32 s(Vec32{3, 5});
  CHECK_EQ(s.range_reduce(6, 9, ReduceOp::Sum), 0u);
  CHECK_EQ(s.range_reduce(6, 9, ReduceOp::Min), UINT32_MAX);
  CHECK_EQ(s.range_reduce(6, 9, ReduceOp::Max), 0u);
  CHECK_EQ(s.range_reduce(6, 9, ReduceOp::BitAnd), UINT32_MAX);
  CHECK_EQ(s.range_reduce(6, 9, ReduceOp::BitOr), 0u);
  CHECK_EQ(s.range_reduce(6, 9, ReduceOp::BitXor), 0u);
  CHECK_EQ(s.range_reduce(9, 6, ReduceOp::Sum), 0u);
}

TEST_CASE(range_reduce_single_element) {
  Set32 s(Vec32{7});
  for (ReduceOp op : {ReduceOp::Sum, ReduceOp::Min, ReduceOp::Max, ReduceOp::BitAnd,
                      ReduceOp::BitOr, ReduceOp::BitXor}) {
    CHECK_EQ(s.range_reduce(7, 8, op), 7u);
  }
}

TEST_CASE(range_reduce_sum_wraps) {
  Set32 s(Vec32{UINT32_MAX, 2});
  CHECK_EQ(s.range_reduce(0, UINT32_MAX, ReduceOp::Sum), 2u);
  // MAX excluded by half-open bound; include it via a full-width u64 set.
  CpuOrderedSet<u64> w(std::vector<u64>{UINT64_MAX, 2});
  CHECK_EQ(w.range_reduce(0, UINT64_MAX, ReduceOp::Sum), u64{2});
}

// ---------------------------------------------------------------- contains

TEST_CASE(contains_batch) {
  Set32 s(Vec32{2, 4, 6});
  auto r = s.contains(Vec32{1, 2, 6, 7, 2, 0});
  CHECK(r == (std::vector<std::uint8_t>{0, 1, 1, 0, 1, 0}));
  CHECK(Set32().contains(Vec32{1}) == std::vector<std::uint8_t>{0});
  CHECK(s.contains(Vec32{}).empty());
}

// ---------------------------------------------------------------- node layout

TEST_CASE(layout_empty) {
  auto L = Set32().to_layout(4);
  CHECK_EQ(L.num_nodes(), 0u);
  CHECK(L.keys.empty());
  CHECK(gpu_index::validate_layout(L).ok());
}

TEST_CASE(layout_partial_final_node) {
  Set32 s(Vec32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  auto L = s.to_layout(4);
  CHECK_EQ(L.num_nodes(), 3u);
  CHECK_EQ(L.keys.size(), 12u);
  CHECK(L.node_counts == (std::vector<std::uint32_t>{4, 4, 2}));
  CHECK(L.node_max == (Vec32{4, 8, 10}));
  CHECK(gpu_index::validate_layout(L).ok());
  CHECK(gpu_index::flatten(L) == s.keys());
}

TEST_CASE(layout_exact_multiple_and_one_over) {
  Vec32 keys;
  for (u32 i = 0; i < 8; ++i) keys.push_back(i * 3);
  auto exact = gpu_index::partition_into_nodes(keys, 4);
  CHECK(exact.node_counts == (std::vector<std::uint32_t>{4, 4}));
  CHECK(gpu_index::validate_layout(exact).ok());
  keys.push_back(100);
  auto over = gpu_index::partition_into_nodes(keys, 4);
  CHECK(over.node_counts == (std::vector<std::uint32_t>{4, 4, 1}));
  CHECK(gpu_index::validate_layout(over).ok());
}

TEST_CASE(layout_capacity_one_and_large) {
  Set32 s(Vec32{5, 6, 7});
  CHECK(gpu_index::validate_layout(s.to_layout(1)).ok());
  CHECK_EQ(s.to_layout(1).num_nodes(), 3u);
  CHECK(gpu_index::validate_layout(s.to_layout(1024)).ok());
  CHECK_EQ(s.to_layout(1024).num_nodes(), 1u);
}

TEST_CASE(layout_rejects_zero_capacity) {
  bool threw = false;
  try {
    (void)Set32(Vec32{1}).to_layout(0);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

// ---------------------------------------------------------------- validator detects corruption

namespace {
HostNodeLayout<u32> good_layout() {
  return Set32(Vec32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}).to_layout(4);  // nodes {1-4}{5-8}{9,10}
}
ValidationError error_of(const HostNodeLayout<u32>& L, bool packed = true) {
  return gpu_index::validate_layout(L, packed).error;
}
}  // namespace

TEST_CASE(validator_detects_each_invariant) {
  {
    auto L = good_layout();
    L.node_capacity = 0;
    CHECK(error_of(L) == ValidationError::InvalidCapacity);
  }
  {
    auto L = good_layout();
    L.node_max.pop_back();
    CHECK(error_of(L) == ValidationError::NodeCountMismatch);
  }
  {
    auto L = good_layout();
    L.keys.resize(10);
    CHECK(error_of(L) == ValidationError::BufferTooSmall);
  }
  {
    auto L = good_layout();
    L.node_counts[2] = 0;
    CHECK(error_of(L) == ValidationError::EmptyNode);
  }
  {
    auto L = good_layout();
    L.node_counts[2] = 5;
    CHECK(error_of(L) == ValidationError::NodeOverCapacity);
  }
  {
    auto L = good_layout();
    L.node_counts[0] = 3;
    L.node_max[0] = 3;
    L.logical_size = 9;
    CHECK(error_of(L) == ValidationError::NodeNotFull);
    CHECK(error_of(L, /*packed=*/false) == ValidationError::None);
  }
  {
    auto L = good_layout();
    std::swap(L.keys[1], L.keys[2]);
    auto r = gpu_index::validate_layout(L);
    CHECK(r.error == ValidationError::KeysOutOfOrder);
    CHECK_EQ(r.index, 2u);
  }
  {
    auto L = good_layout();
    L.keys[6] = L.keys[5];
    auto r = gpu_index::validate_layout(L);
    CHECK(r.error == ValidationError::DuplicateKey);
    CHECK_EQ(r.index, 6u);
  }
  {
    auto L = good_layout();
    L.keys[4] = 4;  // first key of node 1 equals max of node 0
    CHECK(error_of(L) == ValidationError::DuplicateKey);
  }
  {
    auto L = good_layout();
    L.keys[8] = 0;  // node 2 starts below node 1's max
    L.keys[9] = 1;
    L.node_max[2] = 1;
    auto r = gpu_index::validate_layout(L);
    CHECK(r.error == ValidationError::NodeOrderViolated);
    CHECK_EQ(r.index, 2u);
  }
  {
    auto L = good_layout();
    L.node_max[1] = 7;
    CHECK(error_of(L) == ValidationError::NodeMaxMismatch);
  }
  {
    auto L = good_layout();
    L.logical_size = 11;
    CHECK(error_of(L) == ValidationError::SizeMismatch);
  }
  {
    // Padding past the final node's count must be ignored.
    auto L = good_layout();
    L.keys[10] = 0;
    L.keys[11] = 0;
    CHECK(error_of(L) == ValidationError::None);
  }
}

// ---------------------------------------------------------------- generators

TEST_CASE(generators_are_reproducible) {
  for (auto d : {gen::Distribution::Uniform, gen::Distribution::DuplicateHeavy,
                 gen::Distribution::Clustered, gen::Distribution::Skewed}) {
    auto a = gen::generate_keys<u32>(1000, d, 123);
    auto b = gen::generate_keys<u32>(1000, d, 123);
    auto c = gen::generate_keys<u32>(1000, d, 124);
    CHECK(a == b);
    CHECK(a != c);
  }
  auto dup = gen::generate_keys<u32>(10000, gen::Distribution::DuplicateHeavy, 1);
  CHECK(Set32(dup).size() <= 1000u);
  auto bounded = gen::generate_keys<u32>(1000, gen::Distribution::Uniform, 1, 50);
  for (u32 k : bounded) CHECK(k <= 50u);
}

// ---------------------------------------------------------------- randomized vs std::set

namespace {

template <typename Key>
std::vector<Key> as_vector(const std::set<Key>& s) {
  return std::vector<Key>(s.begin(), s.end());
}

// Runs a seeded sequence of mixed operations on CpuOrderedSet and a std::set
// model, comparing logical contents and layouts after every step.
template <typename Key>
void randomized_sequence(std::uint64_t seed, Key max_key) {
  std::mt19937_64 rng(seed);
  const auto dists = {gen::Distribution::Uniform, gen::Distribution::DuplicateHeavy,
                      gen::Distribution::Clustered, gen::Distribution::Skewed};
  auto pick_dist = [&] { return *(dists.begin() + gen::uniform_u64(rng, dists.size() - 1)); };
  auto batch = [&](std::size_t max_n) {
    return gen::generate_keys<Key>(gen::uniform_u64(rng, max_n), pick_dist(), rng(), max_key);
  };
  auto random_key = [&] { return static_cast<Key>(gen::uniform_u64(rng, max_key)); };

  std::vector<Key> initial = batch(3000);
  CpuOrderedSet<Key> s(initial);
  std::set<Key> model(initial.begin(), initial.end());

  for (int step = 0; step < 40; ++step) {
    switch (gen::uniform_u64(rng, 5)) {
      case 0: {
        auto b = batch(1500);
        s.bulk_insert(b);
        model.insert(b.begin(), b.end());
        break;
      }
      case 1: {
        // Mix of present keys (hit rate varies) and arbitrary keys.
        auto b = gen::sample(s.keys(), static_cast<double>(gen::uniform_u64(rng, 100)) / 100.0, rng());
        auto extra = batch(500);
        b.insert(b.end(), extra.begin(), extra.end());
        s.bulk_delete(b);
        for (Key k : b) model.erase(k);
        break;
      }
      case 2: {
        auto other_keys = gen::sample(s.keys(), 0.5, rng());
        auto extra = batch(1500);
        other_keys.insert(other_keys.end(), extra.begin(), extra.end());
        CpuOrderedSet<Key> other(other_keys);
        s = s.union_with(other);
        model.insert(other_keys.begin(), other_keys.end());
        break;
      }
      case 3: {
        const Key p = random_key();
        auto parts = s.split(p);
        std::vector<Key> left, right;
        for (Key k : model) (k < p ? left : right).push_back(k);
        CHECK(parts.first.keys() == left);
        CHECK(parts.second.keys() == right);
        CHECK_EQ(parts.first.size() + parts.second.size(), s.size());
        s = gen::uniform_u64(rng, 1) ? parts.first.union_with(parts.second) : s;
        break;
      }
      case 4: {
        Key lo = random_key(), hi = random_key();
        if (gen::uniform_u64(rng, 1)) std::swap(lo, hi);
        std::vector<Key> expect;
        if (lo < hi) {
          expect.assign(model.lower_bound(lo), model.lower_bound(hi));
        }
        CHECK(s.range_query(lo, hi) == expect);
        for (ReduceOp op : {ReduceOp::Sum, ReduceOp::Min, ReduceOp::Max, ReduceOp::BitAnd,
                            ReduceOp::BitOr, ReduceOp::BitXor}) {
          Key acc = gpu_index::reduce_identity<Key>(op);
          for (Key k : expect) acc = gpu_index::reduce_apply(op, acc, k);
          CHECK_EQ(s.range_reduce(lo, hi, op), acc);
        }
        break;
      }
      case 5: {
        auto q = batch(500);
        auto r = s.contains(q);
        for (std::size_t i = 0; i < q.size(); ++i) CHECK_EQ(r[i] != 0, model.count(q[i]) == 1);
        break;
      }
    }
    REQUIRE(s.validate().ok());
    REQUIRE(s.keys() == as_vector(model));
    for (std::size_t cap : {1u, 7u, 64u}) {
      auto L = s.to_layout(cap);
      REQUIRE(gpu_index::validate_layout(L).ok());
      REQUIRE(gpu_index::flatten(L) == s.keys());
    }
  }
}

}  // namespace

TEST_CASE(randomized_u32_full_range) {
  for (std::uint64_t seed = 1; seed <= 25; ++seed) randomized_sequence<u32>(seed, UINT32_MAX);
}

TEST_CASE(randomized_u32_narrow_range_high_overlap) {
  for (std::uint64_t seed = 100; seed <= 125; ++seed) randomized_sequence<u32>(seed, 4096);
}

TEST_CASE(randomized_u64_full_range) {
  for (std::uint64_t seed = 200; seed <= 215; ++seed) randomized_sequence<u64>(seed, UINT64_MAX);
}

CHECK_MAIN()
