#pragma once

// Seeded, reproducible key generators for tests and benchmarks.
// Uses std::mt19937_64 directly (no std::uniform_int_distribution) so that
// a given seed yields the same keys across standard library implementations.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace gen {

enum class Distribution {
  Uniform,         // uniform over [0, max_key]
  DuplicateHeavy,  // ~n/10 distinct values, so ~90% duplicates
  Clustered,       // dense runs around a few random centers
  Skewed,          // heavily biased toward small keys
};

inline const char* to_string(Distribution d) {
  switch (d) {
    case Distribution::Uniform: return "uniform";
    case Distribution::DuplicateHeavy: return "duplicate_heavy";
    case Distribution::Clustered: return "clustered";
    case Distribution::Skewed: return "skewed";
  }
  return "unknown";
}

// Uniform in [0, bound] (inclusive) via rejection sampling.
inline std::uint64_t uniform_u64(std::mt19937_64& rng, std::uint64_t bound) {
  if (bound == UINT64_MAX) return rng();
  const std::uint64_t range = bound + 1;
  const std::uint64_t limit = UINT64_MAX - UINT64_MAX % range;
  std::uint64_t x;
  do {
    x = rng();
  } while (x >= limit);
  return x % range;
}

template <typename Key>
std::vector<Key> generate_keys(std::size_t n, Distribution dist, std::uint64_t seed,
                               Key max_key = static_cast<Key>(~Key(0))) {
  std::mt19937_64 rng(seed);
  std::vector<Key> out(n);
  switch (dist) {
    case Distribution::Uniform:
      for (auto& k : out) k = static_cast<Key>(uniform_u64(rng, max_key));
      break;
    case Distribution::DuplicateHeavy: {
      const std::uint64_t distinct = std::max<std::uint64_t>(1, n / 10);
      const std::uint64_t bound = std::min<std::uint64_t>(max_key, distinct - 1);
      for (auto& k : out) k = static_cast<Key>(uniform_u64(rng, bound));
      break;
    }
    case Distribution::Clustered: {
      const std::size_t clusters = 16;
      std::vector<std::uint64_t> centers(clusters);
      for (auto& c : centers) c = uniform_u64(rng, max_key);
      const std::uint64_t spread = std::max<std::uint64_t>(1, n);
      for (auto& k : out) {
        const std::uint64_t c = centers[uniform_u64(rng, clusters - 1)];
        const std::uint64_t off = uniform_u64(rng, spread);
        k = static_cast<Key>(max_key - c >= off ? c + off : c - std::min(c, off));
      }
      break;
    }
    case Distribution::Skewed:
      for (auto& k : out) {
        const double u = static_cast<double>(rng() >> 11) * 0x1.0p-53;  // [0, 1)
        k = static_cast<Key>(std::floor(u * u * u * static_cast<double>(max_key)));
      }
      break;
  }
  return out;
}

// Each element of `keys` is kept independently with probability `fraction`.
template <typename Key>
std::vector<Key> sample(const std::vector<Key>& keys, double fraction, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<Key> out;
  for (const Key& k : keys) {
    if (static_cast<double>(rng() >> 11) * 0x1.0p-53 < fraction) out.push_back(k);
  }
  return out;
}

}  // namespace gen
