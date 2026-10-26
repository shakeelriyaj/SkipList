#pragma once

// Reduction operators for range_reduce. Usable from host code and, when
// compiled by nvcc, from device code.
//
// Semantics (v1, unsigned integer keys only):
//   Sum    : wraps modulo 2^bits(Key), identity 0
//   Min    : identity = max Key value
//   Max    : identity = 0
//   BitAnd : identity = all ones
//   BitOr  : identity = 0
//   BitXor : identity = 0
// All operators are associative and exact, so results are deterministic
// regardless of reduction order. An empty range returns the identity.

#include <type_traits>

#if defined(__CUDACC__)
#define GPU_INDEX_HD __host__ __device__
#else
#define GPU_INDEX_HD
#endif

namespace gpu_index {

enum class ReduceOp { Sum, Min, Max, BitAnd, BitOr, BitXor };

template <typename Key>
struct SumOp {
  GPU_INDEX_HD Key operator()(Key a, Key b) const { return static_cast<Key>(a + b); }
};
template <typename Key>
struct MinOp {
  GPU_INDEX_HD Key operator()(Key a, Key b) const { return b < a ? b : a; }
};
template <typename Key>
struct MaxOp {
  GPU_INDEX_HD Key operator()(Key a, Key b) const { return a < b ? b : a; }
};
template <typename Key>
struct BitAndOp {
  GPU_INDEX_HD Key operator()(Key a, Key b) const { return static_cast<Key>(a & b); }
};
template <typename Key>
struct BitOrOp {
  GPU_INDEX_HD Key operator()(Key a, Key b) const { return static_cast<Key>(a | b); }
};
template <typename Key>
struct BitXorOp {
  GPU_INDEX_HD Key operator()(Key a, Key b) const { return static_cast<Key>(a ^ b); }
};

template <typename Key>
GPU_INDEX_HD constexpr Key reduce_identity(ReduceOp op) {
  static_assert(std::is_integral<Key>::value && std::is_unsigned<Key>::value,
                "v1 supports unsigned integer keys only");
  switch (op) {
    case ReduceOp::Min:
    case ReduceOp::BitAnd:
      return static_cast<Key>(~Key(0));
    case ReduceOp::Sum:
    case ReduceOp::Max:
    case ReduceOp::BitOr:
    case ReduceOp::BitXor:
    default:
      return Key(0);
  }
}

// Runtime-dispatched combine. Kernels should prefer the functor types above
// so the operator is resolved at compile time.
template <typename Key>
GPU_INDEX_HD Key reduce_apply(ReduceOp op, Key a, Key b) {
  switch (op) {
    case ReduceOp::Sum: return SumOp<Key>{}(a, b);
    case ReduceOp::Min: return MinOp<Key>{}(a, b);
    case ReduceOp::Max: return MaxOp<Key>{}(a, b);
    case ReduceOp::BitAnd: return BitAndOp<Key>{}(a, b);
    case ReduceOp::BitOr: return BitOrOp<Key>{}(a, b);
    case ReduceOp::BitXor: return BitXorOp<Key>{}(a, b);
  }
  return a;
}

inline const char* to_string(ReduceOp op) {
  switch (op) {
    case ReduceOp::Sum: return "sum";
    case ReduceOp::Min: return "min";
    case ReduceOp::Max: return "max";
    case ReduceOp::BitAnd: return "bit_and";
    case ReduceOp::BitOr: return "bit_or";
    case ReduceOp::BitXor: return "bit_xor";
  }
  return "unknown";
}

}  // namespace gpu_index
