#pragma once

// Device-side validator for the packed node layout (node_layout.hpp).
//
// It checks the same invariants as the host validate_layout(), and returns
// the same first violation (error code and index), so the two can be
// cross-checked in tests.
//
// In the packed layout every node except the last is full, so logical key p
// lives in slot p and node counts are implied by (logical_size, capacity).
// Bookkeeping checks (array sizes) are O(1) and run on the host first, so
// the kernel never reads past an allocation. The per-key checks run in one
// kernel with one thread per key:
//
//   keys[p-1] vs keys[p]   within a node: KeysOutOfOrder / DuplicateKey
//                          across a node boundary: NodeOrderViolated / DuplicateKey
//   node_max[node]         compared at the last key of each node: NodeMaxMismatch
//
// "First" violation: validate_layout walks nodes in order and, within node i,
// reports intra-node key errors (ascending slot), then node_max, then the
// boundary with node i-1. Each device violation gets a rank following that
// order:
//
//   rank = node * (cap + 1) + offset
//   offset = j - 1      intra-node error at in-node position j (1 .. cap-1)
//          = cap - 1    node_max mismatch
//          = cap        boundary with the previous node
//
// (rank << 4 | error_code) is folded with atomicMin, so the smallest rank
// wins regardless of thread scheduling: the result is deterministic.
//
// Cost: one coalesced pass over the keys and node_max, one 8-byte readback,
// and one stream synchronization.

#include <climits>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "gpu_index/common.cuh"
#include "gpu_index/node_layout.hpp"
#include "gpu_index/validation.hpp"

namespace gpu_index {
namespace detail {

using violation_t = unsigned long long;
constexpr violation_t kNoViolation = ULLONG_MAX;
constexpr unsigned kCodeBits = 4;
static_assert(static_cast<int>(ValidationError::SizeMismatch) < (1 << kCodeBits),
              "error codes must fit in kCodeBits");

__host__ __device__ inline violation_t encode_violation(std::size_t rank, ValidationError e) {
  return (static_cast<violation_t>(rank) << kCodeBits) | static_cast<violation_t>(e);
}

template <typename Key>
__global__ void validate_packed_kernel(const Key* __restrict__ keys,
                                       const Key* __restrict__ node_max, std::size_t n,
                                       std::size_t cap, violation_t* first) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t p = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; p < n;
       p += stride) {
    const std::size_t node = p / cap;
    const std::size_t j = p - node * cap;
    const std::size_t node_rank = node * (cap + 1);
    const Key cur = keys[p];

    if (p > 0) {
      const Key prev = keys[p - 1];  // in a packed layout, also the last key of node-1 when j == 0
      if (!(prev < cur)) {
        const bool dup = (prev == cur);
        violation_t v;
        if (j > 0) {
          v = encode_violation(node_rank + (j - 1),
                               dup ? ValidationError::DuplicateKey : ValidationError::KeysOutOfOrder);
        } else {
          v = encode_violation(node_rank + cap, dup ? ValidationError::DuplicateKey
                                                    : ValidationError::NodeOrderViolated);
        }
        atomicMin(first, v);
      }
    }
    if ((j == cap - 1 || p == n - 1) && node_max[node] != cur) {
      atomicMin(first, encode_violation(node_rank + (cap - 1), ValidationError::NodeMaxMismatch));
    }
  }
}

// Converts the winning encoded violation back into the (error, index) that
// validate_layout reports for the same defect.
inline ValidationResult decode_violation(violation_t v, std::size_t cap) {
  if (v == kNoViolation) return ValidationResult::success();
  const auto error = static_cast<ValidationError>(v & ((1u << kCodeBits) - 1));
  const std::size_t rank = static_cast<std::size_t>(v >> kCodeBits);
  const std::size_t node = rank / (cap + 1);
  const std::size_t offset = rank % (cap + 1);
  switch (error) {
    case ValidationError::NodeMaxMismatch:
    case ValidationError::NodeOrderViolated:
      return ValidationResult::failure(error, node);
    case ValidationError::KeysOutOfOrder:
      return ValidationResult::failure(error, node * cap + offset + 1);
    case ValidationError::DuplicateKey:
      // Boundary duplicate: the first key of `node`; intra-node: the later slot.
      return ValidationResult::failure(error, offset == cap ? node * cap : node * cap + offset + 1);
    default:
      throw std::logic_error("decode_violation: unexpected error code from device");
  }
}

// Validates a packed layout held in device memory:
//   key_slots     storage for num_nodes * cap slots (logical keys in [0, logical_size))
//   node_max      one routing key per node
// Returns the same first violation as validate_layout(..., require_packed=true)
// on the equivalent host layout. Synchronizes `stream` once (skipped when the
// index is empty or a bookkeeping check fails).
template <typename Key>
ValidationResult validate_packed_layout_device(DeviceSpan<const Key> key_slots,
                                               DeviceSpan<const Key> node_max,
                                               std::size_t logical_size, std::size_t cap,
                                               cudaStream_t stream) {
  using VE = ValidationError;
  if (cap == 0) return ValidationResult::failure(VE::InvalidCapacity, 0);
  const std::size_t nodes = num_nodes_for(logical_size, cap);
  if (node_max.size() != nodes) return ValidationResult::failure(VE::NodeCountMismatch, node_max.size());
  if (key_slots.size() / cap < nodes) return ValidationResult::failure(VE::BufferTooSmall, key_slots.size());
  if (logical_size == 0) return ValidationResult::success();

  DeviceBuffer<violation_t> first(1);
  GPU_INDEX_CUDA_CHECK(cudaMemsetAsync(first.data(), 0xFF, sizeof(violation_t), stream));

  constexpr unsigned block = 256;
  constexpr std::size_t max_grid = 1u << 16;  // grid-stride loop covers the rest
  std::size_t grid = (logical_size + block - 1) / block;
  if (grid > max_grid) grid = max_grid;
  validate_packed_kernel<Key><<<static_cast<unsigned>(grid), block, 0, stream>>>(
      key_slots.data(), node_max.data(), logical_size, cap, first.data());
  GPU_INDEX_CHECK_LAUNCH();

  violation_t result = kNoViolation;
  GPU_INDEX_CUDA_CHECK(
      cudaMemcpyAsync(&result, first.data(), sizeof(result), cudaMemcpyDeviceToHost, stream));
  GPU_INDEX_CUDA_CHECK(cudaStreamSynchronize(stream));
  return decode_violation(result, cap);
}

}  // namespace detail
}  // namespace gpu_index
