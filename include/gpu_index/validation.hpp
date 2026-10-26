#pragma once

// Result type shared by the CPU layout validator and (later) the GPU
// validator. Kept as a plain error code + index so a device kernel can
// produce it without strings or allocation.

#include <cstddef>

namespace gpu_index {

enum class ValidationError : int {
  None = 0,
  InvalidCapacity,    // node capacity is zero; index unused
  NodeCountMismatch,  // metadata arrays disagree on node count; index = offending array size
  BufferTooSmall,     // key storage smaller than num_nodes * capacity; index = storage size
  EmptyNode,          // node has zero keys; index = node
  NodeOverCapacity,   // node count exceeds capacity; index = node
  NodeNotFull,        // packed layout: a non-final node is partially filled; index = node
  KeysOutOfOrder,     // keys within a node decrease; index = global slot of the later key
  DuplicateKey,       // equal adjacent keys (within or across nodes); index = global slot of the later key
  NodeOrderViolated,  // max(node i-1) > min(node i); index = node i
  NodeMaxMismatch,    // routing key != last valid key of the node; index = node
  SizeMismatch,       // sum of node counts != logical size; index = sum of counts
};

struct ValidationResult {
  ValidationError error = ValidationError::None;
  std::size_t index = 0;

  bool ok() const { return error == ValidationError::None; }
  explicit operator bool() const { return ok(); }

  static ValidationResult success() { return {}; }
  static ValidationResult failure(ValidationError e, std::size_t idx) { return {e, idx}; }
};

inline const char* to_string(ValidationError e) {
  switch (e) {
    case ValidationError::None: return "ok";
    case ValidationError::InvalidCapacity: return "invalid node capacity";
    case ValidationError::NodeCountMismatch: return "node metadata count mismatch";
    case ValidationError::BufferTooSmall: return "key buffer too small";
    case ValidationError::EmptyNode: return "empty node";
    case ValidationError::NodeOverCapacity: return "node over capacity";
    case ValidationError::NodeNotFull: return "non-final node not full";
    case ValidationError::KeysOutOfOrder: return "keys out of order within node";
    case ValidationError::DuplicateKey: return "duplicate key";
    case ValidationError::NodeOrderViolated: return "adjacent nodes out of order";
    case ValidationError::NodeMaxMismatch: return "node max mismatch";
    case ValidationError::SizeMismatch: return "logical size mismatch";
  }
  return "unknown";
}

}  // namespace gpu_index
