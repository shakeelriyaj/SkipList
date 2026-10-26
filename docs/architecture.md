# Architecture

Status: Phase 2 (CPU reference and data model) is complete. The CUDA smoke
target from Phase 1 is in place but must be built and run on the GPU cluster.
See `implementation_plan.md` for the roadmap.

## Building

```sh
cmake -S . -B build                 # Release; CUDA targets added only if nvcc is found
cmake --build build -j
ctest --test-dir build --output-on-failure

# Host tests under AddressSanitizer + UBSan
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DGPU_INDEX_SANITIZE=ON
cmake --build build-asan && ./build-asan/test_cpu_reference [name-filter]
```

On the cluster, set the GPU architecture explicitly when CMake is older than 3.24:
`-DCMAKE_CUDA_ARCHITECTURES=80` (for example). Debug CUDA builds use `-G`; others use `-lineinfo`.

## Source map

| File | Contents |
|---|---|
| `include/gpu_index/cpu_ordered_set.hpp` | `CpuOrderedSet<Key>`: the trusted oracle for all operations |
| `include/gpu_index/node_layout.hpp` | `HostNodeLayout`, `partition_into_nodes`, `validate_layout`, `flatten` |
| `include/gpu_index/validation.hpp` | `ValidationResult` / `ValidationError` (device-friendly: code + index) |
| `include/gpu_index/reduce_ops.hpp` | `ReduceOp`, operator functors, identities (host and device) |
| `include/gpu_index/common.cuh` | `GPU_INDEX_CUDA_CHECK`, `cuda_error`, `DeviceBuffer`, `DeviceSpan` |
| `src/smoke.cu` | Phase 1 toolchain check (CUB radix sort of 1M keys) |
| `tests/check.hpp` | Zero-dependency test harness |
| `tests/random_gen.hpp` | Seeded generators: uniform, duplicate-heavy, clustered, skewed |
| `tests/test_cpu_reference.cpp` | Edge cases from readme sec. 8.1, validator corruption tests, randomized tests against `std::set` |

## Data model

Unique-key **set** of unsigned integers (`uint32_t`, `uint64_t`), ordered by `<`.

### Packed node layout (v1)

```
keys:        [ k0 k1 k2 k3 | k4 k5 k6 k7 | k8 k9 -- -- ]   capacity C = 4
node_counts: [ 4           | 4           | 2           ]
node_max:    [ k3          | k7          | k9          ]   routing layer ("skip" level)
```

- Node `i` owns slots `[i*C, (i+1)*C)`. Node boundaries are implicit, so locating a node is O(1).
- Every node except the last is full. The last holds 1..C keys. An empty index has zero nodes.
- Slots past a node's count are padding. Their contents are unspecified and must not be read.
- `node_max[i]` is the largest key in node `i`. Point and range lookups do a binary search on `node_max` to pick a node, then a binary search within that node.

### Invariants (`validate_layout`, readme sec. 8.2)

Checked in this order. The first violation is returned as `(ValidationError, index)`:

1. `capacity > 0`.
2. `node_max` and `node_counts` have the same length.
3. Key storage holds at least `num_nodes * capacity` slots.
4. Each node has `1 <= count <= capacity`. In packed mode, every non-final node is full.
5. Keys within a node are strictly increasing. Equal neighbors report `DuplicateKey`; decreasing neighbors report `KeysOutOfOrder`.
6. `node_max[i]` equals the last valid key of node `i`.
7. `max(node i-1) < min(node i)`.
8. The sum of `node_counts` equals `logical_size`.

`validate_layout(layout, /*require_packed=*/false)` relaxes rule 4's fullness requirement. That mode is reserved for a later layout that leaves slack in nodes.

## Operation contract

These rules apply to every implementation: the CPU reference, the GPU flat array, and the GPU node index.

| Operation | Mutates? | Semantics |
|---|---|---|
| `build(keys)` | yes (replaces) | Accepts unsorted input with duplicates; duplicates collapse |
| `bulk_insert(batch)` | yes | Set union with the batch; existing keys are no-ops |
| `bulk_delete(batch)` | yes | Set difference; missing or repeated keys are ignored |
| `union_with(other)` | no, returns new | Mathematical set union |
| `split(p)` | no, returns pair | left = `{k < p}`, right = `{k >= p}`; either side may be empty |
| `range_query(lo, hi)` | no, returns keys | Half-open `[lo, hi)`; `lo >= hi` gives an empty result |
| `range_reduce(lo, hi, op)` | no, returns scalar | Same interval as `range_query`; an empty range returns the identity |
| `contains(queries)` | no | One byte per query (1 = present), in query order |

Notes:

- Because intervals are half-open, the maximum key value can never be included in a range result.
- Reduction operators and their identities: `Sum` (0, wraps modulo 2^bits), `Min` (max key), `Max` (0), `BitAnd` (all ones), `BitOr` (0), `BitXor` (0). All are exact and associative, so results are deterministic regardless of reduction order.

## GPU conventions (from `common.cuh`, used starting in Phase 3)

- Every CUDA call goes through `GPU_INDEX_CUDA_CHECK`, which throws `gpu_index::cuda_error`. Kernel launches are followed by `GPU_INDEX_CHECK_LAUNCH()`.
- `DeviceBuffer<T>` is move-only and RAII-managed. `resize_discard(n)` reuses existing capacity and does **not** preserve contents.
- Copies are stream-ordered. Only `to_host()` synchronizes, and only on the stream it is given.
- `DeviceSpan<T>` is a non-owning view. The caller must keep the underlying memory alive.

## Deviations from the plan

- The error macro is named `GPU_INDEX_CUDA_CHECK` rather than `CUDA_CHECK`, to avoid clashing with macros defined by other libraries.
- Node partitioning and the layout validator live in `node_layout.hpp`, not inside `cpu_ordered_set.hpp`. The GPU code reuses them on host copies of its device state.
- `docs/environment.md` (Phase 1) has not been written yet. It requires running on the cluster.
