# GPU-Accelerated Batched Ordered Index

## 1. Project Overview

### 1.1 Objective

Build a GPU-accelerated ordered data structure inspired by skip lists, optimized for high-throughput batch operations rather than low-latency individual updates.

Traditional skip lists provide expected \(O(\log N)\) search, insertion, and deletion by organizing keys into multiple levels of linked nodes. However, pointer-heavy layouts and fine-grained updates can result in irregular memory access and limited GPU efficiency.

This project will explore a **GPU-native, multi-key ordered index** that stores multiple sorted keys in cache-friendly nodes and uses parallel algorithms to process large batches of operations. The goal is to improve throughput and reduce execution time for workloads such as index construction, sorted-data merging, and bulk updates.

The design will prioritize:
- Contiguous, coalesced, and cache-friendly memory access.
- Parallel execution across GPU thread blocks and cores.
- Efficient bulk insertion and deletion.
- Parallel union and split operations.
- Range queries and range reductions.
- Reproducible benchmarks against optimized CPU implementations.

### 1.2 Core Design Principle

The structure is batch-oriented and may sacrifice lock-free, single-key updates in exchange for higher throughput on bulk workloads.

Instead of modifying the index independently for every key, each operation will accept a batch, perform parallel preprocessing, and apply changes through a coordinated GPU execution phase.

The implementation should favor predictable memory access, bulk data movement, and parallel primitives over traditional pointer-based traversal.

### 1.3 Intended Workloads

Primary target workloads:

- Building an ordered index from a large dataset.
- Inserting or deleting large batches of keys.
- Merging multiple sorted datasets.
- Combining two indexes using set union.
- Splitting an index at a specified key.
- Executing range queries and reductions over ordered keys.

Individual updates, highly concurrent writers, and workloads requiring strict low-latency point mutations are outside the primary optimization target.

---

## 2. Requirements and Scope

### 2.1 Functional Requirements

The data structure must support the following operations:

| Operation | Description | Required behavior |
|---|---|---|
| `build` | Construct an index from input keys. | Produce a valid ordered index. |
| `bulk_insert` | Insert a batch of keys. | Preserve ordering and the chosen duplicate policy. |
| `bulk_delete` | Delete a batch of keys. | Remove matching keys and ignore or report missing keys according to the API contract. |
| `union` | Combine two ordered indexes. | Produce the ordered set union or documented multiset union. |
| `split` | Divide an index at a specified key. | Return two ordered indexes with unambiguous boundary semantics. |
| `range_query` | Retrieve keys within a range. | Return matching keys in sorted order. |
| `range_reduce` | Aggregate values over a key range. | Produce the correct result using a parallel reduction when applicable. |
| `contains` | Check membership of keys. | Support efficient batched membership queries. |
| `size` | Return the number of stored keys. | Report the logical number of keys. |
| `validate` | Check structural invariants. | Detect ordering, node occupancy, boundary, and metadata errors. |

The first implementation must prioritize `build`, `bulk_insert`, `bulk_delete`, `union`, `split`, and `range_reduce`. The remaining operations may be implemented as supporting functionality, but they must follow the same correctness and memory-safety requirements.

### 2.2 Non-Functional Requirements

- **Correctness:** Every operation must produce the same logical result as a trusted reference implementation.
- **Parallelism:** Large batches should be processed using GPU parallelism rather than serial per-key execution.
- **Memory efficiency:** Avoid unnecessary pointer chasing, excessive padding, and repeated full-index copies where practical.
- **Scalability:** Evaluate behavior as both dataset size and batch size increase.
- **Determinism:** Operations with identical inputs should produce identical logical results. Physical node layouts may differ unless canonical layout is explicitly required.
- **Safety:** Handle empty inputs, missing keys, duplicate keys, boundary conditions, and allocation failures.
- **Measurability:** Report throughput, latency, memory consumption, and speedup against CPU baselines.

Performance targets must be established experimentally rather than assumed in advance.

---

## 3. Proposed Data Structure

### 3.1 Multi-Key Ordered Nodes

Each node stores multiple keys in sorted order instead of storing a single key and a pointer to the next node.

A conceptual node contains:

```text
Node
├── keys: sorted array of keys
├── values: optional associated values
├── count: number of valid entries
├── capacity: maximum number of entries
└── metadata: optional routing or boundary information
```

For a set index, keys must be strictly increasing within each node. For a map index, each key must follow the project's duplicate-key and value-update policy.

Nodes must also satisfy the global ordering invariant:

```text
max(node[i]) < min(node[i + 1])
```

for nonempty adjacent nodes in a unique-key index.

The implementation should begin with a flat, contiguous array of fixed-capacity nodes or an equivalent compact representation. It should not assume that a pointer-based hierarchy is necessary to obtain the benefits of a skip list.

### 3.2 Node Capacity

Node capacity is a tunable parameter rather than a fixed design assumption.

Evaluate multiple capacities to understand the trade-off between:
- Search and insertion work within a node.
- Memory coalescing and cache utilization.
- Padding and unused storage.
- Load balancing across GPU thread blocks.
- Number of nodes and boundary metadata overhead.

Start with a simple fixed-capacity layout for predictable indexing. Consider variable occupancy or a hierarchical routing layer only after establishing a correct baseline and identifying a measurable bottleneck.

### 3.3 Index Representation

The initial implementation should use a contiguous GPU-resident representation that is easy to inspect and validate.

A possible representation is:

```text
Index
├── device_keys
├── device_values (optional)
├── node_offsets or node descriptors
├── node_counts
├── node_capacity
└── logical_size
```

An implementation using a flat sorted array with logical node boundaries is acceptable for the initial prototype. This representation makes it easier to implement bulk operations using sorting, merging, scans, and compaction before introducing more complex hierarchical navigation.

The data structure should be treated as an ordered index with skip-list-inspired multi-key nodes, not as a requirement to preserve traditional skip-list pointer semantics.

---

## 4. API Requirements

Define a clean API that separates logical operations from CUDA kernel implementation details.

The exact language and function signatures should follow the existing repository. If the repository is empty, use C++ with CUDA as the initial implementation target.

Conceptual interface:

```cpp
template <typename Key>
class GpuOrderedIndex {
public:
    void build(DeviceSpan<const Key> keys);

    void bulk_insert(DeviceSpan<const Key> keys);
    void bulk_delete(DeviceSpan<const Key> keys);

    GpuOrderedIndex union_with(
        const GpuOrderedIndex& other) const;

    std::pair<GpuOrderedIndex, GpuOrderedIndex>
    split(const Key& pivot) const;

    DeviceBuffer<Key> range_query(
        const Key& lower,
        const Key& upper) const;

    ResultType range_reduce(
        const Key& lower,
        const Key& upper) const;

    DeviceBuffer<bool> contains(
        DeviceSpan<const Key> queries) const;

    std::size_t size() const;

    ValidationResult validate() const;
};
```

The above is illustrative rather than a mandatory ABI. Adapt it to the selected CUDA libraries, type system, and repository conventions.

Requirements for every operation:

1. Document whether inputs and outputs reside on the CPU or GPU.
2. Define whether the operation modifies the current index or returns a new index.
3. Define stream and synchronization behavior.
4. Define ownership and lifetime of input and output buffers.
5. Define duplicate-key, missing-key, and error behavior.
6. Avoid unnecessary device-to-host transfers.
7. Ensure that temporary allocations are released safely.
8. Report errors rather than silently returning corrupted output.

Operations should support reuse of GPU memory where practical. A high-level wrapper may synchronize for ease of use, but the internal implementation should avoid unnecessary synchronization between dependent GPU kernels.

---

## 5. Operation Specifications

### 5.1 Build

**Input:** An arbitrary array of keys.

**Output:** A valid ordered index.

Required behavior:
- Accept empty and nonempty inputs.
- Support unsorted input keys.
- Apply the documented duplicate policy.
- Produce sorted, correctly partitioned nodes.
- Allocate storage based on actual logical size and selected capacity.
- Avoid serial insertion of every key.

Preferred initial algorithm:

1. Copy or access the input in device memory.
2. Sort the keys using a GPU sorting primitive.
3. Remove duplicates when operating in set mode.
4. Compute node boundaries using parallel indexing or scans.
5. Populate the node representation.
6. Validate ordering and metadata.

Sorting should use a mature GPU primitive when available instead of a custom implementation unless custom sorting is a specific research goal.

### 5.2 Bulk Insert

**Input:** An existing index and a batch of new keys.

**Output:** The updated logical index.

Preferred initial algorithm:

1. Normalize and sort the input batch.
2. Apply the duplicate policy to the incoming keys.
3. Merge the existing ordered keys with the incoming keys.
4. Resolve duplicates between existing and incoming keys.
5. Repartition the merged output into nodes.
6. Update metadata and logical size.

This merge-based approach provides a reliable baseline and exposes substantial parallel work. It may require rebuilding or copying a significant portion of the index, which must be measured.

A later optimization may identify affected nodes and perform localized updates. Implement this only if measurements show that full merging is too expensive for the target workloads.

### 5.3 Bulk Delete

**Input:** An existing index and a batch of keys to remove.

**Output:** The updated index.

Preferred initial algorithm:

1. Normalize the deletion batch according to the duplicate policy.
2. Sort the deletion keys if necessary.
3. Match deletion keys against the existing ordered index using parallel search or merge.
4. Mark retained keys.
5. Compact retained keys using a parallel prefix sum or stream compaction.
6. Repartition the retained keys into valid nodes.

Missing deletion keys must not corrupt the index. Repeated deletion keys must have well-defined behavior.

### 5.4 Union

**Input:** Two ordered indexes.

**Output:** An ordered index containing their union.

For set semantics, each key appears exactly once in the output.

Preferred initial algorithm:

1. Access both sorted key sequences.
2. Perform a parallel merge or use a parallel merge-path algorithm.
3. Resolve equal keys according to the set or map policy.
4. Compact the result.
5. Rebuild node boundaries and metadata.

The implementation should not insert each key from one index into the other sequentially.

For a map index, define which value wins when both inputs contain the same key. For a set index, the duplicate key is retained once.

### 5.5 Split

**Input:** An index and a pivot key.

**Output:** Two valid indexes.

The initial contract should be:

- Left output: all keys \(k < p\).
- Right output: all keys \(k \ge p\).

Here, \(p\) is the pivot key.

Required behavior:
- Preserve every input key exactly once across the two outputs.
- Preserve sorted order.
- Handle pivots below the minimum or above the maximum.
- Handle pivots absent from the index.
- Rebuild node boundaries if a split divides a node.
- Avoid modifying the original index unless the API explicitly specifies in-place behavior.

Preferred initial algorithm: find the pivot's lower-bound position with binary search, then produce two views or compacted outputs. Use independent output buffers initially for simplicity. Explore shared storage and zero-copy splitting only after establishing correct ownership and lifetime rules.

### 5.6 Range Query

**Input:** A lower and upper key bound.

**Output:** The matching keys in sorted order.

Use a half-open interval:

\[
[\text{lower},\text{upper})
\]

This includes the lower bound and excludes the upper bound.

The initial implementation can use lower-bound searches followed by a contiguous range extraction. A future version may support batched independent range queries.

### 5.7 Range Reduce

**Input:** A key interval and an associated value or aggregation rule.

**Output:** A single aggregate or a documented collection of aggregates.

For a key-value index, reduce the values associated with keys in the specified interval. For a key-only index, the caller must provide a supported mapping from keys to reduction values or select a defined built-in operation.

The initial API must specify:
- The key interval and its boundary semantics.
- The aggregation operator.
- The identity element, if applicable.
- Whether the reduction must be deterministic.
- How floating-point rounding differences are handled.
- What happens when the range is empty.

Start with associative reductions such as integer sum, minimum, maximum, and bitwise operations. Support floating-point sum with documented numerical tolerance. A custom reduction should require an explicitly supported associative operator.

Preferred initial algorithm:

1. Locate the beginning and end of the range.
2. Process the corresponding values in parallel.
3. Use a hierarchical GPU reduction.
4. Return the aggregate with minimal host-device synchronization.

Do not implement the reduction as a sequential loop over the selected keys.

---

## 6. Parallelization Strategy

The implementation should use GPU parallel primitives as building blocks rather than translating CPU pointer operations directly into GPU kernels.

| Workload | Initial parallel strategy |
|---|---|
| Sorting | GPU radix sort or another suitable device sorting primitive |
| Duplicate removal | Adjacent-key comparison followed by compaction |
| Merging | Parallel merge or merge path |
| Membership | Batched binary search or merge-based matching |
| Deletion | Parallel marking and stream compaction |
| Partitioning | Prefix sums, scans, and index calculations |
| Range reduction | Hierarchical reduction |
| Node construction | Parallel scatter into contiguous storage |
| Validation | Parallel adjacent-key and boundary checks |

Use the simplest correct implementation first. Introduce custom kernels when profiling demonstrates that a library primitive or intermediate buffer is a bottleneck.

### 6.1 Memory Layout

The primary goal is to reduce irregular memory accesses.

Requirements:
- Store adjacent keys contiguously wherever practical.
- Align storage to suitable device-memory boundaries.
- Avoid per-key heap allocations.
- Avoid unnecessary pointer chasing.
- Use coalesced loads and stores.
- Minimize temporary buffers and redundant copies.
- Measure memory usage, bandwidth, and kernel occupancy where tooling permits.

### 6.2 GPU Execution

The design should expose enough parallel work to use the GPU effectively.

Requirements:
- Support large batches without a serial per-key bottleneck.
- Minimize kernel-launch overhead for small batches where possible.
- Avoid unnecessary CPU synchronization.
- Reuse allocated buffers when practical.
- Check GPU allocation and kernel execution errors.
- Handle workloads that exceed a single allocation when feasible, or report explicit size limits.

A GPU implementation is not automatically faster than a CPU implementation. Small batches may be dominated by launch overhead and data transfer, while large batches may benefit from parallel execution. Benchmark both regimes.

---

## 7. Duplicate Semantics and Data Model

The initial version should implement a unique-key set index.

**Set mode requirements:**
- Each key occurs at most once.
- Keys follow a documented total ordering.
- Insertion of an existing key does not increase logical size.
- Deletion removes the key if present.
- Union produces the mathematical set union.

If key-value mapping is later supported, define a separate policy for duplicate keys and conflicting values.

Do not leave duplicate behavior dependent on sorting stability, execution order, or thread scheduling. All public operations must follow the same documented semantics.

The first version should use a clear, well-defined key type, such as 32-bit or 64-bit integers. Generalize to additional types only after the initial implementation and benchmarks are stable.

---

## 8. Correctness and Testing

Every operation must be tested against a simple trusted CPU reference implementation.

### 8.1 Functional Tests

Required cases:
- Empty index and empty batch.
- Single-key index.
- Sorted and unsorted build inputs.
- Duplicate-heavy inputs.
- Insertion of existing and new keys.
- Deletion of present and missing keys.
- Union of disjoint indexes.
- Union of identical indexes.
- Union of partially overlapping indexes.
- Split below the minimum and above the maximum.
- Split at an existing key and at an absent key.
- Empty and full-range queries.
- Empty and single-element reductions.
- Multiple nodes and partially filled final nodes.
- Repeated operation sequences.
- Very large inputs and allocation failure paths.

### 8.2 Invariant Checks

The `validate()` method should check:

1. All keys are ordered within each node.
2. Adjacent nonempty nodes obey the global ordering invariant.
3. Node counts do not exceed capacity.
4. Logical size equals the number of valid keys.
5. Every key is represented exactly once in set mode.
6. Node boundaries and offsets are valid.
7. All allocated buffers are large enough for their logical contents.
8. Results remain correct after every supported mutation.

### 8.3 Randomized Testing

Generate random inputs with varied:
- Dataset sizes.
- Batch sizes.
- Key distributions.
- Duplicate rates.
- Deletion hit rates.
- Overlap between indexes.
- Split pivots.
- Range widths.

Run each operation against the CPU reference and compare the logical output. Include reproducible random seeds so failures can be repeated and debugged.

Use GPU memory checking and race-detection tools available in the development environment when practical.

---

## 9. Benchmarking Requirements

Performance claims must be based on repeatable measurements.

### 9.1 Baselines

Compare against:

1. An optimized CPU implementation of the same ordered-set semantics.
2. Standard CPU sorting and merging algorithms.
3. A GPU flat sorted-array baseline using the same GPU primitives where applicable.
4. The proposed multi-key-node implementation.

The flat-array baseline is especially important. It will help determine whether multi-key nodes offer an advantage beyond the benefits of GPU sorting, merging, and compaction alone.

If a traditional CPU skip list is included, document its implementation and tuning. Do not compare against an intentionally inefficient CPU implementation.

### 9.2 Dataset Sizes

Start with datasets ranging from thousands to millions of keys, subject to available hardware memory.

Include larger datasets when the device supports them. Measure both total index size and incoming batch size independently.

### 9.3 Metrics

Report:
- Total operation latency.
- Throughput in keys per second.
- GPU kernel time.
- End-to-end time, including transfers and allocation where applicable.
- Memory consumption and temporary allocation size.
- Speedup over each baseline.
- Scaling with dataset size and batch size.
- Performance under varying duplicate and overlap rates.

For each benchmark, record hardware, compiler and library versions, optimization flags, input distribution, batch size, warm-up policy, and number of repetitions.

Report median latency and a measure of variability. Separate device-only timings from end-to-end timings; otherwise, performance claims may hide transfer costs.

### 9.4 Benchmark Matrix

| Experiment | Dataset | Batch size | Primary question |
|---|---|---|---|
| Build | Small to large | Entire input | How quickly can the index be constructed? |
| Insert | Small to large | Small to large | When does batch merging outperform repeated CPU insertion? |
| Delete | Small to large | Small to large | How efficiently can retained keys be compacted? |
| Union | Two indexes | Both inputs vary | How does overlap affect performance? |
| Split | Small to large | Single pivot | What is the cost of partitioning? |
| Range reduce | Small to large | Range width varies | How does throughput scale with range size? |
| Node tuning | Fixed datasets | Fixed batches | Which node capacity provides the best trade-off? |

Do not report a speedup without identifying the exact baseline, input size, and timing scope.

---

## 10. Implementation Plan

Implement the project in the following phases. Each phase must pass its correctness tests before the next phase begins.

### Phase 1: Repository and Environment Inspection

The coding agent must:
- Inspect the existing repository and its build configuration.
- Identify the language, GPU programming model, compiler, and available libraries.
- Inspect the target GPU and available memory.
- Preserve existing conventions where practical.
- Document any missing dependencies or environment limitations.

Do not replace the project's language or build system without a clear reason.

### Phase 2: CPU Reference and Data Model

Implement:
- Unique-key set semantics.
- CPU reference operations.
- Node capacity and representation.
- A validator for ordering and metadata.
- Unit tests for the complete operation contract.

This phase establishes correctness before GPU complexity is introduced.

### Phase 3: GPU Build and Representation

Implement:
- Device allocation and ownership.
- GPU sorting.
- Duplicate removal.
- Node construction.
- Logical size and metadata.
- GPU validation.

Benchmark construction and memory usage against the flat-array baseline.

### Phase 4: Core Batch Operations

Implement in this order:
1. `bulk_insert`
2. `bulk_delete`
3. `union`
4. `split`
5. `range_query`
6. `range_reduce`

Prefer mature GPU primitives for sorting, merging, scans, and reductions. Add custom kernels only where necessary.

### Phase 5: Benchmark Suite

Create reproducible benchmarks that compare the GPU implementation, flat-array baseline, and CPU reference.

Include warm-up runs, repeated measurements, correctness verification, and both device-only and end-to-end timing.

### Phase 6: Optimization

Use profiling results to investigate:
- Node capacity.
- Memory coalescing.
- Temporary-buffer usage.
- Kernel-launch overhead.
- Merge and compaction performance.
- Allocation overhead.
- Occupancy and memory bandwidth.
- Small-batch versus large-batch behavior.

Every optimization must preserve correctness and demonstrate a measurable benefit.

### Phase 7: Documentation and Final Evaluation

Document the data structure, API, supported hardware, build instructions, benchmark methodology, known limitations, and measured results.

The final report must distinguish verified performance improvements from hypotheses and future work.

---

## 11. Suggested Repository Structure

Adapt this layout to the repository rather than restructuring existing code unnecessarily.

```text
gpu-ordered-index/
├── README.md
├── CMakeLists.txt
├── include/
│   └── gpu_ordered_index.cuh
├── src/
│   ├── index.cu
│   ├── build.cu
│   ├── bulk_insert.cu
│   ├── bulk_delete.cu
│   ├── union.cu
│   ├── split.cu
│   ├── range_query.cu
│   └── range_reduce.cu
├── reference/
│   └── cpu_ordered_set.cpp
├── tests/
│   ├── test_build.cpp
│   ├── test_insert.cpp
│   ├── test_delete.cpp
│   ├── test_union.cpp
│   ├── test_split.cpp
│   └── test_range_reduce.cpp
├── benchmarks/
│   ├── benchmark_build.cpp
│   ├── benchmark_updates.cpp
│   └── benchmark_operations.cpp
└── docs/
    ├── architecture.md
    └── benchmarks.md
```

This is a suggested organization, not a requirement to create empty files in advance. Prefer a small, working project over a large collection of unimplemented modules.

---

## 12. Definition of Done

The initial project is complete when:

- [ ] The data structure builds successfully in the target environment.
- [ ] The documented node and ordering invariants are enforced.
- [ ] The CPU reference implementation passes all correctness tests.
- [ ] GPU build produces the same logical result as the CPU reference.
- [ ] Bulk insertion and deletion work correctly for duplicates and missing keys.
- [ ] Union and split satisfy the specified set semantics.
- [ ] Range queries return correctly bounded results.
- [ ] Range reductions match the reference within the documented numerical tolerance.
- [ ] Randomized and edge-case tests pass.
- [ ] GPU errors and allocation failures are handled explicitly.
- [ ] Benchmarks measure device-only and end-to-end performance.
- [ ] Results include comparisons against a flat GPU array and a CPU baseline.
- [ ] The documentation explains the trade-offs and measured performance.

A performance improvement is not a prerequisite for declaring the prototype functionally complete. However, demonstrating a meaningful advantage on the intended batch workloads is necessary to validate the project's main hypothesis.

---

## 13. Instructions for the Coding Agent

Act as a systems engineer implementing a research prototype. Work incrementally, prioritize correctness, and avoid premature optimization.

Before writing substantial code:

1. Inspect the repository, available hardware, build system, and existing dependencies.
2. Identify constraints that affect implementation.
3. Write down the proposed architecture and operation semantics.
4. Implement the smallest complete vertical slice: CPU reference, GPU build, validation, and a correctness test.
5. Compile and run the implementation before expanding the API.

After each phase:
- Run the relevant tests.
- Fix correctness and memory-safety issues before proceeding.
- Record implementation decisions and any deviations from this specification.
- Benchmark only after verifying correctness.
- Do not invent performance results or claim an optimization succeeded without measurements.

Use existing GPU libraries when appropriate. Avoid building a custom sorting library, memory allocator, or general-purpose concurrency framework unless profiling or a specific project requirement justifies it.

Keep the first implementation simple. Establish a correct and measurable batch-oriented index before exploring advanced hierarchical layouts, incremental updates, adaptive node capacities, or specialized GPU kernels.

**Primary success criterion:** Demonstrate that a contiguous, multi-key ordered index with GPU-parallel batch operations can provide a measurable throughput or end-to-end latency advantage over appropriate CPU and GPU baselines for large batch-oriented workloads.


The most important experiment is comparing three implementations:
- CPU ordered set: a strong CPU baseline using appropriate sorting and merging algorithms.
- GPU flat array: sorted keys operated on using GPU sorting, merging, scans, and reductions.
- GPU multi-key index: your proposed representation with node boundaries and any additional routing metadata.