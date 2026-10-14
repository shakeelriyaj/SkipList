# Implementation Plan: GPU-Accelerated Batched Ordered Index

This plan turns `readme.md` into a concrete, ordered sequence of work. Development happens
on a laptop without CUDA; compilation, testing, and benchmarking happen on the school GPU
cluster VM. Each phase ends with a checkpoint that must pass on the cluster before the next
phase starts.

---

## 0. Fixed Decisions (made up front so later phases don't re-litigate them)

| Decision | Choice | Rationale |
|---|---|---|
| Language / model | C++17 + CUDA (nvcc) | Readme default for an empty repo. |
| Build system | CMake >= 3.18 with `CUDA` language enabled, `CMAKE_CUDA_ARCHITECTURES` configurable | Standard; cluster GPU arch unknown until Phase 1. |
| GPU primitives | CUB (sort, scan, select, reduce, merge-sort) and Thrust (`merge`, `set_union`, `lower_bound`, `unique`) | Both ship with the CUDA toolkit; no external downloads needed on a locked-down cluster. |
| Key type | `uint32_t` first, `uint64_t` via template parameter, tested for both | Readme sec. 7. Radix sort supports both natively. |
| Data model | Unique-key **set**; values optional and off by default | Readme sec. 7. Map mode is future work. |
| Node capacity | Compile-time template parameter `NodeCap`, default 256; sweep 64/128/256/512/1024 in Phase 6 | Fixed capacity gives O(1) node indexing. |
| Node layout (v1) | Fully packed: node `i` owns `keys[i*NodeCap, (i+1)*NodeCap)`; only the last node may be partial | Simplest correct baseline; node boundaries are implicit. |
| Routing layer (v1) | `node_max[]`: one key per node (the node's largest key). Searched by binary search to locate a node, then a second binary search inside the node | This is the "skip" level. It is what distinguishes the index from the flat-array baseline. |
| Node layout (v2, Phase 6 only) | Nodes with slack (fill factor < 1) so that inserts/deletes touch only affected nodes instead of re-merging the whole array | Only if Phase 5 measurements show full-merge is the bottleneck. |
| Test framework | Tiny self-contained header (`tests/check.hpp`) with `CHECK`/`REQUIRE` and a test registry. No GoogleTest/Catch2 | Cluster may have no internet; zero dependencies beyond the CUDA toolkit. |
| Benchmark timing | `cudaEvent` pairs for device-only time; `std::chrono::steady_clock` for end-to-end. Median of N reps + IQR | Readme sec. 9.3. |
| Error handling | `CUDA_CHECK` macro throws `gpu_index::cuda_error`; allocation failures surface as exceptions, never silent | Readme sec. 4 item 8. |
| Streams | Every public op takes an optional `cudaStream_t` (default stream 0). Ops are stream-ordered; no internal `cudaDeviceSynchronize`. Results that return host scalars (`size`, `range_reduce`) sync on that stream only | Readme sec. 4 item 3. |
| Mutation vs. new index | `build`, `bulk_insert`, `bulk_delete` mutate `*this`. `union_with`, `split`, `range_query` return new objects. Inputs are never modified | Readme sec. 5. |

### Operation semantics (frozen contract)

- **Duplicates on build / insert:** collapsed; logical size counts unique keys.
- **Delete of a missing key:** silently ignored. Repeated delete keys are equivalent to one.
- **Union:** mathematical set union; equal keys kept once.
- **Split at pivot p:** left = `{k < p}`, right = `{k >= p}`. Pivot absent or out of range is fine; one side may be empty.
- **Range query / reduce on `[lo, hi)`:** half-open. If `lo >= hi` the range is empty.
- **Range reduce:** supported ops = `sum`, `min`, `max`, `bit_and`, `bit_or`, `bit_xor` over keys (set mode). Identity returned for an empty range (`0`, `max`, `min`, all-ones, `0`, `0`). Integer only in v1; float sum deferred with documented tolerance.
- **Ordering:** unsigned integer `<`.

---

## 1. Repository Layout (built incrementally, not pre-created)

```
SkipList/
├── readme.md                         (existing spec)
├── CMakeLists.txt
├── cmake/                            CUDA arch detection helper
├── include/gpu_index/
│   ├── common.cuh                    CUDA_CHECK, cuda_error, DeviceBuffer<T>, DeviceSpan<T>
│   ├── reduce_ops.hpp                reduction operator tags + identities
│   ├── validation.hpp                ValidationResult
│   ├── cpu_ordered_set.hpp           CPU reference (header-only, std::vector based)
│   ├── flat_sorted_array.cuh         GPU flat-array baseline
│   └── gpu_ordered_index.cuh         GpuOrderedIndex<Key, NodeCap> public API
├── src/
│   ├── device_buffer.cu
│   ├── flat_sorted_array.cu
│   ├── index_build.cu
│   ├── index_insert.cu
│   ├── index_delete.cu
│   ├── index_union.cu
│   ├── index_split.cu
│   ├── index_range.cu                range_query + range_reduce + contains
│   └── index_validate.cu
├── tests/
│   ├── check.hpp                     minimal test harness
│   ├── random_gen.hpp                seeded generators (uniform, skewed, clustered, dup-heavy)
│   ├── test_cpu_reference.cpp
│   ├── test_build.cu
│   ├── test_insert_delete.cu
│   ├── test_union_split.cu
│   ├── test_range.cu
│   ├── test_validate.cu
│   └── test_randomized.cu            property tests vs CPU reference, all ops, fixed seeds
├── benchmarks/
│   ├── bench_common.hpp              timers, stats, CSV writer, hardware/version capture
│   ├── bench_build.cu
│   ├── bench_updates.cu              insert + delete
│   ├── bench_operations.cu           union, split, range_query, range_reduce
│   ├── bench_node_tuning.cu
│   └── scripts/run_all.sh, plot.py   run matrix and produce tables/plots from CSV
└── docs/
    ├── implementation_plan.md        (this file)
    ├── architecture.md               written in Phase 2, updated through Phase 6
    ├── environment.md                written in Phase 1 on the cluster
    └── benchmarks.md                 written in Phase 7
```

---

## 2. Phases

### Phase 1 — Environment inspection (on the cluster, ~half a day)

Deliverable: `docs/environment.md`, a `CMakeLists.txt` that configures and compiles a hello-world CUDA target.

1. Record: `nvidia-smi`, `nvcc --version`, GPU model, compute capability, device memory, host CPU, RAM, `cmake --version`, host compiler, CUB/Thrust versions (from toolkit headers), whether `compute-sanitizer` and `nsys`/`ncu` exist, module-load commands, and job-submission constraints (Slurm? interactive?).
2. Set `CMAKE_CUDA_ARCHITECTURES` to the detected arch; add `-O3 -lineinfo` for Release and `-G` for Debug.
3. Confirm a trivial CUB radix sort of 1M keys compiles and runs. This validates the whole toolchain before any real code.

Exit criterion: `cmake -B build && cmake --build build && ./build/smoke` passes on the VM.

### Phase 2 — CPU reference and data model (~1–2 days, can be done on the laptop)

Deliverable: `cpu_ordered_set.hpp`, `reduce_ops.hpp`, `validation.hpp`, `tests/check.hpp`, `tests/random_gen.hpp`, `tests/test_cpu_reference.cpp`, `docs/architecture.md`.

1. `CpuOrderedSet<Key>`: sorted `std::vector<Key>`. Every op implemented with `std::sort` + `std::unique`, `std::merge`/`std::set_union`, `std::set_difference`, `std::lower_bound`, `std::accumulate`. It is the oracle, so it must be obviously correct, not fast. (A separate *optimized* CPU baseline is a benchmark concern, Phase 5.)
2. Define the node model on the CPU side too: a `partition_into_nodes(sorted_keys, NodeCap)` helper that produces `node_counts` and `node_max`, plus `validate_layout(...)` implementing the eight invariant checks from readme sec. 8.2. This same validator is reused on host-copied GPU state in Phase 3 before a device-side validator exists.
3. Tests cover every functional case in readme sec. 8.1 against hand-computed expectations, plus the semantics table above.
4. `docs/architecture.md`: layout diagram, invariants, op contracts, stream/ownership rules.

Exit criterion: all CPU tests pass with `-fsanitize=address,undefined` on the laptop (these files are plain C++).

### Phase 3 — GPU build and representation (~2–3 days)

Deliverable: `common.cuh`, `DeviceBuffer`, `FlatSortedArray`, `GpuOrderedIndex::build/size/validate/contains`, `test_build.cu`, `test_validate.cu`, first `bench_build.cu`.

1. `DeviceBuffer<T>`: RAII `cudaMalloc`/`cudaFree`, move-only, `resize` with capacity reuse, `copy_from_host`/`to_host` helpers. `DeviceSpan<T>` is a non-owning pointer+size.
2. `FlatSortedArray<Key>` baseline: `build` = CUB radix sort then CUB `DeviceSelect::Unique`. This class later gets insert/delete/union/split/range by the same merge-based algorithms so it is a fair baseline (readme sec. 9.1 item 3).
3. `GpuOrderedIndex<Key, NodeCap>`:
   - storage: `DeviceBuffer<Key> keys` (padded to a multiple of `NodeCap`), `DeviceBuffer<Key> node_max`, `size_t logical_size`, `size_t num_nodes`.
   - `build`: sort, unique, then one kernel `fill_node_max` (thread per node reads its last valid key). Allocation sized from actual unique count.
   - `contains(queries)`: kernel, one thread per query: binary search `node_max` to pick a node, then binary search inside the node. This is the first place the routing layer earns its keep and gives a measurable contrast with the flat baseline's single wide binary search.
   - `validate()`: device kernels for (1) intra-node order, (2) cross-node order via `node_max[i] < keys[(i+1)*NodeCap]`, (3) count/capacity, (4) `node_max` consistency; plus host checks for size/allocation bookkeeping. Results reduced to a `ValidationResult{ok, first_error_code, first_error_index}`.
4. Tests: build from empty / 1 / unsorted / dup-heavy / exactly-NodeCap / NodeCap+1 / 10M keys; compare `to_host()` with CPU reference; `validate()` after each; run under `compute-sanitizer --tool memcheck`.
5. Benchmark build time and peak memory vs flat baseline at 1e4…1e7 keys.

Exit criterion: GPU build matches CPU reference on all tests and randomized seeds; memcheck clean.

### Phase 4 — Core batch operations (~1 week), in this order

Each op is implemented first on `FlatSortedArray` (baseline), then on `GpuOrderedIndex` (initially by delegating to the same merge pipeline plus `fill_node_max`). This keeps the two implementations semantically identical, so the Phase 6 comparison measures *only* the representation difference.

1. **bulk_insert**: sort+unique batch → `thrust::set_union` (or CUB merge + adjacent-unique + select) into a fresh buffer → swap buffers → rebuild `node_max`. Buffer reuse: keep the old buffer as scratch for the next op.
2. **bulk_delete**: sort+unique batch → `thrust::set_difference` (keys \ batch) → swap → rebuild `node_max`. Alternative measured in Phase 6: mark via batched binary search + `DeviceSelect::Flagged`.
3. **union_with**: `thrust::set_union` of the two key arrays into a new index.
4. **split**: `lower_bound(pivot)` on device → two `cudaMemcpyAsync` into two new indexes → `fill_node_max` on each. Original untouched.
5. **range_query**: two lower bounds (on `node_max` then within node) → contiguous `cudaMemcpyAsync` into a `DeviceBuffer<Key>`.
6. **range_reduce**: locate `[begin,end)` positions → `cub::DeviceReduce::Reduce` with the selected operator over `keys + begin` → single scalar copied to host (one sync). Deterministic for integers because the operators are exact.

Tests per op: all readme sec. 8.1 cases, plus `test_randomized.cu` running 200+ seeded random operation sequences (mixed op types, sizes 0…1e6, duplicate rates 0…90%, delete hit rates 0…100%, overlap 0…100%, random pivots and ranges) against the CPU reference, validating after every mutation.

Exit criterion: all tests pass for `uint32_t` and `uint64_t`, memcheck clean, `validate()` never fails in the randomized suite.

### Phase 5 — Benchmark suite (~3 days)

Deliverable: `benchmarks/*`, `scripts/run_all.sh`, `plot.py`, raw CSV output committed under `docs/results/<gpu>/`.

1. **Baselines**: (a) `CpuOrderedSet` oracle, (b) an optimized CPU baseline: sorted `std::vector` with `std::sort`/`std::inplace_merge`/`std::set_union`, compiled `-O3 -march=native`; optionally multithreaded sort via `std::execution::par` if the cluster's libstdc++/TBB allows it (document either way), (c) `FlatSortedArray`, (d) `GpuOrderedIndex`.
2. **Matrix** (readme sec. 9.4): index sizes 1e4, 1e5, 1e6, 1e7 (and 1e8 if memory allows); batch sizes 1e2…1e6; overlap 0/50/100 %; delete hit rate 0/50/100 %; range widths 0.1/1/10/100 % of keys; NodeCap 64…1024.
3. **Protocol**: 3 warm-up runs, 10+ timed reps, report median and IQR; device-only (`cudaEvent`) and end-to-end (including H2D of the batch and any allocation) recorded separately; correctness of every benchmark output verified against the oracle before the timing is accepted; hardware/compiler/library/flags/seed captured in each CSV header.
4. Also record peak device memory (`cudaMemGetInfo` before/after) and temporary allocation size.

Exit criterion: `run_all.sh` produces the full CSV set and plots unattended in one job.

### Phase 6 — Optimization (open-ended; driven by Phase 5 data)

Only pursue items with a measured bottleneck. Candidate list, in expected order of payoff:

1. **Node capacity sweep** and pick a default from data.
2. **Localized updates (node slack)**: switch nodes to fill factor ~0.75 and implement insert as: batched search of each batch key's target node → per-node count via histogram/scan → nodes whose count + incoming <= NodeCap are updated in place by a block-per-node merge in shared memory; overflowing nodes fall back to a localized rebuild of just the affected node range. This is the main hypothesis test: does the node structure beat full re-merge for small-to-medium batches?
3. **Temporary buffer reuse**: a per-index scratch arena so CUB `temp_storage` and merge outputs are not re-`cudaMalloc`ed every op; measure allocation overhead first with `nsys`.
4. **Kernel fusion**: fuse unique + `fill_node_max`, or lower-bound + memcpy for range query.
5. **Small-batch path**: for batches below a threshold, a single fused kernel instead of sort+merge pipeline.
6. Profile with `ncu` for bandwidth/occupancy on the top three kernels and document findings.

Each optimization is a separate commit with before/after CSV in the message or `docs/benchmarks.md`.

### Phase 7 — Documentation and final evaluation (~2 days)

1. `docs/architecture.md` final update; `readme.md` gains build/run instructions and supported-hardware section.
2. `docs/benchmarks.md`: methodology, tables/plots, speedups each tagged with baseline + input size + timing scope, variance, and a clear **verified vs. hypothesis** split.
3. Known limitations: set-only, integer keys, single-GPU, single allocation limit, no concurrent writers.
4. Walk the readme sec. 12 checklist and tick items with pointers to the test or result that proves each.

---

## 3. Milestone Summary

| # | Milestone | Proof |
|---|---|---|
| M1 | Toolchain works on cluster | smoke sort runs |
| M2 | CPU reference + validator + tests | CPU tests pass under sanitizers |
| M3 | GPU build/contains/validate match reference | test_build, test_validate, memcheck |
| M4 | All six core ops on both GPU representations | randomized suite green for u32 and u64 |
| M5 | Reproducible benchmark CSVs and plots | one-command run |
| M6 | At least one data-backed optimization (node tuning + localized update attempt) | before/after CSV |
| M7 | Docs + final report | readme sec. 12 checklist complete |

---

## 4. Risks and Mitigations

- **Cluster toolkit is old (CUDA < 11.x)**: CUB may not ship with it. Mitigation: vendor CUB/Thrust headers as a git submodule or tarball in `third_party/`; check in Phase 1.
- **Multi-key index shows no advantage over flat array in v1**: expected, since v1 packed nodes are nearly the same layout. The hypothesis is tested by Phase 6 item 2. Report honestly either way.
- **Memory limits at 1e8 keys**: merge-based ops need ~3x key storage transiently. Document the limit and fail with an explicit exception.
- **Thrust `set_union` performance vs. custom merge-path**: measure; only write a custom merge kernel if Thrust is the top hotspot.
- **No GPU on the dev laptop**: keep CPU reference, validator, generators, and harness in pure C++ so most logic is testable locally; CUDA files are compiled only on the cluster.

---

## 5. First Concrete Steps

1. Phase 1 on the VM: capture environment, commit `docs/environment.md` and a compiling `CMakeLists.txt` with the smoke target.
2. Phase 2 on the laptop: `cpu_ordered_set.hpp`, validator, test harness, generators, CPU tests.
3. Phase 3 vertical slice on the VM: `DeviceBuffer`, `FlatSortedArray::build`, `GpuOrderedIndex::build/validate`, `test_build.cu`. Compile, run, memcheck, then expand.
