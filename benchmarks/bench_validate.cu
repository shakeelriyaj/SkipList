// bench_validate: compares the two ways to check a GpuOrderedIndex.
//
//   gpu   GpuOrderedIndex::validate()          one kernel on the device,
//                                              8-byte result copied back
//   host  GpuOrderedIndex::validate_on_host()  copies the whole index to the
//                                              host, runs validate_layout()
//
// Timing scope: wall-clock time of the full call as a user sees it,
// including kernel launch, device-to-host copies and stream synchronization.
// Index construction and input generation are NOT timed.
//
// Each size: 1 warm-up call per method, then --reps timed calls; the table
// reports median and min. Both methods must report the index valid, or the
// row is marked FAIL and the exit code is 1.
//
// Usage:
//   bench_validate [--sizes 1e4,1e5,1e6,1e7,1e8] [--reps 10] [--key 32|64] [--cap 256]
// --cap accepts 64, 128, 256, 512, 1024 (compile-time node capacities).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_index/common.cuh"
#include "gpu_index/gpu_ordered_index.cuh"
#include "random_gen.hpp"

namespace {

struct Options {
  std::vector<std::size_t> sizes = {10'000, 100'000, 1'000'000, 10'000'000, 100'000'000};
  int reps = 10;
  int key_bits = 32;
  std::size_t cap = 256;
};

std::vector<std::size_t> parse_sizes(const char* s) {
  std::vector<std::size_t> out;
  std::string item;
  for (const char* p = s;; ++p) {
    if (*p == ',' || *p == '\0') {
      if (!item.empty()) out.push_back(static_cast<std::size_t>(std::stod(item)));  // allows 1e7
      item.clear();
      if (*p == '\0') break;
    } else {
      item += *p;
    }
  }
  return out;
}

Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has_value = i + 1 < argc;
    if (a == "--sizes" && has_value) o.sizes = parse_sizes(argv[++i]);
    else if (a == "--reps" && has_value) o.reps = std::max(1, std::atoi(argv[++i]));
    else if (a == "--key" && has_value) o.key_bits = std::atoi(argv[++i]);
    else if (a == "--cap" && has_value) o.cap = std::strtoull(argv[++i], nullptr, 10);
    else {
      std::fprintf(stderr,
                   "usage: %s [--sizes 1e4,1e5,...] [--reps N] [--key 32|64] "
                   "[--cap 64|128|256|512|1024]\n",
                   argv[0]);
      std::exit(a == "-h" || a == "--help" ? 0 : 2);
    }
  }
  if (o.key_bits != 32 && o.key_bits != 64) throw std::invalid_argument("--key must be 32 or 64");
  if (o.cap != 64 && o.cap != 128 && o.cap != 256 && o.cap != 512 && o.cap != 1024) {
    throw std::invalid_argument("--cap must be one of 64, 128, 256, 512, 1024");
  }
  if (o.sizes.empty()) throw std::invalid_argument("--sizes is empty");
  return o;
}

struct Stats {
  double median_ms = 0, min_ms = 0;
};

// Warm-up once, then time `reps` calls. `ok` is false if any call reports a
// violation.
template <typename F>
Stats time_calls(int reps, bool& ok, F&& call) {
  ok = call().ok();
  std::vector<double> ms(reps);
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    ok = call().ok() && ok;
    ms[r] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  std::sort(ms.begin(), ms.end());
  const double median = reps % 2 ? ms[reps / 2] : 0.5 * (ms[reps / 2 - 1] + ms[reps / 2]);
  return {median, ms.front()};
}

std::string human_count(std::size_t n) {
  char buf[32];
  if (n >= 1'000'000'000 && n % 1'000'000'000 == 0) std::snprintf(buf, sizeof buf, "%zuB", n / 1'000'000'000);
  else if (n >= 1'000'000 && n % 1'000'000 == 0) std::snprintf(buf, sizeof buf, "%zuM", n / 1'000'000);
  else if (n >= 1'000 && n % 1'000 == 0) std::snprintf(buf, sizeof buf, "%zuK", n / 1'000);
  else std::snprintf(buf, sizeof buf, "%zu", n);
  return buf;
}

void print_header(const Options& o) {
  int device = 0, runtime = 0, driver = 0;
  cudaDeviceProp prop{};
  GPU_INDEX_CUDA_CHECK(cudaGetDevice(&device));
  GPU_INDEX_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  GPU_INDEX_CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  GPU_INDEX_CUDA_CHECK(cudaDriverGetVersion(&driver));

  std::printf("\nValidation benchmark: GpuOrderedIndex::validate() vs validate_on_host()\n");
  std::printf("  device      : %s (cc %d.%d, %.1f GiB, %d SMs), CUDA runtime %d.%d / driver %d.%d\n",
              prop.name, prop.major, prop.minor, prop.totalGlobalMem / double(1 << 30),
              prop.multiProcessorCount, runtime / 1000, (runtime % 1000) / 10, driver / 1000,
              (driver % 1000) / 10);
  std::printf("  index       : uint%d_t keys, node capacity %zu, uniform random input (seed 1)\n",
              o.key_bits, o.cap);
  std::printf("  timing      : wall-clock per call incl. launch, copies and sync; build not timed\n");
  std::printf("  repetitions : 1 warm-up + %d timed calls per method; median and min reported\n",
              o.reps);
  std::printf("  gpu         : one validation kernel, 8-byte result copied back\n");
  std::printf("  host        : copy whole index to host, then validate_layout() on the CPU\n");
  std::printf("  GB/s        : bytes the GPU validator reads (keys + node_max) / gpu median\n\n");
}

// Table layout: every row is built from these columns, so they always align.
struct Column {
  const char* title;
  const char* unit;
  int width;
};
const Column kColumns[] = {
    {"keys", "input", 6},          {"unique", "keys", 10},        {"nodes", "", 7},
    {"gpu median", "(ms)", 10},    {"gpu min", "(ms)", 10},       {"host median", "(ms)", 11},
    {"host min", "(ms)", 10},      {"speedup", "(x)", 7},         {"gpu GB/s", "", 8},
    {"valid", "", 5},
};
constexpr std::size_t kNumColumns = sizeof(kColumns) / sizeof(kColumns[0]);

void print_rule() {
  for (const Column& c : kColumns) std::printf("+%s", std::string(c.width + 2, '-').c_str());
  std::printf("+\n");
}

void print_row(const std::vector<std::string>& cells) {
  for (std::size_t i = 0; i < kNumColumns; ++i) {
    std::printf("| %*s ", kColumns[i].width, cells[i].c_str());
  }
  std::printf("|\n");
}

void print_table_header() {
  std::vector<std::string> titles, units;
  for (const Column& c : kColumns) {
    titles.push_back(c.title);
    units.push_back(c.unit);
  }
  print_rule();
  print_row(titles);
  print_row(units);
  print_rule();
}

std::string fmt(const char* f, double v) {
  char buf[64];
  std::snprintf(buf, sizeof buf, f, v);
  return buf;
}

template <typename Key, std::size_t Cap>
bool run_size(std::size_t n, int reps) {
  gpu_index::GpuOrderedIndex<Key, Cap> index;
  index.build_from_host(gen::generate_keys<Key>(n, gen::Distribution::Uniform, 1));

  bool gpu_ok = false, host_ok = false;
  const Stats gpu = time_calls(reps, gpu_ok, [&] { return index.validate(); });
  const Stats host = time_calls(reps, host_ok, [&] { return index.validate_on_host(); });

  const double bytes = double(index.size() + index.num_nodes()) * sizeof(Key);
  const double gbps = gpu.median_ms > 0 ? bytes / (gpu.median_ms * 1e6) : 0;
  const bool ok = gpu_ok && host_ok;
  print_row({human_count(n), std::to_string(index.size()), std::to_string(index.num_nodes()),
             fmt("%.3f", gpu.median_ms), fmt("%.3f", gpu.min_ms), fmt("%.3f", host.median_ms),
             fmt("%.3f", host.min_ms), fmt("%.0f", host.median_ms / gpu.median_ms),
             fmt("%.1f", gbps), ok ? "ok" : "FAIL"});
  std::fflush(stdout);
  return ok;
}

template <typename Key, std::size_t Cap>
bool run_all(const Options& o) {
  bool ok = true;
  for (std::size_t n : o.sizes) ok = run_size<Key, Cap>(n, o.reps) && ok;
  return ok;
}

template <typename Key>
bool dispatch_cap(const Options& o) {
  switch (o.cap) {
    case 64: return run_all<Key, 64>(o);
    case 128: return run_all<Key, 128>(o);
    case 256: return run_all<Key, 256>(o);
    case 512: return run_all<Key, 512>(o);
    case 1024: return run_all<Key, 1024>(o);
    default: throw std::invalid_argument("--cap must be one of 64, 128, 256, 512, 1024");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options o = parse_args(argc, argv);
    print_header(o);
    print_table_header();
    const bool ok = o.key_bits == 32 ? dispatch_cap<std::uint32_t>(o) : dispatch_cap<std::uint64_t>(o);
    print_rule();
    std::printf("speedup = host median / gpu median.  'unique' < 'keys' because random input has "
                "duplicates.\n\n");
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "bench_validate failed: %s\n", e.what());
    return 1;
  }
}
