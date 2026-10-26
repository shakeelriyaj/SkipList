#pragma once

// CUDA error handling and device memory ownership.
//
// GPU_INDEX_CUDA_CHECK(expr) throws gpu_index::cuda_error on failure, so
// allocation and launch errors are never silently ignored.
//
// DeviceBuffer<T>: move-only RAII owner of a cudaMalloc allocation with a
// logical size and a reusable capacity. Contents are uninitialized after
// allocation. All copies are stream-ordered; only to_host() synchronizes,
// and only on the given stream.
//
// DeviceSpan<T>: non-owning (pointer, size) view of device memory. The
// caller guarantees the memory outlives the span.

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gpu_index {

class cuda_error : public std::runtime_error {
 public:
  cuda_error(cudaError_t code, const char* expr, const char* file, int line)
      : std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + expr +
                           " failed: " + cudaGetErrorName(code) + " (" +
                           cudaGetErrorString(code) + ")"),
        code_(code) {}

  cudaError_t code() const noexcept { return code_; }

 private:
  cudaError_t code_;
};

namespace detail {
inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
  if (err != cudaSuccess) {
    (void)cudaGetLastError();  // clear non-sticky error state
    throw cuda_error(err, expr, file, line);
  }
}
}  // namespace detail

#define GPU_INDEX_CUDA_CHECK(expr) ::gpu_index::detail::cuda_check((expr), #expr, __FILE__, __LINE__)

// Checks for launch-configuration errors after a <<<>>> launch.
#define GPU_INDEX_CHECK_LAUNCH() GPU_INDEX_CUDA_CHECK(cudaGetLastError())

template <typename T>
class DeviceSpan {
 public:
  DeviceSpan() = default;
  DeviceSpan(T* data, std::size_t size) : data_(data), size_(size) {}

  T* data() const { return data_; }
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  operator DeviceSpan<const T>() const { return {data_, size_}; }

 private:
  T* data_ = nullptr;
  std::size_t size_ = 0;
};

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t n) { resize_discard(n); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      swap(other);
    }
    return *this;
  }

  ~DeviceBuffer() { release(); }

  // Sets the logical size to n. Reuses the allocation when n <= capacity;
  // otherwise reallocates. Contents are NOT preserved in either case.
  void resize_discard(std::size_t n) {
    if (n > capacity_) {
      if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::length_error("DeviceBuffer: allocation size overflow");
      }
      T* fresh = nullptr;
      GPU_INDEX_CUDA_CHECK(cudaMalloc(&fresh, n * sizeof(T)));
      release();
      data_ = fresh;
      capacity_ = n;
    }
    size_ = n;
  }

  void copy_from_host(const T* src, std::size_t n, cudaStream_t stream = 0) {
    resize_discard(n);
    if (n > 0) {
      GPU_INDEX_CUDA_CHECK(
          cudaMemcpyAsync(data_, src, n * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
  }

  void copy_from_host(const std::vector<T>& src, cudaStream_t stream = 0) {
    copy_from_host(src.data(), src.size(), stream);
  }

  // Copies the logical contents to the host and synchronizes `stream`.
  std::vector<T> to_host(cudaStream_t stream = 0) const {
    std::vector<T> out(size_);
    if (size_ > 0) {
      GPU_INDEX_CUDA_CHECK(
          cudaMemcpyAsync(out.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }
    GPU_INDEX_CUDA_CHECK(cudaStreamSynchronize(stream));
    return out;
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return size_; }
  std::size_t capacity() const { return capacity_; }
  bool empty() const { return size_ == 0; }

  DeviceSpan<T> span() { return {data_, size_}; }
  DeviceSpan<const T> span() const { return {data_, size_}; }

  void swap(DeviceBuffer& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
    std::swap(capacity_, other.capacity_);
  }

 private:
  void release() noexcept {
    if (data_) cudaFree(data_);  // errors cannot be reported from a destructor
    data_ = nullptr;
    size_ = capacity_ = 0;
  }

  T* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
};

}  // namespace gpu_index
