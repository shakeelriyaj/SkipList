cmake_minimum_required(VERSION 3.18)
project(gpu_ordered_index LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(GPU_INDEX_ENABLE_CUDA "Build CUDA targets when a CUDA compiler is available" ON)
option(GPU_INDEX_SANITIZE "Build host-only tests with AddressSanitizer and UBSan" OFF)

# Header-only host code: CPU reference, node layout, validation, reduction ops.
add_library(gpu_index_host INTERFACE)
target_include_directories(gpu_index_host INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

enable_testing()

function(gpu_index_add_host_test name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE gpu_index_host)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
  if(GPU_INDEX_SANITIZE)
    target_compile_options(${name} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(${name} PRIVATE -fsanitize=address,undefined)
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

gpu_index_add_host_test(test_cpu_reference tests/test_cpu_reference.cpp)

# CUDA targets: only configured when nvcc is found (e.g. on the GPU cluster).
if(GPU_INDEX_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
      else()
        set(CMAKE_CUDA_ARCHITECTURES 70)
      endif()
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER} (arch: ${CMAKE_CUDA_ARCHITECTURES})")

    set(GPU_INDEX_CUDA_FLAGS
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-lineinfo>)

    add_executable(smoke src/smoke.cu)
    target_link_libraries(smoke PRIVATE gpu_index_host)
    target_include_directories(smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_compile_options(smoke PRIVATE ${GPU_INDEX_CUDA_FLAGS})
    add_test(NAME smoke COMMAND smoke)
  else()
    message(STATUS "No CUDA compiler found: building host-only targets")
  endif()
endif()
