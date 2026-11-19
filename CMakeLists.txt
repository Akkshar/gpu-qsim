cmake_minimum_required(VERSION 3.28)

# RTX 3070 Ti Laptop = Ampere GA104 = sm_86. Override with -DCMAKE_CUDA_ARCHITECTURES=... elsewhere.
# Must come BEFORE project(): project() fills in the compiler's default (sm_75 for CUDA 13), and
# until 2026-11-20 that silently made every kernel a Turing build JIT-compiled by the driver.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(gpu_qsim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# cudart_static is built against the static CRT (LIBCMT); match it so the linker doesn't mix CRTs (LNK4098).
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# -lineinfo lets Nsight Compute map hot spots back to source lines.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# ---- core library: circuit IR + parser + CPU reference simulator (CUDA backends join in M2) ----
add_library(qsim_core STATIC
  src/io/circuit_io.cpp
  src/cpu/gates.cpp
  src/cpu/statevector.cpp
  src/fusion/fuse.cpp)
target_include_directories(qsim_core PUBLIC include)
if(MSVC)
  target_compile_options(qsim_core PRIVATE /W4 /permissive- /utf-8)
else()
  target_compile_options(qsim_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- CUDA backend (M2) ----
find_package(CUDAToolkit REQUIRED)
add_library(qsim_cuda STATIC src/cuda/statevector.cu)
target_link_libraries(qsim_cuda PUBLIC qsim_core CUDA::cudart_static)
target_compile_options(qsim_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3 -Xcompiler=/utf-8 -Xcompiler=/Zc:preprocessor>)
# Print registers / stack / spills per kernel at build time ("ptxas info" lines). Spills are the
# first thing to check when a kernel is slower than memory bandwidth says it should be.
option(QSIM_PTXAS_INFO "Print per-kernel register and spill usage" ON)
if(QSIM_PTXAS_INFO)
  target_compile_options(qsim_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
endif()

add_executable(qsim_cli tools/qsim_cli.cpp)
target_link_libraries(qsim_cli PRIVATE qsim_core qsim_cuda)

# M0: memory-bandwidth probe — measures the practical DRAM ceiling the simulator is scored against.
add_executable(bwprobe tools/bwprobe.cu)

# ---- tests ----
option(QSIM_BUILD_TESTS "Build the Catch2 test suite" ON)
if(QSIM_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  FetchContent_MakeAvailable(Catch2)

  add_executable(qsim_tests
    tests/test_bits_gates.cpp
    tests/test_parser.cpp
    tests/test_statevector.cpp
    tests/test_cuda.cpp        # [cuda] tests need an NVIDIA GPU; exclude with "~[cuda]"
    tests/test_fusion.cpp)
  target_link_libraries(qsim_tests PRIVATE qsim_core qsim_cuda Catch2::Catch2WithMain)

  enable_testing()
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(Catch)
  catch_discover_tests(qsim_tests DISCOVERY_MODE PRE_TEST)
endif()
