cmake_minimum_required(VERSION 3.28)

# RTX 3070 Ti Laptop = Ampere GA104 = sm_86. Override with -DCMAKE_CUDA_ARCHITECTURES=... elsewhere.
# Must come BEFORE project()/enable_language(CUDA): they fill in the compiler's default (sm_75 for
# CUDA 13), and until 2026-11-20 that silently made every kernel a Turing build JIT-compiled by the driver.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

project(gpu_qsim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# -fPIC everywhere: the static libraries are linked into the Python extension module (a shared
# library) on Linux, which fails with "relocation ... in read-only section" otherwise.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA is optional so the CPU reference, parser, fusion and their tests build anywhere (CI runners
# have no GPU). Default: on if a CUDA compiler is found.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  set(_qsim_cuda_default ON)
else()
  set(_qsim_cuda_default OFF)
endif()
option(QSIM_ENABLE_CUDA "Build the CUDA backend" ${_qsim_cuda_default})

if(QSIM_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # -lineinfo lets Nsight Compute map hot spots back to source lines.
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

# cudart_static is built against the static CRT (LIBCMT); match it so the linker doesn't mix CRTs (LNK4098).
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# ---- core library: circuit IR + parsers + CPU reference simulator + fusion ----
add_library(qsim_core STATIC
  src/io/circuit_io.cpp
  src/io/qasm.cpp
  src/cpu/gates.cpp
  src/cpu/statevector.cpp
  src/fusion/fuse.cpp)
target_include_directories(qsim_core PUBLIC include)
if(MSVC)
  target_compile_options(qsim_core PRIVATE /W4 /permissive- /utf-8)
else()
  target_compile_options(qsim_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---- CUDA backend (M2+) ----
if(QSIM_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(qsim_cuda STATIC src/cuda/statevector.cu)
  target_link_libraries(qsim_cuda PUBLIC qsim_core CUDA::cudart_static)
  target_compile_definitions(qsim_cuda PUBLIC QSIM_HAS_CUDA=1)
  if(MSVC)
    target_compile_options(qsim_cuda PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3 -Xcompiler=/utf-8 -Xcompiler=/Zc:preprocessor>)
  endif()
  # Print registers / stack / spills per kernel at build time ("ptxas info" lines). Spills are the
  # first thing to check when a kernel is slower than memory bandwidth says it should be.
  option(QSIM_PTXAS_INFO "Print per-kernel register and spill usage" ON)
  if(QSIM_PTXAS_INFO)
    target_compile_options(qsim_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
  endif()

  # M0: memory-bandwidth probe — measures the practical DRAM ceiling the simulator is scored against.
  add_executable(bwprobe tools/bwprobe.cu)
endif()

add_executable(qsim_cli tools/qsim_cli.cpp)
target_link_libraries(qsim_cli PRIVATE qsim_core $<$<BOOL:${QSIM_ENABLE_CUDA}>:qsim_cuda>)

# ---- Python bindings (M5): nanobind, found via `python -m nanobind --cmake_dir` ----
option(QSIM_BUILD_PYTHON "Build the qsim Python module (needs nanobind installed in Python_EXECUTABLE)" OFF)
if(QSIM_BUILD_PYTHON)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_qsim python/src/bindings.cpp)
  target_link_libraries(_qsim PRIVATE qsim_core $<$<BOOL:${QSIM_ENABLE_CUDA}>:qsim_cuda>)
  # Drop the built module straight into the package directory (python/qsim/).
  set_target_properties(_qsim PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/qsim"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/qsim"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/qsim")
endif()

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
    tests/test_qasm.cpp
    tests/test_statevector.cpp
    tests/test_fusion.cpp)
  if(QSIM_ENABLE_CUDA)
    target_sources(qsim_tests PRIVATE tests/test_cuda.cpp)  # [cuda] tests need an NVIDIA GPU
  endif()
  target_link_libraries(qsim_tests PRIVATE qsim_core $<$<BOOL:${QSIM_ENABLE_CUDA}>:qsim_cuda>
                        Catch2::Catch2WithMain)

  enable_testing()
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  include(Catch)
  catch_discover_tests(qsim_tests DISCOVERY_MODE PRE_TEST)
endif()
