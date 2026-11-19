// bwprobe — how fast can this GPU actually move memory?
//
// A state-vector gate pass reads every amplitude once and writes it once. So the ceiling for any
// gate kernel is "read N bytes + write N bytes" at the best achievable DRAM bandwidth. The spec
// sheet says 448 GB/s for the RTX 3070 Ti Laptop; this measures what is really reachable.
//
// Buffer size defaults to a 28-qubit complex64 state (2^28 * 8 B = 2 GiB). Pass a qubit count to
// change it:  bwprobe 24

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call)                                                                    \
  do {                                                                                 \
    cudaError_t err_ = (call);                                                         \
    if (err_ != cudaSuccess) {                                                         \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
      std::exit(1);                                                                    \
    }                                                                                  \
  } while (0)

// Read + write, 16 bytes per thread per iteration (two complex64 amplitudes).
__global__ void copy_f4(const float4* __restrict__ in, float4* __restrict__ out, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    out[i] = in[i];
}

// In place read-modify-write, the same traffic shape as applying a gate to the state.
__global__ void scale_inplace_f4(float4* __restrict__ buf, size_t n, float s) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x) {
    float4 v = buf[i];
    v.x *= s; v.y *= s; v.z *= s; v.w *= s;
    buf[i] = v;
  }
}

template <typename Launch>
static float median_ms(Launch launch, int reps) {
  cudaEvent_t a, b;
  CHECK(cudaEventCreate(&a));
  CHECK(cudaEventCreate(&b));
  for (int i = 0; i < 3; ++i) launch();  // warm-up (clocks ramp, caches settle)
  CHECK(cudaDeviceSynchronize());
  std::vector<float> t(reps);
  for (int r = 0; r < reps; ++r) {
    CHECK(cudaEventRecord(a));
    launch();
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    CHECK(cudaEventElapsedTime(&t[r], a, b));
  }
  CHECK(cudaGetLastError());
  CHECK(cudaEventDestroy(a));
  CHECK(cudaEventDestroy(b));
  std::sort(t.begin(), t.end());
  return t[reps / 2];
}

int main(int argc, char** argv) {
  const int qubits = argc > 1 ? std::atoi(argv[1]) : 28;
  const size_t amps = size_t(1) << qubits;
  const size_t bytes = amps * 8;           // complex64
  const size_t n4 = bytes / sizeof(float4);
  const int reps = 20;

  cudaDeviceProp p{};
  CHECK(cudaGetDeviceProperties(&p, 0));
  int mem_clock_khz = 0, bus_width = 0;
  CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0));
  CHECK(cudaDeviceGetAttribute(&bus_width, cudaDevAttrGlobalMemoryBusWidth, 0));
  // Memory clock is reported as the base clock; GDDR6 transfers 2x per reported clock here.
  const double spec_gbs = 2.0 * mem_clock_khz * 1e3 * (bus_width / 8) / 1e9;
  size_t free_b = 0, total_b = 0;
  CHECK(cudaMemGetInfo(&free_b, &total_b));

  std::printf("GPU            %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
  std::printf("Memory         %.2f GiB total, %.2f GiB free, %d-bit bus\n", total_b / 1073741824.0,
              free_b / 1073741824.0, bus_width);
  std::printf("Spec peak      %.0f GB/s (from driver-reported clock)\n", spec_gbs);
  std::printf("State          %d qubits complex64 = %.2f GiB\n\n", qubits, bytes / 1073741824.0);

  const bool two_bufs = 2 * bytes + (256ull << 20) < free_b;
  float4 *a = nullptr, *b = nullptr;
  CHECK(cudaMalloc(&a, bytes));
  CHECK(cudaMemset(a, 0, bytes));
  if (two_bufs) {
    CHECK(cudaMalloc(&b, bytes));
    CHECK(cudaMemset(b, 0, bytes));
  }

  const int block = 256;
  const int grid = p.multiProcessorCount * 32;  // grid-stride: enough blocks to fill every SM
  const double moved = 2.0 * bytes;             // one read + one write of the whole buffer

  auto report = [&](const char* name, float ms) {
    const double gbs = moved / (ms * 1e-3) / 1e9;
    std::printf("%-22s %8.3f ms  %7.1f GB/s  %5.1f%% of spec\n", name, ms, gbs, 100.0 * gbs / spec_gbs);
  };

  if (two_bufs) {
    report("cudaMemcpy D2D", median_ms([&] { CHECK(cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice)); }, reps));
    report("copy kernel (float4)", median_ms([&] { copy_f4<<<grid, block>>>(a, b, n4); }, reps));
  } else {
    std::printf("(skipping two-buffer tests: not enough free memory for 2 x %.2f GiB)\n", bytes / 1073741824.0);
  }
  report("in-place RMW (float4)", median_ms([&] { scale_inplace_f4<<<grid, block>>>(a, n4, 1.0f); }, reps));

  std::printf("\nThe best of these is the practical ceiling. A 1-qubit gate at %d qubits can't beat\n"
              "%.2f ms; M2/M3 kernels are scored as a %% of the in-place RMW figure.\n",
              qubits, moved / (spec_gbs * 1e9) * 1e3);

  CHECK(cudaFree(a));
  if (b) CHECK(cudaFree(b));
  return 0;
}
