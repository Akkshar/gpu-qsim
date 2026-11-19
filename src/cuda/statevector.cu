// CUDA backend, M2: one kernel launch per op, indexing identical to the CPU reference.
//
// Every op touches groups of 2^k amplitudes (k = number of targets). A group's base index has all
// control bits = 1 and all target bits = 0; thread g builds it from its group number by inserting a
// 0 at each target/control position (ascending) and OR-ing in the control mask. No thread ever
// skips work, and consecutive threads touch consecutive indices wherever the low bits allow.

#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>

#include <algorithm>
#include <cmath>
#include <cuda/std/functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "qsim/bits.hpp"
#include "qsim/cuda_statevector.hpp"
#include "qsim/random.hpp"

namespace qsim {
namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(e));
}
#define QSIM_CUDA(call) check((call), #call)

constexpr int kBlock = 256;
constexpr int kMaxTargets = 6;    // same limit as the parser's `mat`
constexpr int kMaxFixed = 40;     // targets + controls; states are at most 34 qubits
constexpr std::size_t kOpScratch = (std::size_t{1} << (2 * kMaxTargets)) * 16 + 64 * 8;  // matrix + offsets

// Same memory layout as std::complex<T> (two T's, real first), so buffers convert by memcpy.
template <class T>
struct alignas(2 * sizeof(T)) cplx {
  T re, im;
};

template <class T>
__host__ __device__ inline cplx<T> cmul(cplx<T> a, cplx<T> b) {
  return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
template <class T>
__host__ __device__ inline cplx<T> cadd(cplx<T> a, cplx<T> b) {
  return {a.re + b.re, a.im + b.im};
}

// Positions where a 0 gets inserted (targets and controls, ascending) plus the control mask.
// Passed by value as a kernel argument.
struct FixedBits {
  int count;
  int pos[kMaxFixed];
  std::uint64_t ctrl_mask;
};

__device__ inline std::uint64_t group_base(std::uint64_t g, const FixedBits& f) {
  for (int i = 0; i < f.count; ++i) g = insert_zero_bit(g, f.pos[i]);
  return g | f.ctrl_mask;
}

#define GRID_STRIDE(i, n)                                                         \
  for (std::uint64_t i = blockIdx.x * std::uint64_t(blockDim.x) + threadIdx.x; i < (n); \
       i += std::uint64_t(gridDim.x) * blockDim.x)

// One target: each thread owns one amplitude pair; the 2x2 matrix arrives in registers.
template <class T>
__global__ void apply_1t(cplx<T>* __restrict__ a, std::uint64_t groups, FixedBits f, std::uint64_t bit,
                         cplx<T> m00, cplx<T> m01, cplx<T> m10, cplx<T> m11) {
  GRID_STRIDE(g, groups) {
    const std::uint64_t i0 = group_base(g, f), i1 = i0 | bit;
    const cplx<T> a0 = a[i0], a1 = a[i1];
    a[i0] = cadd(cmul(m00, a0), cmul(m01, a1));
    a[i1] = cadd(cmul(m10, a0), cmul(m11, a1));
  }
}

// k = 2..5 targets, k known at compile time (M3). Loops fully unroll, so in[] lives in registers
// instead of a local-memory stack array, and every matrix index is a constant: the matrix and member
// offsets travel in the kernel's argument block (constant bank), read by all threads of a warp at
// once as a broadcast. __grid_constant__ stops the compiler copying the big argument to local memory.
// Argument size: k=5 in double is 32*32*16 + 32*8 = 16.6 KB, inside the 32 KB limit (CUDA 12.1+, sm_70+).
template <class T, int K>
struct DenseOp {
  cplx<T> m[(1 << K) * (1 << K)];  // row-major, already converted to T
  std::uint64_t offs[1 << K];      // member r of a group sits at base | offs[r]
};

template <class T, int K>
__global__ void apply_kt_fixed(cplx<T>* __restrict__ a, std::uint64_t groups, FixedBits f,
                               const __grid_constant__ DenseOp<T, K> op) {
  constexpr int D = 1 << K;
  GRID_STRIDE(g, groups) {
    const std::uint64_t base = group_base(g, f);
    cplx<T> in[D];
#pragma unroll
    for (int r = 0; r < D; ++r) in[r] = a[base | op.offs[r]];
#pragma unroll
    for (int r = 0; r < D; ++r) {
      cplx<T> s{T(0), T(0)};
#pragma unroll
      for (int c = 0; c < D; ++c) s = cadd(s, cmul(op.m[r * D + c], in[c]));
      a[base | op.offs[r]] = s;
    }
  }
}

// Same as apply_kt_fixed, but each block first copies the matrix into shared memory. For k = 5 the
// matrix (8 KB in f32, 16 KB in f64) is larger than the per-SM constant cache, so constant-bank
// reads keep missing; a shared-memory read where all lanes use the same address is a broadcast.
template <class T, int K>
__global__ void apply_kt_smem(cplx<T>* __restrict__ a, std::uint64_t groups, FixedBits f,
                              const __grid_constant__ DenseOp<T, K> op) {
  constexpr int D = 1 << K;
  __shared__ cplx<T> m[D * D];
  for (int i = threadIdx.x; i < D * D; i += blockDim.x) m[i] = op.m[i];
  __syncthreads();
  GRID_STRIDE(g, groups) {
    const std::uint64_t base = group_base(g, f);
    cplx<T> in[D];
#pragma unroll
    for (int r = 0; r < D; ++r) in[r] = a[base | op.offs[r]];
    // Only the inner loop unrolls (in[c] needs constant indices to stay in registers). Unrolling
    // the row loop too makes the compiler hoist all D*D shared loads and spill ~8-16 KB per thread.
#pragma unroll 1
    for (int r = 0; r < D; ++r) {
      cplx<T> s{T(0), T(0)};
#pragma unroll
      for (int c = 0; c < D; ++c) s = cadd(s, cmul(m[r * D + c], in[c]));
      a[base | op.offs[r]] = s;
    }
  }
}

// Any k up to 6, k known only at run time (M2 version, now only used for k = 6). Matrix and member
// offsets come from global memory; in[] is a 64-entry local-memory array (see BENCH.md, M3).
template <class T>
__global__ void apply_kt(cplx<T>* __restrict__ a, std::uint64_t groups, FixedBits f, int dim,
                         const cplx<T>* __restrict__ m, const std::uint64_t* __restrict__ offsets) {
  GRID_STRIDE(g, groups) {
    const std::uint64_t base = group_base(g, f);
    cplx<T> in[1 << kMaxTargets];
    for (int r = 0; r < dim; ++r) in[r] = a[base | offsets[r]];
    for (int r = 0; r < dim; ++r) {
      cplx<T> s{T(0), T(0)};
      for (int c = 0; c < dim; ++c) s = cadd(s, cmul(m[r * dim + c], in[c]));
      a[base | offsets[r]] = s;
    }
  }
}

// Keep the half where (i & bit) == keep, scaled; zero the other half.
template <class T>
__global__ void collapse(cplx<T>* __restrict__ a, std::uint64_t n, std::uint64_t bit, std::uint64_t keep, T scale) {
  GRID_STRIDE(i, n) {
    if ((i & bit) == keep) {
      a[i].re *= scale;
      a[i].im *= scale;
    } else {
      a[i] = cplx<T>{T(0), T(0)};
    }
  }
}

template <class T>
struct NormSq {
  __host__ __device__ double operator()(const cplx<T>& z) const {
    return double(z.re) * double(z.re) + double(z.im) * double(z.im);
  }
};

template <class T>
struct NormSqIfBit {
  const cplx<T>* a;
  std::uint64_t bit;
  __host__ __device__ double operator()(std::uint64_t i) const {
    return (i & bit) ? NormSq<T>{}(a[i]) : 0.0;
  }
};

unsigned grid_for(std::uint64_t work, unsigned cap) {
  const std::uint64_t blocks = (work + kBlock - 1) / kBlock;
  return static_cast<unsigned>(std::max<std::uint64_t>(1, std::min<std::uint64_t>(blocks, cap)));
}

// offs[r] = the target bits set in member r of a group. targets[j] maps to bit j of r (matrix order).
void member_offsets(const Op& op, std::uint64_t* offs) {
  const int k = op.arity();
  for (int r = 0; r < (1 << k); ++r) {
    std::uint64_t off = 0;
    for (int j = 0; j < k; ++j)
      if ((r >> j) & 1) off |= std::uint64_t{1} << op.targets[static_cast<std::size_t>(j)];
    offs[r] = off;
  }
}

template <class T, int K>
void launch_dense(cplx<T>* amps, std::uint64_t groups, const FixedBits& f, const Op& op, unsigned grid) {
  constexpr int D = 1 << K;
  DenseOp<T, K> d{};
  for (int i = 0; i < D * D; ++i) d.m[i] = {T(op.matrix[i].real()), T(op.matrix[i].imag())};
  member_offsets(op, d.offs);
  if constexpr (K >= 5) {
    apply_kt_smem<T, K><<<grid, kBlock>>>(amps, groups, f, d);  // matrix too big for the constant cache
  } else {
    apply_kt_fixed<T, K><<<grid, kBlock>>>(amps, groups, f, d);
  }
}

}  // namespace

template <class T>
CudaStateVector<T>::CudaStateVector(int num_qubits) : n_(num_qubits) {
  if (num_qubits < 1 || num_qubits > 34)
    throw std::invalid_argument("CudaStateVector: num_qubits must be 1..34, got " + std::to_string(num_qubits));
  int dev = 0, sms = 0;
  QSIM_CUDA(cudaGetDevice(&dev));
  QSIM_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  grid_cap_ = static_cast<unsigned>(sms) * 32;  // enough blocks to fill every SM (see tools/bwprobe.cu)

  const std::size_t bytes = size() * sizeof(cplx<T>);
  cudaError_t e = cudaMalloc(&d_amps_, bytes);
  if (e != cudaSuccess) {
    throw std::runtime_error("CudaStateVector: cannot allocate " + std::to_string(bytes >> 20) + " MiB for " +
                             std::to_string(n_) + " qubits: " + cudaGetErrorString(e));
  }
  e = cudaMalloc(&d_op_, kOpScratch);
  if (e != cudaSuccess) {
    cudaFree(d_amps_);
    check(e, "cudaMalloc(op scratch)");
  }
  reset();
}

template <class T>
CudaStateVector<T>::~CudaStateVector() {
  cudaFree(d_op_);
  cudaFree(d_amps_);
}

template <class T>
void CudaStateVector<T>::reset() {
  QSIM_CUDA(cudaMemset(d_amps_, 0, size() * sizeof(cplx<T>)));
  const cplx<T> one{T(1), T(0)};
  QSIM_CUDA(cudaMemcpy(d_amps_, &one, sizeof one, cudaMemcpyHostToDevice));
}

template <class T>
void CudaStateVector<T>::run(const Circuit& c) {
  if (c.num_qubits != n_)
    throw std::invalid_argument("circuit has " + std::to_string(c.num_qubits) + " qubits, state has " +
                                std::to_string(n_));
  for (const Op& op : c.ops) apply(op);
}

template <class T>
void CudaStateVector<T>::synchronize() const {
  QSIM_CUDA(cudaDeviceSynchronize());
}

template <class T>
void CudaStateVector<T>::apply(const Op& op) {
  const int k = op.arity();
  if (k <= 0) return;
  if (k > kMaxTargets) throw std::invalid_argument("CudaStateVector::apply: at most 6 targets");

  FixedBits f{};
  std::vector<int> fixed(op.targets);
  fixed.insert(fixed.end(), op.controls.begin(), op.controls.end());
  std::sort(fixed.begin(), fixed.end());
  f.count = static_cast<int>(fixed.size());
  std::copy(fixed.begin(), fixed.end(), f.pos);
  for (int c : op.controls) f.ctrl_mask |= std::uint64_t{1} << c;

  const std::uint64_t groups = std::uint64_t{1} << (n_ - f.count);
  auto* amps = static_cast<cplx<T>*>(d_amps_);
  auto conv = [&](std::size_t i) { return cplx<T>{T(op.matrix[i].real()), T(op.matrix[i].imag())}; };

  const unsigned grid = grid_for(groups, grid_cap_);
  if (k == 1) {
    apply_1t<T><<<grid, kBlock>>>(amps, groups, f, std::uint64_t{1} << op.targets[0], conv(0), conv(1),
                                  conv(2), conv(3));
  } else if (k <= 5) {
    switch (k) {
      case 2: launch_dense<T, 2>(amps, groups, f, op, grid); break;
      case 3: launch_dense<T, 3>(amps, groups, f, op, grid); break;
      case 4: launch_dense<T, 4>(amps, groups, f, op, grid); break;
      default: launch_dense<T, 5>(amps, groups, f, op, grid); break;
    }
  } else {
    const int dim = 1 << k;
    // Matrix (converted to T) followed by the member offsets, in one async copy.
    std::vector<unsigned char> staging(sizeof(cplx<T>) * dim * dim + sizeof(std::uint64_t) * dim);
    auto* m = reinterpret_cast<cplx<T>*>(staging.data());
    auto* offs = reinterpret_cast<std::uint64_t*>(staging.data() + sizeof(cplx<T>) * dim * dim);
    for (int i = 0; i < dim * dim; ++i) m[i] = conv(static_cast<std::size_t>(i));
    member_offsets(op, offs);
    // Pageable host memory: the copy is staged before this call returns, so `staging` may die, and
    // stream order guarantees the previous op's kernel is done with d_op_ before it's overwritten.
    QSIM_CUDA(cudaMemcpyAsync(d_op_, staging.data(), staging.size(), cudaMemcpyHostToDevice));
    auto* dm = static_cast<const cplx<T>*>(d_op_);
    auto* doffs = reinterpret_cast<const std::uint64_t*>(static_cast<unsigned char*>(d_op_) +
                                                         sizeof(cplx<T>) * dim * dim);
    apply_kt<T><<<grid_for(groups, grid_cap_), kBlock>>>(amps, groups, f, dim, dm, doffs);
  }
  QSIM_CUDA(cudaGetLastError());
}

template <class T>
std::vector<typename CudaStateVector<T>::amp_t> CudaStateVector<T>::download() const {
  std::vector<amp_t> out(size());
  QSIM_CUDA(cudaMemcpy(out.data(), d_amps_, size() * sizeof(amp_t), cudaMemcpyDeviceToHost));
  return out;
}

template <class T>
void CudaStateVector<T>::upload(std::span<const amp_t> amps) {
  if (amps.size() != size()) throw std::invalid_argument("CudaStateVector::upload: wrong number of amplitudes");
  QSIM_CUDA(cudaMemcpy(d_amps_, amps.data(), size() * sizeof(amp_t), cudaMemcpyHostToDevice));
}

template <class T>
double CudaStateVector<T>::norm_squared() const {
  const auto* a = static_cast<const cplx<T>*>(d_amps_);
  return thrust::transform_reduce(thrust::device, a, a + size(), NormSq<T>{}, 0.0, cuda::std::plus<double>{});
}

template <class T>
std::vector<double> CudaStateVector<T>::probabilities() const {
  const auto* a = static_cast<const cplx<T>*>(d_amps_);
  thrust::device_vector<double> p(size());
  thrust::transform(thrust::device, a, a + size(), p.begin(), NormSq<T>{});
  std::vector<double> out(size());
  thrust::copy(p.begin(), p.end(), out.begin());
  return out;
}

// Same algorithm and the same uniform draws as the CPU: cumulative distribution (a parallel scan
// here), then one binary search per shot. Needs 8 bytes of GPU memory per amplitude for the scan.
template <class T>
std::vector<std::uint64_t> CudaStateVector<T>::sample(int shots, std::uint64_t seed) const {
  if (shots < 0) throw std::invalid_argument("CudaStateVector::sample: shots must be >= 0");
  if (shots == 0) return {};

  const auto* a = static_cast<const cplx<T>*>(d_amps_);
  thrust::device_vector<double> cdf(size());
  thrust::transform_inclusive_scan(thrust::device, a, a + size(), cdf.begin(), NormSq<T>{},
                                   cuda::std::plus<double>{});
  const double total = cdf.back();
  if (!(total > 0.0)) throw std::runtime_error("CudaStateVector::sample: state has zero norm");

  std::mt19937_64 rng(seed);
  std::vector<double> u(static_cast<std::size_t>(shots));
  for (auto& x : u) x = uniform01(rng) * total;
  thrust::device_vector<double> du(u.begin(), u.end());
  thrust::device_vector<std::uint64_t> idx(static_cast<std::size_t>(shots));
  thrust::upper_bound(thrust::device, cdf.begin(), cdf.end(), du.begin(), du.end(), idx.begin());

  std::vector<std::uint64_t> out(static_cast<std::size_t>(shots));
  thrust::copy(idx.begin(), idx.end(), out.begin());
  for (auto& i : out) i = std::min(i, size() - 1);  // u can round up to total
  return out;
}

template <class T>
int CudaStateVector<T>::measure(int qubit, std::mt19937_64& rng) {
  if (qubit < 0 || qubit >= n_) throw std::invalid_argument("CudaStateVector::measure: qubit out of range");
  const std::uint64_t bit = std::uint64_t{1} << qubit;
  auto* a = static_cast<cplx<T>*>(d_amps_);

  const double total = norm_squared();
  const double p1 = thrust::transform_reduce(thrust::device, thrust::counting_iterator<std::uint64_t>(0),
                                             thrust::counting_iterator<std::uint64_t>(size()),
                                             NormSqIfBit<T>{a, bit}, 0.0, cuda::std::plus<double>{});
  const double p0 = total - p1;
  const double p0n = total > 0.0 ? p0 / total : 0.0;

  const int outcome = uniform01(rng) < p0n ? 0 : 1;
  const double p = outcome ? p1 : p0;
  if (!(p > 0.0)) throw std::runtime_error("CudaStateVector::measure: zero probability for the drawn outcome");

  collapse<T><<<grid_for(size(), grid_cap_), kBlock>>>(a, size(), bit, outcome ? bit : 0,
                                                       static_cast<T>(1.0 / std::sqrt(p)));
  QSIM_CUDA(cudaGetLastError());
  return outcome;
}

template class CudaStateVector<float>;
template class CudaStateVector<double>;

}  // namespace qsim
