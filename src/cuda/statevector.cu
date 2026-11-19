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

// k targets: each thread owns one group of 2^k amplitudes. Matrix and member offsets come from
// global memory (identical for every thread, so they stay hot in cache).
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

  if (k == 1) {
    apply_1t<T><<<grid_for(groups, grid_cap_), kBlock>>>(amps, groups, f, std::uint64_t{1} << op.targets[0],
                                                         conv(0), conv(1), conv(2), conv(3));
  } else {
    const int dim = 1 << k;
    // Matrix (converted to T) followed by the member offsets, in one async copy. targets[j] maps to
    // bit j of the member index, exactly as on the CPU.
    std::vector<unsigned char> staging(sizeof(cplx<T>) * dim * dim + sizeof(std::uint64_t) * dim);
    auto* m = reinterpret_cast<cplx<T>*>(staging.data());
    auto* offs = reinterpret_cast<std::uint64_t*>(staging.data() + sizeof(cplx<T>) * dim * dim);
    for (int i = 0; i < dim * dim; ++i) m[i] = conv(static_cast<std::size_t>(i));
    for (int r = 0; r < dim; ++r) {
      std::uint64_t off = 0;
      for (int j = 0; j < k; ++j)
        if ((r >> j) & 1) off |= std::uint64_t{1} << op.targets[static_cast<std::size_t>(j)];
      offs[r] = off;
    }
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
