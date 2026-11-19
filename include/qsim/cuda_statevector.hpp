#pragma once
// CUDA state-vector backend. Same interface and semantics as CpuStateVector, except the amplitudes
// live in GPU memory: read them with download(), set them with upload().
//
// M2 version: straightforward kernels, one launch per op, no fusion or shared-memory tricks yet.
// Kernels are queued asynchronously; anything that returns data (download, norm_squared, sample,
// measure, ...) waits for them. Time with synchronize().
//
// Plain C++ header (no CUDA types), so code that only uses the class compiles with any C++ compiler.

#include <complex>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "qsim/circuit.hpp"

namespace qsim {

// Which kernel runs ops with 2..5 targets (M3.3):
//   Direct:  one thread per group of 2^k amplitudes. At the memory floor unless a target is a low
//            qubit: then each group is contiguous, lanes sit 2^k amplitudes apart, and L1 thrashes.
//   Shuffle: one lane per amplitude; a group's members are adjacent lanes of one warp and exchange
//            values with warp shuffles, so loads are coalesced whatever the targets are.
//   Auto picks per op (rule in use_shuffle(), measured in BENCH.md, M3.3).
enum class MultiTargetKernel { Auto, Shuffle, Direct };

template <class T>  // float or double
class CudaStateVector {
 public:
  using amp_t = std::complex<T>;

  explicit CudaStateVector(int num_qubits);  // 2^n amplitudes on the GPU, starts in |0...0>
  ~CudaStateVector();
  CudaStateVector(const CudaStateVector&) = delete;
  CudaStateVector& operator=(const CudaStateVector&) = delete;

  int num_qubits() const { return n_; }
  std::uint64_t size() const { return std::uint64_t{1} << n_; }

  void reset();                // back to |0...0>
  void run(const Circuit& c);  // queue every op in order
  void apply(const Op& op);    // up to 6 targets, any controls
  void synchronize() const;    // wait for queued work
  void set_multi_target_kernel(MultiTargetKernel k) { mt_kernel_ = k; }

  std::vector<amp_t> download() const;
  void upload(std::span<const amp_t> amps);

  double norm_squared() const;
  std::vector<double> probabilities() const;
  std::vector<std::uint64_t> sample(int shots, std::uint64_t seed) const;  // leaves state unchanged
  int measure(int qubit, std::mt19937_64& rng);  // 0 or 1; collapses and renormalises

 private:
  bool use_shuffle(const Op& op) const;

  int n_;
  MultiTargetKernel mt_kernel_ = MultiTargetKernel::Auto;
  unsigned grid_cap_ = 0;   // max blocks per launch (grid-stride loops cover the rest)
  void* d_amps_ = nullptr;  // 2^n complex<T>
  void* d_op_ = nullptr;    // per-op scratch: matrix + member offsets for multi-target ops
};

extern template class CudaStateVector<float>;
extern template class CudaStateVector<double>;

}  // namespace qsim
