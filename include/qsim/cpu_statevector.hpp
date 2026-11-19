#pragma once
// CPU reference state-vector simulator. Slow and simple on purpose: it's the ground truth the GPU
// backends are tested against, so clarity beats speed here.

#include <complex>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "qsim/circuit.hpp"

namespace qsim {

template <class T>  // float or double
class CpuStateVector {
 public:
  using amp_t = std::complex<T>;

  explicit CpuStateVector(int num_qubits);  // 2^n amplitudes, starts in |0...0>

  int num_qubits() const { return n_; }
  std::uint64_t size() const { return std::uint64_t{1} << n_; }
  std::span<amp_t> amps() { return amps_; }
  std::span<const amp_t> amps() const { return amps_; }

  void reset();                 // back to |0...0>
  void run(const Circuit& c);   // apply every op in order

  // ---- M1: yours, in src/cpu/statevector.cpp ----
  void apply(const Op& op);                         // any number of targets, any controls
  double norm_squared() const;                      // sum of |amp|^2 (1 for a valid state)
  std::vector<double> probabilities() const;        // |amp_i|^2 for every basis state i
  std::vector<std::uint64_t> sample(int shots, std::uint64_t seed) const;  // leaves state unchanged
  int measure(int qubit, std::mt19937_64& rng);     // 0 or 1; collapses and renormalises

 private:
  int n_;
  std::vector<amp_t> amps_;
};

extern template class CpuStateVector<float>;
extern template class CpuStateVector<double>;

}  // namespace qsim
