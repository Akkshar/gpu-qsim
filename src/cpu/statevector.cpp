#include "qsim/cpu_statevector.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "qsim/bits.hpp"

namespace qsim {

template <class T>
CpuStateVector<T>::CpuStateVector(int num_qubits) : n_(num_qubits) {
  if (num_qubits < 1 || num_qubits > 34)
    throw std::invalid_argument("CpuStateVector: num_qubits must be 1..34, got " +
                                std::to_string(num_qubits));
  amps_.assign(std::size_t{1} << num_qubits, amp_t{0});
  amps_[0] = amp_t{1};
}

template <class T>
void CpuStateVector<T>::reset() {
  std::fill(amps_.begin(), amps_.end(), amp_t{0});
  amps_[0] = amp_t{1};
}

template <class T>
void CpuStateVector<T>::run(const Circuit& c) {
  if (c.num_qubits != n_)
    throw std::invalid_argument("circuit has " + std::to_string(c.num_qubits) +
                                " qubits, state has " + std::to_string(n_));
  for (const Op& op : c.ops) apply(op);
}

// ============================ M1: your code from here down ============================
// docs/M1.md walks through the order to do these in. Delete each throw as you go.

#define QSIM_TODO(what) throw std::logic_error("TODO(M1): " what " (src/cpu/statevector.cpp)")

// Apply op.matrix to op.targets wherever every op.controls qubit is 1.
// Start with the 1-qubit, no-controls case (pairs via insert_zero_bit), get its tests green, then
// add controls, then general k targets. Remember op.matrix is in double; amps_ may be float.
template <class T>
void CpuStateVector<T>::apply(const Op& op) {
  (void)op;
  QSIM_TODO("CpuStateVector::apply");
}

template <class T>
double CpuStateVector<T>::norm_squared() const {
  QSIM_TODO("CpuStateVector::norm_squared");
}

template <class T>
std::vector<double> CpuStateVector<T>::probabilities() const {
  QSIM_TODO("CpuStateVector::probabilities");
}

// Draw `shots` basis-state indices from the distribution |amp|^2, seeded so results are reproducible.
template <class T>
std::vector<std::uint64_t> CpuStateVector<T>::sample(int shots, std::uint64_t seed) const {
  (void)shots;
  (void)seed;
  QSIM_TODO("CpuStateVector::sample");
}

// Measure one qubit: pick 0/1 with the right probability, zero the other half, renormalise.
template <class T>
int CpuStateVector<T>::measure(int qubit, std::mt19937_64& rng) {
  (void)qubit;
  (void)rng;
  QSIM_TODO("CpuStateVector::measure");
}

template class CpuStateVector<float>;
template class CpuStateVector<double>;

}  // namespace qsim
