#include "qsim/cpu_statevector.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

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

template <class T>
void CpuStateVector<T>::apply(const Op& op) {
  using ComplexD = std::complex<double>;

  const int k = static_cast<int>(op.targets.size());
  if (k <= 0) return;

  const std::uint64_t dim = std::uint64_t{1} << k;
  const std::uint64_t num_states = std::uint64_t{1} << n_;

  std::uint64_t ctrl_mask = 0;
  for (int c : op.controls) {
    ctrl_mask |= (std::uint64_t{1} << c);
  }

  std::uint64_t target_mask = 0;
  for (int t : op.targets) {
    target_mask |= (std::uint64_t{1} << t);
  }

  // Precompute bit offsets for each basis state r in [0, 2^k - 1] within the target subspace.
  std::vector<std::uint64_t> offsets(static_cast<std::size_t>(dim), 0);
  for (std::uint64_t r = 0; r < dim; ++r) {
    std::uint64_t off = 0;
    for (int j = 0; j < k; ++j) {
      if ((r >> j) & std::uint64_t{1}) {
        off |= (std::uint64_t{1} << op.targets[static_cast<std::size_t>(j)]);
      }
    }
    offsets[static_cast<std::size_t>(r)] = off;
  }

  std::vector<std::uint64_t> indices(static_cast<std::size_t>(dim));
  std::vector<ComplexD> in(static_cast<std::size_t>(dim));
  std::vector<ComplexD> out(static_cast<std::size_t>(dim));

  for (std::uint64_t base = 0; base < num_states; ++base) {
    if ((base & ctrl_mask) != ctrl_mask) continue;
    if ((base & target_mask) != 0) continue;

    for (std::uint64_t r = 0; r < dim; ++r) {
      const std::uint64_t idx = base | offsets[static_cast<std::size_t>(r)];
      indices[static_cast<std::size_t>(r)] = idx;
      const amp_t a = amps_[static_cast<std::size_t>(idx)];
      in[static_cast<std::size_t>(r)] =
          ComplexD{static_cast<double>(a.real()), static_cast<double>(a.imag())};
    }

    for (std::uint64_t r = 0; r < dim; ++r) {
      ComplexD sum{0.0, 0.0};
      const std::size_t row_offset = static_cast<std::size_t>(r * dim);
      for (std::uint64_t c = 0; c < dim; ++c) {
        sum += op.matrix[row_offset + static_cast<std::size_t>(c)] *
               in[static_cast<std::size_t>(c)];
      }
      out[static_cast<std::size_t>(r)] = sum;
    }

    for (std::uint64_t r = 0; r < dim; ++r) {
      const ComplexD val = out[static_cast<std::size_t>(r)];
      amps_[static_cast<std::size_t>(indices[static_cast<std::size_t>(r)])] =
          amp_t{static_cast<T>(val.real()), static_cast<T>(val.imag())};
    }
  }
}

template <class T>
double CpuStateVector<T>::norm_squared() const {
  long double total = 0.0L;
  for (const amp_t& a : amps_) {
    const long double re = static_cast<long double>(a.real());
    const long double im = static_cast<long double>(a.imag());
    total += re * re + im * im;
  }
  return static_cast<double>(total);
}

template <class T>
std::vector<double> CpuStateVector<T>::probabilities() const {
  std::vector<double> probs(amps_.size());
  for (std::size_t i = 0; i < amps_.size(); ++i) {
    const double re = static_cast<double>(amps_[i].real());
    const double im = static_cast<double>(amps_[i].imag());
    probs[i] = re * re + im * im;
  }
  return probs;
}

// Draw `shots` basis-state indices from the distribution |amp|^2, seeded so results are reproducible.
template <class T>
std::vector<std::uint64_t> CpuStateVector<T>::sample(int shots, std::uint64_t seed) const {
  if (shots < 0) {
    throw std::invalid_argument("CpuStateVector::sample: shots must be >= 0");
  }
  std::vector<std::uint64_t> samples;
  samples.reserve(static_cast<std::size_t>(shots));
  if (shots == 0) return samples;

  const std::vector<double> probs = probabilities();
  std::mt19937_64 rng(seed);
  std::discrete_distribution<std::uint64_t> dist(probs.begin(), probs.end());

  for (int i = 0; i < shots; ++i) {
    samples.push_back(dist(rng));
  }
  return samples;
}

// Measure one qubit: pick 0/1 with the right probability, zero the other half, renormalise.
template <class T>
int CpuStateVector<T>::measure(int qubit, std::mt19937_64& rng) {
  if (qubit < 0 || qubit >= n_) {
    throw std::invalid_argument("CpuStateVector::measure: qubit out of range");
  }

  const std::uint64_t bit = std::uint64_t{1} << qubit;
  long double p0_ld = 0.0L;
  long double p1_ld = 0.0L;

  for (std::size_t idx = 0; idx < amps_.size(); ++idx) {
    const long double re = static_cast<long double>(amps_[idx].real());
    const long double im = static_cast<long double>(amps_[idx].imag());
    const long double p = re * re + im * im;
    if ((static_cast<std::uint64_t>(idx) & bit) == 0) {
      p0_ld += p;
    } else {
      p1_ld += p;
    }
  }

  const long double total = p0_ld + p1_ld;
  const double p0 = (total > 0.0L) ? static_cast<double>(p0_ld / total) : 0.0;

  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  const double r = uniform(rng);
  const int outcome = (r < p0) ? 0 : 1;

  if (outcome == 0) {
    if (p0_ld <= 0.0L) {
      throw std::runtime_error("CpuStateVector::measure: zero probability for outcome 0");
    }
    const T scale = static_cast<T>(1.0L / std::sqrt(p0_ld));
    for (std::size_t idx = 0; idx < amps_.size(); ++idx) {
      if ((static_cast<std::uint64_t>(idx) & bit) != 0) {
        amps_[idx] = amp_t{0};
      } else {
        amps_[idx] *= scale;
      }
    }
  } else {
    if (p1_ld <= 0.0L) {
      throw std::runtime_error("CpuStateVector::measure: zero probability for outcome 1");
    }
    const T scale = static_cast<T>(1.0L / std::sqrt(p1_ld));
    for (std::size_t idx = 0; idx < amps_.size(); ++idx) {
      if ((static_cast<std::uint64_t>(idx) & bit) == 0) {
        amps_[idx] = amp_t{0};
      } else {
        amps_[idx] *= scale;
      }
    }
  }

  return outcome;
}

template class CpuStateVector<float>;
template class CpuStateVector<double>;

}  // namespace qsim