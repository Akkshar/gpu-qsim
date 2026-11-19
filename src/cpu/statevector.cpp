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

namespace {

// Plain complex multiply. std::complex's operator* is allowed to do C99 Annex G inf/NaN recovery
// (GCC calls __muldc3 for it unless -ffast-math), which is far slower than these four FMAs.
template <class T>
inline std::complex<T> cmul(std::complex<T> a, std::complex<T> b) {
  return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}

// Uniform double in [0, 1) from the top 53 bits of one engine draw. The algorithms behind
// std::uniform_real_distribution / std::discrete_distribution are implementation-defined, so the
// same seed gives different results on MSVC and libstdc++; mt19937_64's raw output is specified.
inline double uniform01(std::mt19937_64& rng) {
  return static_cast<double>(rng() >> 11) * 0x1.0p-53;
}

}  // namespace

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

// ============================ M1: simulator core ============================

// Apply op.matrix to op.targets wherever every op.controls qubit is 1.
//
// Every group of amplitudes the op touches has a "base" index with all control bits = 1 and all
// target bits = 0. Rather than scanning all 2^n indices and skipping the rest, count g over the
// 2^(n - targets - controls) groups and build each base directly: insert a 0 at every target and
// control position, lowest position first (inserting a low bit shifts every higher bit up by one,
// so going in ascending order keeps the later positions correct), then OR in the control mask.
template <class T>
void CpuStateVector<T>::apply(const Op& op) {
  using ComplexD = std::complex<double>;

  const int k = static_cast<int>(op.targets.size());
  if (k <= 0) return;

  std::uint64_t ctrl_mask = 0;
  for (int c : op.controls) ctrl_mask |= (std::uint64_t{1} << c);

  std::vector<int> fixed(op.targets);
  fixed.insert(fixed.end(), op.controls.begin(), op.controls.end());
  std::sort(fixed.begin(), fixed.end());
  const std::uint64_t groups = std::uint64_t{1} << (n_ - static_cast<int>(fixed.size()));

  auto base_of = [&](std::uint64_t g) {
    for (int p : fixed) g = insert_zero_bit(g, p);
    return g | ctrl_mask;
  };

  // ---- Fast path: one target. 2x2 in registers, matrix converted to the state's precision once.
  if (k == 1) {
    const int t = op.targets[0];
    const std::uint64_t bit = std::uint64_t{1} << t;
    const amp_t m00(op.matrix[0]), m01(op.matrix[1]), m10(op.matrix[2]), m11(op.matrix[3]);
    auto pair = [&](std::uint64_t i0) {
      const std::size_t a = static_cast<std::size_t>(i0), b = static_cast<std::size_t>(i0 | bit);
      const amp_t a0 = amps_[a], a1 = amps_[b];
      amps_[a] = cmul(m00, a0) + cmul(m01, a1);
      amps_[b] = cmul(m10, a0) + cmul(m11, a1);
    };
    if (op.controls.empty()) {
      for (std::uint64_t g = 0; g < groups; ++g) pair(insert_zero_bit(g, t));
    } else {
      for (std::uint64_t g = 0; g < groups; ++g) pair(base_of(g));
    }
    return;
  }

  // ---- General path: k targets. Gather 2^k amplitudes, multiply in double, scatter back.
  const std::uint64_t dim = std::uint64_t{1} << k;

  // offsets[r]: which target bits are set for member r of a group. Uses targets in *matrix order*
  // (targets[0] = bit 0 of r), not the sorted order used to build the base.
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

  for (std::uint64_t g = 0; g < groups; ++g) {
    const std::uint64_t base = base_of(g);

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
        sum += cmul(op.matrix[row_offset + static_cast<std::size_t>(c)],
                    in[static_cast<std::size_t>(c)]);
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

// Draw `shots` basis-state indices from the distribution |amp|^2, seeded so results are reproducible
// (on every platform: see uniform01). Cumulative sum + binary search: one 8-byte-per-amplitude array.
template <class T>
std::vector<std::uint64_t> CpuStateVector<T>::sample(int shots, std::uint64_t seed) const {
  if (shots < 0) {
    throw std::invalid_argument("CpuStateVector::sample: shots must be >= 0");
  }
  std::vector<std::uint64_t> samples;
  samples.reserve(static_cast<std::size_t>(shots));
  if (shots == 0) return samples;

  std::vector<double> cdf(amps_.size());
  double running = 0.0;
  for (std::size_t i = 0; i < amps_.size(); ++i) {
    const double re = static_cast<double>(amps_[i].real());
    const double im = static_cast<double>(amps_[i].imag());
    running += re * re + im * im;
    cdf[i] = running;
  }
  const double total = running;  // ~1, but use the real sum so rounding drift can't skew draws
  if (!(total > 0.0)) throw std::runtime_error("CpuStateVector::sample: state has zero norm");

  std::mt19937_64 rng(seed);
  for (int i = 0; i < shots; ++i) {
    const double u = uniform01(rng) * total;
    // First entry strictly greater than u: zero-probability states (equal neighbouring cdf values)
    // can never be picked.
    auto it = std::upper_bound(cdf.begin(), cdf.end(), u);
    if (it == cdf.end()) --it;  // u can round up to total
    samples.push_back(static_cast<std::uint64_t>(it - cdf.begin()));
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

  const double r = uniform01(rng);
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
