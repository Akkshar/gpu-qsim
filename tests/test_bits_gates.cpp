// Spec for include/qsim/bits.hpp and src/cpu/gates.cpp.   Run: qsim_tests "[bits],[gates]"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>
#include <vector>

#include "qsim/bits.hpp"
#include "qsim/gates.hpp"

using qsim::cdouble;
using qsim::Matrix;

namespace {

constexpr double kTol = 1e-12;
constexpr double kPi = std::numbers::pi;
const cdouble I{0, 1};

std::size_t dim_of(const Matrix& m) { return static_cast<std::size_t>(std::lround(std::sqrt(double(m.size())))); }

Matrix matmul(const Matrix& a, const Matrix& b) {
  const std::size_t d = dim_of(a);
  Matrix c(d * d);
  for (std::size_t i = 0; i < d; ++i)
    for (std::size_t j = 0; j < d; ++j)
      for (std::size_t k = 0; k < d; ++k) c[i * d + j] += a[i * d + k] * b[k * d + j];
  return c;
}

Matrix dagger(const Matrix& a) {
  const std::size_t d = dim_of(a);
  Matrix c(d * d);
  for (std::size_t i = 0; i < d; ++i)
    for (std::size_t j = 0; j < d; ++j) c[j * d + i] = std::conj(a[i * d + j]);
  return c;
}

Matrix scaled(cdouble s, Matrix m) {
  for (auto& x : m) x *= s;
  return m;
}

Matrix identity(std::size_t d) {
  Matrix m(d * d);
  for (std::size_t i = 0; i < d; ++i) m[i * d + i] = 1;
  return m;
}

bool close(const Matrix& a, const Matrix& b, double tol = kTol) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::abs(a[i] - b[i]) > tol) return false;
  return true;
}

}  // namespace

TEST_CASE("insert_zero_bit", "[bits]") {
  using qsim::insert_zero_bit;
  CHECK(insert_zero_bit(0b1011u, 2) == 0b10011u);
  CHECK(insert_zero_bit(0b1011u, 0) == 0b10110u);
  CHECK(insert_zero_bit(0b1011u, 10) == 0b1011u);
  CHECK(insert_zero_bit(0u, 5) == 0u);
  CHECK(insert_zero_bit((std::uint64_t{1} << 40) - 1, 20) == ((std::uint64_t{1} << 41) - 1) - (std::uint64_t{1} << 20));

  // For each position, k = 0..15 must map onto exactly the 16 indices in 0..31 with that bit clear.
  for (int pos = 0; pos < 5; ++pos) {
    std::vector<bool> seen(32, false);
    for (std::uint64_t k = 0; k < 16; ++k) {
      const std::uint64_t i = insert_zero_bit(k, pos);
      INFO("pos " << pos << ", k " << k << " -> " << i);
      REQUIRE(i < 32);
      CHECK(((i >> pos) & 1u) == 0u);
      CHECK_FALSE(seen[i]);
      seen[i] = true;
    }
  }
}

TEST_CASE("gates are unitary", "[gates]") {
  using namespace qsim::gates;
  const std::vector<Matrix> all = {H(),   X(),     Y(),      Z(),       S(),
                                   Sdg(), T(),     Tdg(),    RX(0.3),   RY(-1.1),
                                   RZ(2.5), U3(0.4, 1.3, -0.7), SWAP()};
  for (const Matrix& m : all) {
    REQUIRE((m.size() == 4 || m.size() == 16));
    CHECK(close(matmul(dagger(m), m), identity(dim_of(m))));
  }
}

TEST_CASE("gate definitions match Qiskit exactly, including global phase", "[gates]") {
  using namespace qsim::gates;
  const double r = 1 / std::sqrt(2.0);
  const double th = 0.7;

  CHECK(close(H(), {r, r, r, -r}));
  CHECK(close(X(), {0, 1, 1, 0}));
  CHECK(close(Y(), {0, -I, I, 0}));
  CHECK(close(Z(), {1, 0, 0, -1}));
  CHECK(close(matmul(S(), S()), Z()));
  CHECK(close(matmul(T(), T()), S()));
  CHECK(close(Sdg(), dagger(S())));
  CHECK(close(Tdg(), dagger(T())));

  CHECK(close(RZ(th), {std::exp(-I * th / 2.0), 0, 0, std::exp(I * th / 2.0)}));
  CHECK(close(RX(kPi), scaled(-I, X())));
  CHECK(close(RY(kPi), scaled(-I, Y())));
  CHECK(close(RZ(kPi), scaled(-I, Z())));

  CHECK(close(U3(kPi / 2, 0, kPi), H()));
  CHECK(close(U3(kPi, 0, kPi), X()));
  CHECK(close(U3(th, -kPi / 2, kPi / 2), RX(th)));
  CHECK(close(U3(th, 0, 0), RY(th)));

  // Angles where cos(t/2) and sin(t/2) have different signs, written out from the formula with
  // std::exp so the expected value never depends on std::polar's handling of negative magnitudes.
  for (double t : {-2.0, 3.9, -5.5}) {
    const double p = 0.3, l = -1.1, c = std::cos(t / 2), s = std::sin(t / 2);
    INFO("theta " << t);
    CHECK(close(U3(t, p, l), {c, -std::exp(I * l) * s, std::exp(I * p) * s, std::exp(I * (p + l)) * c}));
  }

  CHECK(close(SWAP(), {1, 0, 0, 0,  0, 0, 1, 0,  0, 1, 0, 0,  0, 0, 0, 1}));
}
