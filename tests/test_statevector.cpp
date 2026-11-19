// Spec for CpuStateVector (src/cpu/statevector.cpp). Every test runs in float and double.
//   Run: qsim_tests "[sv]"      or one area: qsim_tests "[apply]" / "[measure]"

#include <algorithm>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <type_traits>

#include "qsim/circuit.hpp"
#include "qsim/cpu_statevector.hpp"

using qsim::CpuStateVector;

namespace {

template <class T>
double tol() { return std::is_same_v<T, float> ? 1e-5 : 1e-12; }

template <class T>
CpuStateVector<T> run_text(const std::string& text) {
  const auto c = qsim::parse_circuit(text);
  CpuStateVector<T> sv(c.num_qubits);
  sv.run(c);
  return sv;
}

// Exactly one amplitude is non-zero, with |amp| == 1.
template <class T>
void check_basis(const CpuStateVector<T>& sv, std::uint64_t idx) {
  for (std::uint64_t i = 0; i < sv.size(); ++i) {
    INFO("amplitude " << i << " (expected basis state " << idx << ")");
    CHECK(std::abs(std::abs(std::complex<double>(sv.amps()[i])) - (i == idx ? 1.0 : 0.0)) < tol<T>());
  }
}

std::string num(double x) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.17g", x);
  return buf;
}

// A random circuit followed by its exact inverse, so the result must be |0...0> again.
std::string random_then_inverse(int n, int depth, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> ang(-6.3, 6.3);
  std::vector<std::string> fwd, inv;
  std::vector<int> qs(n);
  for (int i = 0; i < n; ++i) qs[i] = i;
  for (int d = 0; d < depth; ++d) {
    std::shuffle(qs.begin(), qs.end(), rng);
    const std::string a = std::to_string(qs[0]), b = std::to_string(qs[1]), c = std::to_string(qs[2]);
    const double t = ang(rng), p = ang(rng), l = ang(rng);
    switch (rng() % 9) {
      case 0: fwd.push_back("h " + a); inv.push_back("h " + a); break;
      case 1: fwd.push_back("s " + a); inv.push_back("sdg " + a); break;
      case 2: fwd.push_back("t " + a); inv.push_back("tdg " + a); break;
      case 3: fwd.push_back("rx " + num(t) + " " + a); inv.push_back("rx " + num(-t) + " " + a); break;
      case 4:  // U3(t,p,l)^dagger = U3(-t,-l,-p)
        fwd.push_back("u3 " + num(t) + " " + num(p) + " " + num(l) + " " + a);
        inv.push_back("u3 " + num(-t) + " " + num(-l) + " " + num(-p) + " " + a);
        break;
      case 5: fwd.push_back("cx " + a + " " + b); inv.push_back("cx " + a + " " + b); break;
      case 6: fwd.push_back("swap " + a + " " + b); inv.push_back("swap " + a + " " + b); break;
      case 7: fwd.push_back("ccx " + a + " " + b + " " + c); inv.push_back("ccx " + a + " " + b + " " + c); break;
      default: fwd.push_back("crz " + num(t) + " " + a + " " + b); inv.push_back("crz " + num(-t) + " " + a + " " + b); break;
    }
  }
  std::string text = "qubits " + std::to_string(n) + "\n";
  for (const auto& s : fwd) text += s + "\n";
  for (auto it = inv.rbegin(); it != inv.rend(); ++it) text += *it + "\n";
  return text;
}

std::string bell = "qubits 2\nh 0\ncx 0 1\n";

}  // namespace

// ---------------------------------------------------------------- support code (passes already)

TEMPLATE_TEST_CASE("starts in |0...0> and validates its inputs", "[sv][setup]", float, double) {
  CpuStateVector<TestType> sv(4);
  CHECK(sv.size() == 16);
  check_basis(sv, 0);
  CHECK_THROWS(CpuStateVector<TestType>(0));
  CHECK_THROWS(CpuStateVector<TestType>(35));
  CHECK_THROWS_AS(sv.run(qsim::parse_circuit("qubits 3")), std::invalid_argument);
}

// ---------------------------------------------------------------- apply: 1 qubit, no controls

TEMPLATE_TEST_CASE("H on qubit 0", "[sv][apply]", float, double) {
  const auto sv = run_text<TestType>("qubits 3\nh 0");
  const double r = 1 / std::sqrt(2.0);
  for (std::uint64_t i = 0; i < 8; ++i)
    CHECK(std::abs(std::complex<double>(sv.amps()[i]) - (i < 2 ? r : 0.0)) < tol<TestType>());
}

TEMPLATE_TEST_CASE("X on every qubit position", "[sv][apply]", float, double) {
  for (int q = 0; q < 5; ++q) {
    INFO("x " << q);
    check_basis(run_text<TestType>("qubits 5\nx " + std::to_string(q)), std::uint64_t{1} << q);
  }
}

TEMPLATE_TEST_CASE("H on the highest qubit", "[sv][apply]", float, double) {
  const auto sv = run_text<TestType>("qubits 6\nh 5");
  const double r = 1 / std::sqrt(2.0);
  CHECK(std::abs(std::complex<double>(sv.amps()[0]) - r) < tol<TestType>());
  CHECK(std::abs(std::complex<double>(sv.amps()[32]) - r) < tol<TestType>());
}

// ---------------------------------------------------------------- apply: controls

TEMPLATE_TEST_CASE("Bell state", "[sv][apply][controls]", float, double) {
  const auto sv = run_text<TestType>(bell);
  const double r = 1 / std::sqrt(2.0);
  CHECK(std::abs(std::complex<double>(sv.amps()[0]) - r) < tol<TestType>());
  CHECK(std::abs(std::complex<double>(sv.amps()[3]) - r) < tol<TestType>());
  CHECK(std::abs(std::complex<double>(sv.amps()[1])) < tol<TestType>());
  CHECK(std::abs(std::complex<double>(sv.amps()[2])) < tol<TestType>());
}

TEMPLATE_TEST_CASE("cx only acts when the control is 1", "[sv][apply][controls]", float, double) {
  check_basis(run_text<TestType>("qubits 2\nx 0\ncx 1 0"), 0b01);  // control q1 = 0: no-op
  check_basis(run_text<TestType>("qubits 2\nx 1\ncx 1 0"), 0b11);  // control q1 = 1: flips q0
}

TEMPLATE_TEST_CASE("ccx truth table", "[sv][apply][controls]", float, double) {
  for (std::uint64_t in = 0; in < 8; ++in) {
    std::string text = "qubits 3\n";
    for (int q = 0; q < 3; ++q)
      if ((in >> q) & 1u) text += "x " + std::to_string(q) + "\n";
    text += "ccx 0 1 2\n";
    const std::uint64_t want = ((in & 3u) == 3u) ? (in ^ 4u) : in;
    INFO("input " << in);
    check_basis(run_text<TestType>(text), want);
  }
}

TEMPLATE_TEST_CASE("crz phase only appears when the control is 1", "[sv][apply][controls]", float, double) {
  // |+> on q1 (control), |1> on q0 (target): crz(t) gives the q1=1 branch a relative phase.
  const double t = 0.9;
  const auto sv = run_text<TestType>("qubits 2\nx 0\nh 1\ncrz " + num(t) + " 1 0");
  const std::complex<double> a1(sv.amps()[1]), a3(sv.amps()[3]);
  const double r = 1 / std::sqrt(2.0);
  CHECK(std::abs(a1 - r) < tol<TestType>());                                       // control 0: untouched
  CHECK(std::abs(a3 - r * std::exp(std::complex<double>(0, t / 2))) < tol<TestType>());  // RZ on |1>
}

// ---------------------------------------------------------------- apply: multi-qubit targets

TEMPLATE_TEST_CASE("swap and cswap", "[sv][apply][multi]", float, double) {
  check_basis(run_text<TestType>("qubits 4\nx 0\nswap 0 3"), 0b1000);
  check_basis(run_text<TestType>("qubits 3\nx 0\nx 1\ncswap 0 1 2"), 0b101);  // control on: q1<->q2
  check_basis(run_text<TestType>("qubits 3\nx 1\ncswap 0 1 2"), 0b010);       // control off
}

TEMPLATE_TEST_CASE("mat: targets[0] is the low bit of the matrix index", "[sv][apply][multi]", float, double) {
  // 4x4 matrix that flips bit 0 of the matrix index: M[r][c] = 1 iff r == c ^ 1.
  std::string m;
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) m += (r == (c ^ 1)) ? " 1 0" : " 0 0";
  check_basis(run_text<TestType>("qubits 4\nmat 2 3 1" + m), 0b1000);  // bit 0 <-> qubit 3
  check_basis(run_text<TestType>("qubits 4\nmat 2 1 3" + m), 0b0010);  // bit 0 <-> qubit 1
}

TEMPLATE_TEST_CASE("mat: 3 targets, scattered and out of order", "[sv][apply][multi]", float, double) {
  // Cyclic shift on the 3-bit matrix index: M[r][c] = 1 iff r == (c + 1) mod 8.
  std::string m;
  for (int r = 0; r < 8; ++r)
    for (int c = 0; c < 8; ++c) m += (r == (c + 1) % 8) ? " 1 0" : " 0 0";
  const std::string op = "mat 3 4 0 2" + m + "\n";  // index bit 0 <-> q4, bit 1 <-> q0, bit 2 <-> q2
  std::string text = "qubits 5\n";
  const std::uint64_t want[] = {0b10000, 0b00001, 0b10001, 0b00100};  // matrix index 1, 2, 3, 4
  for (int k = 0; k < 4; ++k) {
    text += op;
    INFO("after " << (k + 1) << " applications");
    check_basis(run_text<TestType>(text), want[k]);
  }
}

// ---------------------------------------------------------------- whole circuits

TEMPLATE_TEST_CASE("a circuit followed by its inverse returns to |0>", "[sv][apply][random]", float, double) {
  const double t = std::is_same_v<TestType, float> ? 1e-3 : 1e-9;
  for (unsigned seed = 1; seed <= 5; ++seed) {
    const auto sv = run_text<TestType>(random_then_inverse(8, 150, seed));
    INFO("seed " << seed);
    CHECK(std::abs(std::abs(std::complex<double>(sv.amps()[0])) - 1.0) < t);
  }
}

TEMPLATE_TEST_CASE("norm is preserved", "[sv][apply][random]", float, double) {
  const double t = std::is_same_v<TestType, float> ? 1e-4 : 1e-10;
  auto sv = run_text<TestType>(random_then_inverse(10, 300, 99));
  CHECK(std::abs(sv.norm_squared() - 1.0) < t);
}

// ---------------------------------------------------------------- measurement

TEMPLATE_TEST_CASE("probabilities", "[sv][measure]", float, double) {
  const auto p = run_text<TestType>(bell).probabilities();
  REQUIRE(p.size() == 4);
  CHECK(std::abs(p[0] - 0.5) < tol<TestType>());
  CHECK(std::abs(p[1]) < tol<TestType>());
  CHECK(std::abs(p[2]) < tol<TestType>());
  CHECK(std::abs(p[3] - 0.5) < tol<TestType>());
}

TEMPLATE_TEST_CASE("sample", "[sv][measure]", float, double) {
  const auto sv = run_text<TestType>(bell);
  const std::vector<typename CpuStateVector<TestType>::amp_t> before(sv.amps().begin(), sv.amps().end());

  const auto s = sv.sample(10000, 42);
  REQUIRE(s.size() == 10000);
  int zeros = 0;
  for (auto x : s) {
    REQUIRE((x == 0 || x == 3));  // a Bell state never gives 01 or 10
    zeros += (x == 0);
  }
  CHECK(std::abs(zeros - 5000) < 300);  // ~6 sigma

  CHECK(sv.sample(10000, 42) == s);    // same seed, same samples
  CHECK(sv.sample(10000, 43) != s);
  CHECK(std::equal(before.begin(), before.end(), sv.amps().begin()));  // sampling doesn't collapse
}

TEMPLATE_TEST_CASE("measure collapses and renormalises", "[sv][measure]", float, double) {
  std::set<int> outcomes;
  for (std::uint64_t seed = 0; seed < 20; ++seed) {
    auto sv = run_text<TestType>(bell);
    std::mt19937_64 rng(seed);
    const int r = sv.measure(0, rng);
    REQUIRE((r == 0 || r == 1));
    outcomes.insert(r);
    INFO("seed " << seed << " measured " << r);
    check_basis(sv, r ? 0b11 : 0b00);          // Bell: q1 collapses with q0
    CHECK(sv.measure(1, rng) == r);            // and agrees when measured next
    CHECK(std::abs(sv.norm_squared() - 1.0) < tol<TestType>());
  }
  CHECK(outcomes.size() == 2);  // both results show up over 20 seeds
}
