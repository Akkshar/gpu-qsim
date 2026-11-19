// CUDA backend vs the CPU reference (itself checked against Qiskit). Needs an NVIDIA GPU.
//   Run: qsim_tests "[cuda]"        Skip on machines without a GPU: qsim_tests "~[cuda]"

#include <algorithm>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "qsim/circuit.hpp"
#include "qsim/cpu_statevector.hpp"
#include "qsim/cuda_statevector.hpp"

using qsim::CpuStateVector;
using qsim::CudaStateVector;

namespace {

template <class T>
double tol() { return std::is_same_v<T, float> ? 2e-5 : 1e-12; }

std::string num(double x) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.17g", x);
  return buf;
}

// Unitary k-target matrix as `mat` text: a random permutation with random phases.
std::string random_mat(int k, const std::vector<int>& qubits, std::mt19937& rng) {
  const int dim = 1 << k;
  std::vector<int> perm(dim);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  std::uniform_real_distribution<double> ph(-3.14159, 3.14159);
  std::vector<std::complex<double>> m(static_cast<std::size_t>(dim * dim));
  for (int c = 0; c < dim; ++c) m[static_cast<std::size_t>(perm[c] * dim + c)] = std::polar(1.0, ph(rng));
  std::string s = "mat " + std::to_string(k);
  for (int j = 0; j < k; ++j) s += " " + std::to_string(qubits[static_cast<std::size_t>(j)]);
  for (const auto& z : m) s += " " + num(z.real()) + " " + num(z.imag());
  return s;
}

// Every op kind the parser knows, on random qubits.
std::string random_circuit(int n, int depth, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> ang(-6.3, 6.3);
  std::vector<int> qs(static_cast<std::size_t>(n));
  std::iota(qs.begin(), qs.end(), 0);
  const char* fixed1[] = {"h", "x", "y", "z", "s", "sdg", "t", "tdg"};
  const char* ctrl1[] = {"cx", "cy", "cz", "ch"};
  std::string text = "qubits " + std::to_string(n) + "\n";
  for (int d = 0; d < depth; ++d) {
    std::shuffle(qs.begin(), qs.end(), rng);
    const std::string a = std::to_string(qs[0]), b = std::to_string(qs[1]), c = std::to_string(qs[2]);
    switch (rng() % 10) {
      case 0: text += std::string(fixed1[rng() % 8]) + " " + a; break;
      case 1: text += "rx " + num(ang(rng)) + " " + a; break;
      case 2: text += "u3 " + num(ang(rng)) + " " + num(ang(rng)) + " " + num(ang(rng)) + " " + a; break;
      case 3: text += std::string(ctrl1[rng() % 4]) + " " + a + " " + b; break;
      case 4: text += "crz " + num(ang(rng)) + " " + a + " " + b; break;
      case 5: text += "swap " + a + " " + b; break;
      case 6: text += "ccx " + a + " " + b + " " + c; break;
      case 7: text += "cswap " + a + " " + b + " " + c; break;
      default: {
        const int k = 1 + static_cast<int>(rng() % static_cast<unsigned>(std::min(n, 4)));
        text += random_mat(k, qs, rng);
      }
    }
    text += "\n";
  }
  return text;
}

template <class T>
double max_diff_vs_cpu(const std::string& text) {
  const auto c = qsim::parse_circuit(text);
  CpuStateVector<double> ref(c.num_qubits);
  ref.run(c);
  CudaStateVector<T> gpu(c.num_qubits);
  gpu.run(c);
  const auto got = gpu.download();
  REQUIRE(got.size() == ref.size());
  double worst = 0;
  for (std::size_t i = 0; i < got.size(); ++i)
    worst = std::max(worst, std::abs(std::complex<double>(got[i]) - ref.amps()[i]));
  return worst;
}

const std::string bell = "qubits 2\nh 0\ncx 0 1\n";

}  // namespace

TEMPLATE_TEST_CASE("GPU starts in |0>, round-trips upload/download", "[cuda]", float, double) {
  CudaStateVector<TestType> sv(5);
  auto a = sv.download();
  REQUIRE(a.size() == 32);
  CHECK(a[0] == std::complex<TestType>(1, 0));
  for (std::size_t i = 1; i < a.size(); ++i) CHECK(a[i] == std::complex<TestType>(0, 0));

  for (std::size_t i = 0; i < a.size(); ++i) a[i] = {TestType(i), TestType(-0.5 * double(i))};
  sv.upload(a);
  CHECK(sv.download() == a);
  sv.reset();
  CHECK(sv.download()[0] == std::complex<TestType>(1, 0));
  CHECK_THROWS(CudaStateVector<TestType>(0));
}

TEMPLATE_TEST_CASE("GPU matches CPU: a 1-qubit gate on every qubit position", "[cuda]", float, double) {
  for (int q = 0; q < 9; ++q) {
    std::string text = "qubits 9\n";
    for (int j = 0; j < 9; ++j) text += "ry " + num(0.3 + 0.1 * j) + " " + std::to_string(j) + "\n";
    text += "u3 1.1 -0.4 2.2 " + std::to_string(q) + "\n";
    INFO("target " << q);
    CHECK(max_diff_vs_cpu<TestType>(text) < tol<TestType>());
  }
}

TEMPLATE_TEST_CASE("GPU matches CPU: controls and multi-target ops", "[cuda]", float, double) {
  std::mt19937 rng(11);
  std::string text = "qubits 7\n";
  for (int j = 0; j < 7; ++j) text += "h " + std::to_string(j) + "\nt " + std::to_string(j) + "\n";
  text += "cx 0 6\ncx 6 0\ncy 3 5\nch 2 4\nccx 5 1 3\nccx 0 6 2\ncswap 2 6 0\ncrz 0.3 6 1\nswap 0 6\n";
  text += random_mat(2, {5, 1}, rng) + "\n";
  text += random_mat(3, {4, 0, 6}, rng) + "\n";
  text += random_mat(6, {6, 2, 0, 5, 3, 1}, rng) + "\n";  // largest op the parser allows
  CHECK(max_diff_vs_cpu<TestType>(text) < tol<TestType>());
}

TEMPLATE_TEST_CASE("GPU matches CPU: every matrix size, scattered and controlled", "[cuda]", float, double) {
  // One op of each target count 2..6 (each k has its own kernel), targets out of order and spread
  // over low and high qubits, plus controlled 2-target ops.
  std::mt19937 rng(23);
  std::string text = "qubits 10\n";
  for (int j = 0; j < 10; ++j) text += "ry " + num(0.2 + 0.15 * j) + " " + std::to_string(j) + "\n";
  text += random_mat(2, {9, 0}, rng) + "\n";
  text += random_mat(3, {1, 8, 4}, rng) + "\n";
  text += random_mat(4, {7, 0, 9, 3}, rng) + "\n";
  text += random_mat(5, {2, 9, 0, 6, 5}, rng) + "\n";
  text += random_mat(6, {8, 1, 3, 9, 0, 4}, rng) + "\n";
  text += random_mat(5, {0, 1, 2, 3, 4}, rng) + "\n";  // all low qubits
  text += random_mat(5, {9, 8, 7, 6, 5}, rng) + "\n";  // all high qubits, descending
  text += "cswap 0 9 5\ncswap 9 1 2\n";
  CHECK(max_diff_vs_cpu<TestType>(text) < tol<TestType>());
}

TEMPLATE_TEST_CASE("GPU matches CPU: random circuits", "[cuda][random]", float, double) {
  const double t = std::is_same_v<TestType, float> ? 1e-4 : 1e-11;
  for (unsigned seed = 1; seed <= 12; ++seed) {
    const int n = 3 + static_cast<int>(seed);  // 4 .. 15 qubits
    INFO("seed " << seed << ", " << n << " qubits");
    CHECK(max_diff_vs_cpu<TestType>(random_circuit(n, 150, seed)) < t);
  }
}

TEMPLATE_TEST_CASE("GPU measurement matches CPU", "[cuda][measure]", float, double) {
  const auto c = qsim::parse_circuit(bell);
  CudaStateVector<TestType> gpu(2);
  gpu.run(c);
  CpuStateVector<TestType> cpu(2);
  cpu.run(c);

  CHECK(std::abs(gpu.norm_squared() - 1.0) < tol<TestType>());
  const auto p = gpu.probabilities();
  REQUIRE(p.size() == 4);
  CHECK(std::abs(p[0] - 0.5) < tol<TestType>());
  CHECK(std::abs(p[3] - 0.5) < tol<TestType>());

  // Same algorithm and the same uniform draws as the CPU, so the same seed gives the same samples.
  const auto s = gpu.sample(10000, 42);
  CHECK(s == cpu.sample(10000, 42));
  CHECK(gpu.sample(10000, 42) == s);

  for (std::uint64_t seed = 0; seed < 10; ++seed) {
    CudaStateVector<TestType> g(2);
    g.run(c);
    CpuStateVector<TestType> h(2);
    h.run(c);
    std::mt19937_64 rg(seed), rh(seed);
    const int r = g.measure(0, rg);
    INFO("seed " << seed);
    CHECK(r == h.measure(0, rh));
    const auto a = g.download();
    CHECK(std::abs(std::abs(std::complex<double>(a[r ? 3 : 0])) - 1.0) < tol<TestType>());
    CHECK(std::abs(g.norm_squared() - 1.0) < tol<TestType>());
  }
}

TEMPLATE_TEST_CASE("GPU sample on a larger random state matches CPU", "[cuda][measure]", float, double) {
  const auto c = qsim::parse_circuit(random_circuit(12, 120, 77));
  CudaStateVector<TestType> gpu(12);
  gpu.run(c);
  CpuStateVector<double> cpu(12);
  cpu.run(c);
  const auto gs = gpu.sample(4000, 5);
  const auto cs = cpu.sample(4000, 5);
  // The GPU's parallel scan rounds differently from the CPU's running sum, so a draw that lands
  // within rounding of a boundary can pick the neighbour. Expect (almost) all to agree.
  std::size_t same = 0;
  for (std::size_t i = 0; i < gs.size(); ++i) same += gs[i] == cs[i];
  CHECK(same >= gs.size() - (std::is_same_v<TestType, float> ? 40 : 2));
}
