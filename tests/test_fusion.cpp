// Gate fusion: fused circuits must give exactly the same state as the original, and the pass must
// follow its rules (block size cap, single ops untouched, order preserved).
//   Run: qsim_tests "[fusion]"

#include <algorithm>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "qsim/circuit.hpp"
#include "qsim/cpu_statevector.hpp"
#include "qsim/cuda_statevector.hpp"
#include "qsim/fusion.hpp"

using qsim::Circuit;
using qsim::CpuStateVector;

namespace {

std::string num(double x) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.17g", x);
  return buf;
}

// Mixed circuit biased towards neighbouring qubits (so there is something to fuse), with every
// op kind: 1-qubit, parameterised, controlled, swap, ccx/cswap, explicit 1-3 qubit matrices.
std::string circuit_text(int n, int depth, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> ang(-6.3, 6.3);
  auto near = [&](int q, int d) { return (q + 1 + static_cast<int>(rng() % static_cast<unsigned>(d))) % n; };
  std::string t = "qubits " + std::to_string(n) + "\n";
  for (int i = 0; i < depth; ++i) {
    const int a = static_cast<int>(rng() % static_cast<unsigned>(n));
    int b = near(a, 2), c = near(b, 2);
    if (c == a) c = (b + 1) % n == a ? (b + 2) % n : (b + 1) % n;
    const std::string A = std::to_string(a), B = std::to_string(b), C = std::to_string(c);
    switch (rng() % 9) {
      case 0: t += "h " + A; break;
      case 1: t += "u3 " + num(ang(rng)) + " " + num(ang(rng)) + " " + num(ang(rng)) + " " + A; break;
      case 2: t += "p " + num(ang(rng)) + " " + A; break;
      case 3: t += "cx " + A + " " + B; break;
      case 4: t += "cp " + num(ang(rng)) + " " + A + " " + B; break;
      case 5: t += "swap " + A + " " + B; break;
      case 6: t += (c != a && c != b) ? "ccx " + A + " " + B + " " + C : "cz " + A + " " + B; break;
      case 7: t += (c != a && c != b) ? "cswap " + A + " " + B + " " + C : "cy " + A + " " + B; break;
      default: {  // 2-qubit permutation-with-phases matrix
        std::vector<int> perm(4);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        std::vector<std::complex<double>> m(16);
        for (int col = 0; col < 4; ++col) m[static_cast<std::size_t>(perm[col] * 4 + col)] = std::polar(1.0, ang(rng));
        t += "mat 2 " + B + " " + A;
        for (const auto& z : m) t += " " + num(z.real()) + " " + num(z.imag());
      }
    }
    t += "\n";
  }
  return t;
}

double max_diff(const std::vector<std::complex<double>>& a, std::span<const std::complex<double>> b) {
  double w = 0;
  for (std::size_t i = 0; i < a.size(); ++i) w = std::max(w, std::abs(a[i] - b[i]));
  return w;
}

std::vector<std::complex<double>> run_cpu(const Circuit& c) {
  CpuStateVector<double> sv(c.num_qubits);
  sv.run(c);
  return {sv.amps().begin(), sv.amps().end()};
}

int qubits_touched(const qsim::Op& op) { return op.arity() + static_cast<int>(op.controls.size()); }

}  // namespace

TEST_CASE("fusion never changes the result", "[fusion]") {
  for (unsigned seed = 1; seed <= 8; ++seed) {
    const auto c = qsim::parse_circuit(circuit_text(8, 120, seed));
    const auto want = run_cpu(c);
    for (int k = 1; k <= 6; ++k) {
      const auto f = qsim::fuse_circuit(c, k);
      CpuStateVector<double> sv(c.num_qubits);
      sv.run(f);
      INFO("seed " << seed << ", fuse " << k << ": " << c.ops.size() << " -> " << f.ops.size() << " ops");
      CHECK(max_diff(want, sv.amps()) < 1e-11);
      for (const auto& op : f.ops)
        if (op.name == "fused") CHECK(op.arity() <= k);
    }
  }
}

TEST_CASE("fusion merges what it should and leaves the rest alone", "[fusion]") {
  using qsim::fuse_circuit;
  using qsim::parse_circuit;

  // Same qubit, three gates -> one 1-qubit block.
  auto f = fuse_circuit(parse_circuit("qubits 3\nh 0\nt 0\nh 0"), 1);
  REQUIRE(f.ops.size() == 1);
  CHECK(f.ops[0].name == "fused");
  CHECK(f.ops[0].targets == std::vector<int>{0});

  // Block cap: with K = 1 each qubit change starts a new block.
  CHECK(fuse_circuit(parse_circuit("qubits 3\nh 0\nh 1\ncx 0 1"), 1).ops.size() == 3);
  // With K = 2, all three fit on {0, 1}; the control becomes part of the dense block.
  f = fuse_circuit(parse_circuit("qubits 3\nh 0\nh 1\ncx 0 1"), 2);
  REQUIRE(f.ops.size() == 1);
  CHECK(f.ops[0].targets == std::vector<int>{0, 1});
  CHECK(f.ops[0].controls.empty());

  // A lone op stays exactly as it was, controls and all.
  f = fuse_circuit(parse_circuit("qubits 3\ncx 2 0"), 4);
  REQUIRE(f.ops.size() == 1);
  CHECK(f.ops[0].name == "cx");
  CHECK(f.ops[0].controls == std::vector<int>{2});

  // An op wider than K passes through and splits the blocks around it.
  f = fuse_circuit(parse_circuit("qubits 4\nh 0\nh 0\nccx 0 1 2\nh 3\nh 3"), 2);
  REQUIRE(f.ops.size() == 3);
  CHECK(f.ops[1].name == "ccx");

  // Order preserved: a gate on qubit 1 between two blocks on qubit 0 can't be skipped over.
  f = fuse_circuit(parse_circuit("qubits 2\nh 0\nx 1\nh 0"), 1);
  CHECK(f.ops.size() == 3);
}

TEST_CASE("fused blocks never touch more than K qubits", "[fusion]") {
  const auto c = qsim::parse_circuit(circuit_text(12, 300, 99));
  for (int k = 1; k <= 6; ++k) {
    const auto f = qsim::fuse_circuit(c, k);
    INFO("fuse " << k);
    for (const auto& op : f.ops) CHECK((op.name != "fused" || qubits_touched(op) <= k));
    CHECK(f.ops.size() <= c.ops.size());
  }
  CHECK_THROWS(qsim::fuse_circuit(c, 0));
  CHECK_THROWS(qsim::fuse_circuit(c, 7));
}

TEMPLATE_TEST_CASE("GPU runs fused circuits correctly", "[fusion][cuda]", float, double) {
  const double tol = std::is_same_v<TestType, float> ? 1e-4 : 1e-11;
  for (unsigned seed = 1; seed <= 4; ++seed) {
    const auto c = qsim::parse_circuit(circuit_text(14, 200, seed));
    const auto want = run_cpu(c);
    for (int k = 2; k <= 6; ++k) {
      const auto f = qsim::fuse_circuit(c, k);
      qsim::CudaStateVector<TestType> gpu(c.num_qubits);
      gpu.run(f);
      const auto got = gpu.download();
      double w = 0;
      for (std::size_t i = 0; i < got.size(); ++i) w = std::max(w, std::abs(std::complex<double>(got[i]) - want[i]));
      INFO("seed " << seed << ", fuse " << k);
      CHECK(w < tol);
    }
  }
}
