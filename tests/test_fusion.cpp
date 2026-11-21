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
#ifdef QSIM_HAS_CUDA
#include "qsim/cuda_statevector.hpp"
#endif
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

TEST_CASE("fusion never changes the result (both strategies)", "[fusion]") {
  using S = qsim::FusionStrategy;
  for (unsigned seed = 1; seed <= 8; ++seed) {
    const auto c = qsim::parse_circuit(circuit_text(8, 120, seed));
    const auto want = run_cpu(c);
    for (S s : {S::InOrder, S::Commuting, S::Best}) {
      for (int k = 1; k <= 6; ++k) {
        const auto f = qsim::fuse_circuit(c, k, s);
        CpuStateVector<double> sv(c.num_qubits);
        sv.run(f);
        INFO("seed " << seed << ", strategy " << static_cast<int>(s) << ", fuse " << k << ": " << c.ops.size()
                     << " -> " << f.ops.size() << " ops");
        CHECK(max_diff(want, sv.amps()) < 1e-11);
        for (const auto& op : f.ops)
          if (op.name == "fused") CHECK(op.arity() <= k);
      }
    }
  }
}

TEST_CASE("commuting fusion is exact on wide, shallow-layer circuits", "[fusion]") {
  // The case it exists for: layers of 1-qubit gates on every qubit plus neighbour 2-qubit gates.
  // Many blocks stay open at once; check the result and that it never needs more passes.
  for (int n : {6, 9, 12}) {
    std::string t = "qubits " + std::to_string(n) + "\n";
    for (int layer = 0; layer < 12; ++layer) {
      for (int q = 0; q < n; ++q)
        t += "u3 " + num(0.1 * (layer + 1)) + " " + num(0.07 * q) + " " + num(-0.3) + " " + std::to_string(q) + "\n";
      for (int q = layer % 2; q + 1 < n; q += 2) t += "cz " + std::to_string(q) + " " + std::to_string(q + 1) + "\n";
      if (layer % 3 == 2) t += "cx " + std::to_string(n - 1) + " 0\n";  // a long-range gate across all blocks
    }
    const auto c = qsim::parse_circuit(t);
    const auto want = run_cpu(c);
    for (int k = 2; k <= 5; ++k) {
      const auto fc = qsim::fuse_circuit(c, k, qsim::FusionStrategy::Commuting);
      const auto fi = qsim::fuse_circuit(c, k, qsim::FusionStrategy::InOrder);
      CpuStateVector<double> sv(n);
      sv.run(fc);
      INFO(n << " qubits, fuse " << k << ": in-order " << fi.ops.size() << " ops, commuting " << fc.ops.size());
      CHECK(max_diff(want, sv.amps()) < 1e-11);
      CHECK(fc.ops.size() <= fi.ops.size());
    }
  }
}

TEST_CASE("Best picks the cheaper of the two strategies", "[fusion]") {
  using S = qsim::FusionStrategy;
  for (unsigned seed = 1; seed <= 6; ++seed) {
    const auto c = qsim::parse_circuit(circuit_text(10, 150, seed));
    for (int k = 2; k <= 5; ++k) {
      const double best = qsim::pass_cost(qsim::fuse_circuit(c, k, S::Best));
      const double in = qsim::pass_cost(qsim::fuse_circuit(c, k, S::InOrder));
      const double co = qsim::pass_cost(qsim::fuse_circuit(c, k, S::Commuting));
      INFO("seed " << seed << ", fuse " << k << ": in-order " << in << ", commuting " << co << ", best " << best);
      CHECK(best == std::min(in, co));
    }
  }
  // Cost model: a lone controlled op costs half a pass per control's worth of skipped state.
  CHECK(qsim::pass_cost(qsim::parse_circuit("qubits 3\nh 0\ncx 0 1\nccx 0 1 2")) == 1.0 + 0.5 + 0.25);
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

  // A gate on qubit 1 between two gates on qubit 0: the in-order strategy can't skip over it...
  f = fuse_circuit(parse_circuit("qubits 2\nh 0\nx 1\nh 0"), 1, qsim::FusionStrategy::InOrder);
  CHECK(f.ops.size() == 3);
  // ...the commuting one can (x 1 commutes with h 0): {h 0, h 0} and {x 1}.
  f = fuse_circuit(parse_circuit("qubits 2\nh 0\nx 1\nh 0"), 1, qsim::FusionStrategy::Commuting);
  CHECK(f.ops.size() == 2);
  // Interleaved 1-qubit gates on two qubits: 4 passes in order, 2 commuting.
  CHECK(fuse_circuit(parse_circuit("qubits 2\nh 0\nh 1\nt 0\nt 1"), 1, qsim::FusionStrategy::InOrder).ops.size() == 4);
  CHECK(fuse_circuit(parse_circuit("qubits 2\nh 0\nh 1\nt 0\nt 1"), 1, qsim::FusionStrategy::Commuting).ops.size() == 2);
  // But never across a gate that shares a qubit: cx 0 1 sits between the two h 0's.
  f = fuse_circuit(parse_circuit("qubits 3\nh 0\ncx 0 1\nh 0\nh 2"), 1, qsim::FusionStrategy::Commuting);
  REQUIRE(f.ops.size() == 4);
  CHECK(f.ops[1].name == "cx");
}

TEST_CASE("fused blocks never touch more than K qubits", "[fusion]") {
  const auto c = qsim::parse_circuit(circuit_text(12, 300, 99));
  for (auto s : {qsim::FusionStrategy::InOrder, qsim::FusionStrategy::Commuting}) {
    for (int k = 1; k <= 6; ++k) {
      const auto f = qsim::fuse_circuit(c, k, s);
      INFO("fuse " << k << ", strategy " << static_cast<int>(s));
      for (const auto& op : f.ops) CHECK((op.name != "fused" || qubits_touched(op) <= k));
      CHECK(f.ops.size() <= c.ops.size());
    }
  }
  CHECK_THROWS(qsim::fuse_circuit(c, 0));
  CHECK_THROWS(qsim::fuse_circuit(c, 7));
}

#ifdef QSIM_HAS_CUDA
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
#endif  // QSIM_HAS_CUDA
