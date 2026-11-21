// OpenQASM 2.0 import: every case compares the state from QASM with the state from equivalent .qc text.
//   Run: qsim_tests "[qasm]"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <complex>
#include <numbers>
#include <string>
#include <vector>

#include "qsim/circuit.hpp"
#include "qsim/cpu_statevector.hpp"

using Catch::Matchers::ContainsSubstring;

namespace {

std::vector<std::complex<double>> state(const qsim::Circuit& c) {
  qsim::CpuStateVector<double> sv(c.num_qubits);
  sv.run(c);
  return {sv.amps().begin(), sv.amps().end()};
}

double diff(const std::string& qasm, const std::string& qc) {
  const auto a = state(qsim::parse_qasm(qasm)), b = state(qsim::parse_circuit(qc));
  REQUIRE(a.size() == b.size());
  double w = 0;
  for (std::size_t i = 0; i < a.size(); ++i) w = std::max(w, std::abs(a[i] - b[i]));
  return w;
}

const std::string hdr = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n";
std::string num(double x) {
  char b[40];
  std::snprintf(b, sizeof b, "%.17g", x);
  return b;
}

}  // namespace

TEST_CASE("QASM: basic circuit, comments, ignored statements", "[qasm]") {
  const std::string q = "// Bell pair\n" + hdr +
                        "qreg q[2];\ncreg c[2];\nh q[0];   // superpose\ncx q[0], q[1];\nbarrier q;\n"
                        "measure q -> c;\n";
  CHECK(qsim::looks_like_qasm(q));
  CHECK_FALSE(qsim::looks_like_qasm("qubits 2\nh 0"));
  CHECK(diff(q, "qubits 2\nh 0\ncx 0 1") < 1e-15);
}

TEST_CASE("QASM: parameter expressions", "[qasm]") {
  const double pi = std::numbers::pi;
  const std::string q = hdr +
                        "qreg q[1];\nrz(pi/2) q[0];\nrx(-2*pi/3) q[0];\nry(pi^2/10) q[0];\n"
                        "u3(sqrt(2), cos(pi/3)+1, -(1.5e-1)) q[0];\nu1(ln(exp(0.25))) q[0];\nu2(0, pi) q[0];\n";
  const std::string qc = "qubits 1\nrz " + num(pi / 2) + " 0\nrx " + num(-2 * pi / 3) + " 0\nry " +
                         num(pi * pi / 10) + " 0\nu3 " + num(std::sqrt(2.0)) + " " + num(std::cos(pi / 3) + 1) +
                         " -0.15 0\np 0.25 0\nu3 " + num(pi / 2) + " 0 " + num(pi) + " 0";
  CHECK(diff(q, qc) < 1e-14);
}

TEST_CASE("QASM: several registers, broadcasting", "[qasm]") {
  // First register is the lowest qubits: a -> 0,1 and b -> 2,3,4.
  CHECK(diff(hdr + "qreg a[2];\nqreg b[3];\nx b[1];\nh a;\ncx a[1], b[2];\n",
             "qubits 5\nx 3\nh 0\nh 1\ncx 1 4") < 1e-15);
  // Two-register broadcast pairs up indices.
  CHECK(diff(hdr + "qreg a[2];\nqreg b[2];\nh a;\ncx a, b;\n", "qubits 4\nh 0\nh 1\ncx 0 2\ncx 1 3") < 1e-15);
}

TEST_CASE("QASM: gates translated specially", "[qasm]") {
  // sx twice is X; sxdg undoes sx.
  CHECK(diff(hdr + "qreg q[1];\nsx q[0];\nsx q[0];\n", "qubits 1\nx 0") < 1e-15);
  CHECK(diff(hdr + "qreg q[1];\nh q[0];\nsx q[0];\nsxdg q[0];\n", "qubits 1\nh 0") < 1e-15);
  // rzz(t) = exp(-i t/2 Z(x)Z): on |++> compare with the explicit diagonal.
  const double t = 0.7;
  const std::string m = " " + num(std::cos(t / 2)) + " " + num(-std::sin(t / 2)) + " 0 0 0 0 0 0" + "  0 0 " +
                        num(std::cos(t / 2)) + " " + num(std::sin(t / 2)) + " 0 0 0 0" + "  0 0 0 0 " +
                        num(std::cos(t / 2)) + " " + num(std::sin(t / 2)) + " 0 0" + "  0 0 0 0 0 0 " +
                        num(std::cos(t / 2)) + " " + num(-std::sin(t / 2));
  CHECK(diff(hdr + "qreg q[2];\nh q;\nrzz(0.7) q[0], q[1];\n", "qubits 2\nh 0\nh 1\nmat 2 0 1" + m) < 1e-14);
  // id / u0 do nothing; CX and U are the built-in spellings.
  CHECK(diff(hdr + "qreg q[2];\nU(0.3,0.2,0.1) q[0];\nid q[1];\nCX q[0],q[1];\n",
             "qubits 2\nu3 0.3 0.2 0.1 0\ncx 0 1") < 1e-15);
  // controlled rotations and cu3
  CHECK(diff(hdr + "qreg q[2];\nh q;\ncrx(0.4) q[0],q[1];\ncry(0.5) q[1],q[0];\ncu3(0.1,0.2,0.3) q[0],q[1];\n",
             "qubits 2\nh 0\nh 1\ncrx 0.4 0 1\ncry 0.5 1 0\ncu3 0.1 0.2 0.3 0 1") < 1e-15);
}

TEST_CASE("QASM: errors name the statement", "[qasm]") {
  using qsim::parse_qasm;
  CHECK_THROWS_WITH(parse_qasm("qreg q[1];"), ContainsSubstring("OPENQASM 2.0"));
  CHECK_THROWS_WITH(parse_qasm("OPENQASM 3.0;"), ContainsSubstring("only OpenQASM 2"));
  CHECK_THROWS_WITH(parse_qasm(hdr + "qreg q[1];\nfoo q[0];"), ContainsSubstring("statement 4") &&
                                                                    ContainsSubstring("unsupported gate 'foo'"));
  CHECK_THROWS_WITH(parse_qasm(hdr + "qreg q[1];\nh q[3];"), ContainsSubstring("out of range"));
  CHECK_THROWS_WITH(parse_qasm(hdr + "qreg q[2];\ncx q[0];"), ContainsSubstring("takes 2 qubit(s)"));
  CHECK_THROWS_WITH(parse_qasm(hdr + "qreg q[1];\nrz q[0];"), ContainsSubstring("takes 1 parameter(s)"));
  CHECK_THROWS_WITH(parse_qasm(hdr + "qreg q[1];\nrz(pi/) q[0];"), ContainsSubstring("expression"));
  CHECK_THROWS_WITH(parse_qasm(hdr + "gate my a { h a; }\nqreg q[1];"), ContainsSubstring("custom gate"));
  CHECK_THROWS_WITH(parse_qasm(hdr + "qreg q[1];\ncreg c[1];\nif(c==1) x q[0];"), ContainsSubstring("if"));
  CHECK_THROWS_WITH(parse_qasm(hdr + "qreg q[1];\nreset q[0];"), ContainsSubstring("reset"));
  CHECK_THROWS_WITH(parse_qasm(hdr + "qreg q[1];\nh q[0]"), ContainsSubstring("missing ';'"));
  CHECK_THROWS_WITH(parse_qasm(hdr + "qreg a[2];\nqreg b[3];\ncx a, b;"), ContainsSubstring("same size"));
}
