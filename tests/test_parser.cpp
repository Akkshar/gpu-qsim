// Tests for the circuit parser (support code; these should pass from the start).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <string>

#include "qsim/circuit.hpp"

using Catch::Matchers::ContainsSubstring;
using qsim::cdouble;
using qsim::parse_circuit;

TEST_CASE("header, comments and blank lines", "[parser]") {
  const auto c = parse_circuit("# a comment\n\n   qubits 3   # trailing comment\n\n");
  CHECK(c.num_qubits == 3);
  CHECK(c.ops.empty());

  CHECK_THROWS_WITH(parse_circuit(""), ContainsSubstring("qubits"));
  CHECK_THROWS_WITH(parse_circuit("h 0"), ContainsSubstring("line 1") && ContainsSubstring("qubits N"));
  CHECK_THROWS_WITH(parse_circuit("qubits 0"), ContainsSubstring("1..40"));
  CHECK_THROWS_WITH(parse_circuit("qubits x"), ContainsSubstring("integer"));
  CHECK_THROWS_WITH(parse_circuit("qubits 3\nqubits 3"), ContainsSubstring("twice"));
}

TEST_CASE("mat keeps target order and entries", "[parser]") {
  std::string text = "qubits 4\nmat 2 3 1";
  for (int i = 0; i < 16; ++i) text += " " + std::to_string(i) + " " + std::to_string(-0.5 * i);
  const auto c = parse_circuit(text);
  REQUIRE(c.ops.size() == 1);
  const auto& op = c.ops[0];
  CHECK(op.name == "mat");
  CHECK(op.targets == std::vector<int>{3, 1});
  CHECK(op.controls.empty());
  REQUIRE(op.matrix.size() == 16);
  for (int i = 0; i < 16; ++i) CHECK(op.matrix[i] == cdouble(i, -0.5 * i));
}

TEST_CASE("mat errors", "[parser]") {
  CHECK_THROWS_WITH(parse_circuit("qubits 2\nmat 1 0 1 0 0 0"), ContainsSubstring("needs 4 complex entries"));
  CHECK_THROWS_WITH(parse_circuit("qubits 2\nmat 0"), ContainsSubstring("arity"));
  CHECK_THROWS_WITH(parse_circuit("qubits 9\nmat 7 0 1 2 3 4 5 6"), ContainsSubstring("arity"));
  CHECK_THROWS_WITH(parse_circuit("qubits 2\nmat 2 1 1 0"), ContainsSubstring("used twice"));
}

TEST_CASE("errors carry the line number", "[parser]") {
  CHECK_THROWS_WITH(parse_circuit("qubits 2\nh 2"), ContainsSubstring("line 2") && ContainsSubstring("out of range"));
  CHECK_THROWS_WITH(parse_circuit("qubits 2\n\nfoo 0"), ContainsSubstring("line 3") && ContainsSubstring("unknown gate"));
  CHECK_THROWS_WITH(parse_circuit("qubits 3\ncx 1 1"), ContainsSubstring("used twice"));
  CHECK_THROWS_WITH(parse_circuit("qubits 3\ncx 0"), ContainsSubstring("takes 0 parameter(s) and 2 qubit(s)"));
  CHECK_THROWS_WITH(parse_circuit("qubits 3\nrx abc 0"), ContainsSubstring("line 2") && ContainsSubstring("number"));
  CHECK_THROWS_WITH(parse_circuit("qubits 3\nh 1.5"), ContainsSubstring("integer"));
}
