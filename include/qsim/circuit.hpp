#pragma once
// Circuit IR: a circuit is a qubit count plus a flat list of ops. Every op is a dense unitary on its
// target qubits, applied only where all of its control qubits are 1.
//
// Conventions (same as Qiskit, so results compare directly):
//   * Qubit 0 is the least significant bit of a state index: |q2 q1 q0>, index = q0 + 2*q1 + 4*q2.
//   * An op's matrix is row-major, dimension 2^k x 2^k for k = targets.size().
//   * targets[0] is the least significant bit of the matrix's row/column index, targets[1] the next...
//     So `mat 2 3 1 <M>` means: matrix index bit 0 <-> qubit 3, matrix index bit 1 <-> qubit 1.
//
// Text format (*.qc), one statement per line, '#' starts a comment:
//   qubits 3                 first statement, required
//   h 0                      1-qubit gates: h x y z s sdg t tdg
//   rx 0.25 2                parameterised: rx ry rz <theta> q,  u3 <theta> <phi> <lambda> q,
//                            p <lambda> q  (phase gate diag(1, e^{i lambda}))
//   cx 0 1                   controls come first, then targets (Qiskit order): cx cy cz ch,
//                            crx/cry/crz <theta> c t, cp <lambda> c t, cu3 <theta> <phi> <lambda> c t
//   swap 0 2                 ccx c1 c2 t,  cswap c t1 t2
//   mat 1 0  re im re im re im re im      explicit matrix: mat <k> <k target qubits> <(2^k)^2 complex entries>

#include <complex>
#include <string>
#include <string_view>
#include <vector>

namespace qsim {

using cdouble = std::complex<double>;
using Matrix = std::vector<cdouble>;  // row-major, (2^k) x (2^k)

struct Op {
  std::string name;           // gate name from the file ("cx", "mat", ...), for messages and stats
  std::vector<int> targets;   // targets[0] = least significant bit of the matrix index
  std::vector<int> controls;  // all must be 1 for the op to act
  Matrix matrix;              // the *uncontrolled* gate: for cx this is X, for ccx it's X, for crz it's RZ

  int arity() const { return static_cast<int>(targets.size()); }
};

struct Circuit {
  int num_qubits = 0;
  std::vector<Op> ops;
};

// Throw std::runtime_error("line N: ...") on malformed input.
Circuit parse_circuit(std::string_view text);

// OpenQASM 2.0 subset (see src/io/qasm.cpp for what's supported). Errors: "qasm statement N: ...".
Circuit parse_qasm(std::string_view text);
bool looks_like_qasm(std::string_view text);  // starts with "OPENQASM" (after comments/whitespace)

// Reads a .qc or OpenQASM file (detected by content, not extension).
Circuit load_circuit(const std::string& path);

}  // namespace qsim
