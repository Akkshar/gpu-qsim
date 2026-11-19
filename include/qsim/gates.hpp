#pragma once
// Gate matrices, row-major. YOU implement these in src/cpu/gates.cpp (M1).
//
// Conventions must match Qiskit exactly, including global phase, because controlled versions
// (cx, crz, ...) make a gate's global phase observable:
//   H   = 1/sqrt2 [[1, 1], [1, -1]]
//   S   = diag(1, i)                 Sdg = S^dagger
//   T   = diag(1, e^{i pi/4})        Tdg = T^dagger
//   RX(t) = [[cos t/2, -i sin t/2], [-i sin t/2, cos t/2]]
//   RY(t) = [[cos t/2, -sin t/2], [sin t/2, cos t/2]]
//   RZ(t) = diag(e^{-i t/2}, e^{i t/2})
//   U3(t, p, l) = [[cos t/2, -e^{i l} sin t/2], [e^{i p} sin t/2, e^{i(p+l)} cos t/2]]
//   SWAP: 4x4, swaps |01> and |10>
// tests/test_bits_gates.cpp checks all of these.

#include "qsim/circuit.hpp"

namespace qsim::gates {

Matrix H();
Matrix X();
Matrix Y();
Matrix Z();
Matrix S();
Matrix Sdg();
Matrix T();
Matrix Tdg();
Matrix RX(double theta);
Matrix RY(double theta);
Matrix RZ(double theta);
Matrix U3(double theta, double phi, double lambda);
Matrix SWAP();

}  // namespace qsim::gates
