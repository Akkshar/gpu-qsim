// M1 TODO (yours): every gate as a row-major matrix. The exact definitions, including global phase,
// are in include/qsim/gates.hpp. tests/test_bits_gates.cpp ([gates]) checks them.

#include "qsim/gates.hpp"

#include <stdexcept>

#define QSIM_TODO(what) throw std::logic_error("TODO(M1): " what " (src/cpu/gates.cpp)")

namespace qsim::gates {

Matrix H() { QSIM_TODO("gates::H"); }
Matrix X() { QSIM_TODO("gates::X"); }
Matrix Y() { QSIM_TODO("gates::Y"); }
Matrix Z() { QSIM_TODO("gates::Z"); }
Matrix S() { QSIM_TODO("gates::S"); }
Matrix Sdg() { QSIM_TODO("gates::Sdg"); }
Matrix T() { QSIM_TODO("gates::T"); }
Matrix Tdg() { QSIM_TODO("gates::Tdg"); }
Matrix RX(double) { QSIM_TODO("gates::RX"); }
Matrix RY(double) { QSIM_TODO("gates::RY"); }
Matrix RZ(double) { QSIM_TODO("gates::RZ"); }
Matrix U3(double, double, double) { QSIM_TODO("gates::U3"); }
Matrix SWAP() { QSIM_TODO("gates::SWAP"); }

}  // namespace qsim::gates
