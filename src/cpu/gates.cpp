#include "qsim/gates.hpp"

#include <cmath>
#include <complex>

namespace qsim::gates {

namespace {
using Complex = std::complex<double>;
constexpr double kInvSqrt2 = 0.70710678118654752440;  // 1 / sqrt(2)
constexpr double kPiOver4 = 0.78539816339744830962;   // pi / 4
}  // namespace

Matrix H() {
  return {
      Complex{kInvSqrt2, 0.0},  Complex{kInvSqrt2, 0.0},
      Complex{kInvSqrt2, 0.0},  Complex{-kInvSqrt2, 0.0},
  };
}

Matrix X() {
  return {
      Complex{0.0, 0.0}, Complex{1.0, 0.0},
      Complex{1.0, 0.0}, Complex{0.0, 0.0},
  };
}

Matrix Y() {
  return {
      Complex{0.0, 0.0}, Complex{0.0, -1.0},
      Complex{0.0, 1.0}, Complex{0.0, 0.0},
  };
}

Matrix Z() {
  return {
      Complex{1.0, 0.0}, Complex{0.0, 0.0},
      Complex{0.0, 0.0}, Complex{-1.0, 0.0},
  };
}

Matrix S() {
  return {
      Complex{1.0, 0.0}, Complex{0.0, 0.0},
      Complex{0.0, 0.0}, Complex{0.0, 1.0},
  };
}

Matrix Sdg() {
  return {
      Complex{1.0, 0.0}, Complex{0.0, 0.0},
      Complex{0.0, 0.0}, Complex{0.0, -1.0},
  };
}

Matrix T() {
  return {
      Complex{1.0, 0.0}, Complex{0.0, 0.0},
      Complex{0.0, 0.0}, std::polar(1.0, kPiOver4),
  };
}

Matrix Tdg() {
  return {
      Complex{1.0, 0.0}, Complex{0.0, 0.0},
      Complex{0.0, 0.0}, std::polar(1.0, -kPiOver4),
  };
}

Matrix RX(double theta) {
  const double half = 0.5 * theta;
  const double c = std::cos(half);
  const double s = std::sin(half);
  return {
      Complex{c, 0.0},  Complex{0.0, -s},
      Complex{0.0, -s}, Complex{c, 0.0},
  };
}

Matrix RY(double theta) {
  const double half = 0.5 * theta;
  const double c = std::cos(half);
  const double s = std::sin(half);
  return {
      Complex{c, 0.0}, Complex{-s, 0.0},
      Complex{s, 0.0}, Complex{c, 0.0},
  };
}

Matrix RZ(double theta) {
  const double half = 0.5 * theta;
  return {
      std::polar(1.0, -half), Complex{0.0, 0.0},
      Complex{0.0, 0.0},      std::polar(1.0, half),
  };
}

Matrix U3(double theta, double phi, double lambda) {
  const double half = 0.5 * theta;
  const double c = std::cos(half);
  const double s = std::sin(half);
  // c and s go negative for many angles; std::polar requires a non-negative magnitude (libc++
  // returns NaN otherwise), so take unit phases and multiply the magnitude in separately.
  return {
      Complex{c, 0.0},
      -s * std::polar(1.0, lambda),
      s * std::polar(1.0, phi),
      c * std::polar(1.0, phi + lambda),
  };
}

Matrix SWAP() {
  return {
      Complex{1.0, 0.0}, Complex{0.0, 0.0}, Complex{0.0, 0.0}, Complex{0.0, 0.0},
      Complex{0.0, 0.0}, Complex{0.0, 0.0}, Complex{1.0, 0.0}, Complex{0.0, 0.0},
      Complex{0.0, 0.0}, Complex{1.0, 0.0}, Complex{0.0, 0.0}, Complex{0.0, 0.0},
      Complex{0.0, 0.0}, Complex{0.0, 0.0}, Complex{0.0, 0.0}, Complex{1.0, 0.0},
  };
}

}  // namespace qsim::gates