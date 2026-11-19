#pragma once
// Randomness shared by every backend, so a given seed gives the same draws everywhere.

#include <cstdint>
#include <random>

namespace qsim {

// Uniform double in [0, 1) from the top 53 bits of one engine draw. The algorithms behind
// std::uniform_real_distribution / std::discrete_distribution are implementation-defined, so the
// same seed gives different results on MSVC and libstdc++; mt19937_64's raw output is specified.
inline double uniform01(std::mt19937_64& rng) {
  return static_cast<double>(rng() >> 11) * 0x1.0p-53;
}

}  // namespace qsim
