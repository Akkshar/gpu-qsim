#pragma once
// Bit tricks for walking a state vector. These become the heart of the GPU kernels in M2, so they
// live in a header (M2 will mark them __host__ __device__).

#include <cassert>
#include <cstdint>

namespace qsim {

// Inserts a 0 bit at position `pos` of k.
// Bits of k below `pos` stay where they are; bits at or above `pos` move up by one.
//   insert_zero_bit(0b1011, 2) == 0b10011
//   insert_zero_bit(0b1011, 0) == 0b10110
// Why it matters: a 1-qubit gate on qubit t pairs every index with bit t = 0 with the same index
// with bit t = 1. If k counts pairs (0 .. 2^(n-1) - 1), insert_zero_bit(k, t) is the pair's
// "bit t = 0" index, with no branches and no wasted iterations. Two or three shifts and masks.
inline std::uint64_t insert_zero_bit(std::uint64_t k, int pos) {
  assert(pos >= 0 && pos < 64);  // shifting a 64-bit value by >= 64 or < 0 is undefined behaviour
  const std::uint64_t low_mask = (std::uint64_t{1} << pos) - std::uint64_t{1};
  const std::uint64_t low = k & low_mask;
  const std::uint64_t high = (k & ~low_mask) << 1;
  return high | low;
}

}  // namespace qsim