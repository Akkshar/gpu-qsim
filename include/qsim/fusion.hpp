#pragma once
// Gate fusion: merge runs of consecutive ops into dense matrices on at most `max_qubits` qubits.
//
// Every kernel already runs at (or near) the memory floor: one pass over the state per op, whatever
// the op. So the only way to go faster is fewer passes. A fused block of m ops on q <= max_qubits
// qubits costs one pass instead of m, at the price of a 2^q x 2^q dense matrix (more arithmetic per
// amplitude, which is free while the kernel stays memory-bound).
//
// Greedy and order-preserving: walk the ops; keep adding to the current block while the union of
// qubits it touches (targets + controls) stays within max_qubits; otherwise close the block and
// start a new one. No reordering, so correctness doesn't depend on commutation rules.
//
//   * A block with a single op is emitted unchanged (a lone cx keeps its control and touches half the
//     state, which is cheaper than a dense 4x4 over all of it).
//   * Fused ops are named "fused", have no controls, and targets in ascending qubit order.

#include <cstddef>

#include "qsim/circuit.hpp"

namespace qsim {

Circuit fuse_circuit(const Circuit& c, int max_qubits);  // max_qubits in 1..6

}  // namespace qsim
