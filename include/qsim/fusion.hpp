#pragma once
// Gate fusion: merge runs of consecutive ops into dense matrices on at most `max_qubits` qubits.
//
// Every kernel already runs at (or near) the memory floor: one pass over the state per op, whatever
// the op. So the only way to go faster is fewer passes. A fused block of m ops on q <= max_qubits
// qubits costs one pass instead of m, at the price of a 2^q x 2^q dense matrix (more arithmetic per
// amplitude, which is free while the kernel stays memory-bound).
//
// Two strategies:
//   InOrder (M3.2): one open block; keep adding ops while the union of touched qubits (targets +
//            controls) stays <= max_qubits, otherwise close it and start a new one.
//   Commuting (M3.4): several open blocks on disjoint qubit sets. Gates on disjoint qubits
//            commute, so a gate can join an earlier block past gates on other qubits; blocks it
//            touches are merged while they fit. Exact (no approximations), just fewer passes.
//            3-4x fewer passes on layered/random circuits, but its keep-the-biggest-blocks heuristic
//            does slightly worse than InOrder on QFT/QAOA ladders.
//   Best (default): run both, keep the one with the lower estimated cost (one full pass per op,
//            1/2^c of a pass for an op with c controls). Fusion is host-side and takes milliseconds.
//
//   * A block with a single op is emitted unchanged (a lone cx keeps its control and touches half the
//     state, which is cheaper than a dense 4x4 over all of it).
//   * Fused ops are named "fused", have no controls, and targets in ascending qubit order.

#include <cstddef>

#include "qsim/circuit.hpp"

namespace qsim {

enum class FusionStrategy { InOrder, Commuting, Best };

Circuit fuse_circuit(const Circuit& c, int max_qubits,  // max_qubits in 1..6
                     FusionStrategy strategy = FusionStrategy::Best);

// Estimated cost in full passes over the state: 1 per op, 1/2^c for an op with c controls.
double pass_cost(const Circuit& c);

}  // namespace qsim
