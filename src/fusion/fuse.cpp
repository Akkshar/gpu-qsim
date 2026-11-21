// Greedy order-preserving gate fusion (see include/qsim/fusion.hpp).

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "qsim/fusion.hpp"

namespace qsim {
namespace {

std::vector<int> qubits_of(const Op& op) {
  std::vector<int> q(op.targets);
  q.insert(q.end(), op.controls.begin(), op.controls.end());
  std::sort(q.begin(), q.end());
  return q;
}

std::vector<int> merged(const std::vector<int>& a, const std::vector<int>& b) {
  std::vector<int> out;
  std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
  return out;
}

// U <- E(op) * U, where U is the D x D matrix of the block so far on qubits `qs` (bit j of a matrix
// index <-> qs[j]) and E(op) is op embedded on those qubits. Done column by column: each column of U
// is a 2^q-amplitude "state" and gets the op applied to it like in the simulator.
void apply_to_block(Matrix& U, const std::vector<int>& qs, const Op& op) {
  const std::size_t D = std::size_t{1} << qs.size();
  auto pos = [&](int q) {
    return static_cast<int>(std::lower_bound(qs.begin(), qs.end(), q) - qs.begin());
  };
  const int k = op.arity();
  const std::size_t K = std::size_t{1} << k;
  std::vector<int> tpos(static_cast<std::size_t>(k));
  std::size_t tmask = 0, cmask = 0;
  for (int j = 0; j < k; ++j) {
    tpos[static_cast<std::size_t>(j)] = pos(op.targets[static_cast<std::size_t>(j)]);
    tmask |= std::size_t{1} << tpos[static_cast<std::size_t>(j)];
  }
  for (int c : op.controls) cmask |= std::size_t{1} << pos(c);

  // member[m] = block-index bits for op-matrix index m (targets[j] <-> bit j of m).
  std::vector<std::size_t> member(K, 0);
  for (std::size_t m = 0; m < K; ++m)
    for (int j = 0; j < k; ++j)
      if ((m >> j) & 1) member[m] |= std::size_t{1} << tpos[static_cast<std::size_t>(j)];

  std::vector<cdouble> in(K), col(D);
  for (std::size_t c = 0; c < D; ++c) {  // each column of U
    for (std::size_t i = 0; i < D; ++i) col[i] = U[i * D + c];
    for (std::size_t base = 0; base < D; ++base) {
      if ((base & tmask) != 0 || (base & cmask) != cmask) continue;  // one visit per group, controls on
      for (std::size_t m = 0; m < K; ++m) in[m] = col[base | member[m]];
      for (std::size_t r = 0; r < K; ++r) {
        cdouble s = 0;
        for (std::size_t m = 0; m < K; ++m) s += op.matrix[r * K + m] * in[m];
        col[base | member[r]] = s;
      }
    }
    for (std::size_t i = 0; i < D; ++i) U[i * D + c] = col[i];
  }
}

Op fuse_block(const std::vector<const Op*>& block, const std::vector<int>& qs) {
  const std::size_t D = std::size_t{1} << qs.size();
  Op out;
  out.name = "fused";
  out.targets = qs;
  out.matrix.assign(D * D, cdouble{0});
  for (std::size_t i = 0; i < D; ++i) out.matrix[i * D + i] = 1;
  for (const Op* op : block) apply_to_block(out.matrix, qs, *op);
  return out;
}

}  // namespace

namespace {

Op emit_block(const std::vector<const Op*>& ops, const std::vector<int>& qs) {
  return ops.size() == 1 ? *ops[0] : fuse_block(ops, qs);
}

// Commutation-aware fusion (M3.4). Keeps several open blocks on pairwise-disjoint qubit sets; a gate
// joins (and if needed merges) the open blocks it touches as long as the union stays <= K qubits.
// Gates on disjoint qubits commute, so letting unrelated blocks stay open while others grow is exact.
// Invariant: for every qubit, the blocks that touch it are emitted in input order (a block is only
// closed when a later gate on one of its qubits can't join it, and that gate's block is emitted after).
Circuit fuse_commuting(const Circuit& c, int K) {
  struct Block {
    std::vector<int> qs;          // sorted
    std::vector<std::size_t> ops; // indices into c.ops, ascending
    bool open = false;
  };
  Circuit out;
  out.num_qubits = c.num_qubits;
  std::vector<Block> blocks;
  std::vector<int> owner(static_cast<std::size_t>(c.num_qubits), -1);  // qubit -> open block, or -1

  auto close = [&](int b) {
    Block& blk = blocks[static_cast<std::size_t>(b)];
    std::vector<const Op*> ops;
    for (std::size_t i : blk.ops) ops.push_back(&c.ops[i]);
    out.ops.push_back(emit_block(ops, blk.qs));
    for (int q : blk.qs) owner[static_cast<std::size_t>(q)] = -1;
    blk.open = false;
  };

  for (std::size_t i = 0; i < c.ops.size(); ++i) {
    const std::vector<int> oq = qubits_of(c.ops[i]);
    std::vector<int> touched;
    for (int q : oq) {
      const int b = owner[static_cast<std::size_t>(q)];
      if (b >= 0 && std::find(touched.begin(), touched.end(), b) == touched.end()) touched.push_back(b);
    }

    if (static_cast<int>(oq.size()) > K) {  // too wide to fuse: flush what it depends on, pass it through
      for (int b : touched) close(b);
      out.ops.push_back(c.ops[i]);
      continue;
    }

    // Keep the biggest touched blocks that still fit together with this gate; close the rest (they
    // share a qubit with the gate, so they have to come before it anyway).
    std::sort(touched.begin(), touched.end(), [&](int a, int b) {
      return blocks[static_cast<std::size_t>(a)].qs.size() > blocks[static_cast<std::size_t>(b)].qs.size();
    });
    std::vector<int> keep, qs = oq;
    for (int b : touched) {
      std::vector<int> m = merged(qs, blocks[static_cast<std::size_t>(b)].qs);
      if (static_cast<int>(m.size()) <= K) {
        qs = std::move(m);
        keep.push_back(b);
      } else {
        close(b);
      }
    }

    Block nb;
    nb.qs = qs;
    nb.open = true;
    for (int b : keep) {
      Block& old = blocks[static_cast<std::size_t>(b)];
      nb.ops.insert(nb.ops.end(), old.ops.begin(), old.ops.end());
      old.open = false;
    }
    std::sort(nb.ops.begin(), nb.ops.end());  // kept blocks are disjoint: any order is valid; keep input order
    nb.ops.push_back(i);
    blocks.push_back(std::move(nb));
    for (int q : qs) owner[static_cast<std::size_t>(q)] = static_cast<int>(blocks.size() - 1);
  }

  // Remaining open blocks are pairwise disjoint; emit them in order of their first gate.
  std::vector<int> rest;
  for (std::size_t b = 0; b < blocks.size(); ++b)
    if (blocks[b].open) rest.push_back(static_cast<int>(b));
  std::sort(rest.begin(), rest.end(), [&](int a, int b) {
    return blocks[static_cast<std::size_t>(a)].ops.front() < blocks[static_cast<std::size_t>(b)].ops.front();
  });
  for (int b : rest) close(b);
  return out;
}

}  // namespace

double pass_cost(const Circuit& c) {
  double cost = 0;
  for (const Op& op : c.ops) cost += 1.0 / static_cast<double>(std::uint64_t{1} << op.controls.size());
  return cost;
}

Circuit fuse_circuit(const Circuit& c, int max_qubits, FusionStrategy strategy) {
  if (max_qubits < 1 || max_qubits > 6)
    throw std::invalid_argument("fuse_circuit: max_qubits must be 1..6, got " + std::to_string(max_qubits));
  if (strategy == FusionStrategy::Best) {
    Circuit a = fuse_circuit(c, max_qubits, FusionStrategy::Commuting);
    Circuit b = fuse_circuit(c, max_qubits, FusionStrategy::InOrder);
    return pass_cost(b) < pass_cost(a) ? b : a;
  }
  if (strategy == FusionStrategy::Commuting) return fuse_commuting(c, max_qubits);

  // In-order greedy (M3.2): one open block; close it as soon as the next gate doesn't fit.
  Circuit out;
  out.num_qubits = c.num_qubits;
  std::vector<const Op*> block;
  std::vector<int> qs;

  auto flush = [&] {
    if (block.size() == 1) out.ops.push_back(*block[0]);
    else if (block.size() > 1) out.ops.push_back(fuse_block(block, qs));
    block.clear();
    qs.clear();
  };

  for (const Op& op : c.ops) {
    const std::vector<int> oq = qubits_of(op);
    if (static_cast<int>(oq.size()) > max_qubits) {  // too wide to fuse with anything
      flush();
      out.ops.push_back(op);
      continue;
    }
    std::vector<int> m = merged(qs, oq);
    if (static_cast<int>(m.size()) > max_qubits) {
      flush();
      m = oq;
    }
    block.push_back(&op);
    qs = std::move(m);
  }
  flush();
  return out;
}

}  // namespace qsim
