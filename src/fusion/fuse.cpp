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

Circuit fuse_circuit(const Circuit& c, int max_qubits) {
  if (max_qubits < 1 || max_qubits > 6)
    throw std::invalid_argument("fuse_circuit: max_qubits must be 1..6, got " + std::to_string(max_qubits));

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
