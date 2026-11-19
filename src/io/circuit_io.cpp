// Parser for the *.qc circuit text format (see include/qsim/circuit.hpp for the format).

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "qsim/circuit.hpp"
#include "qsim/gates.hpp"

namespace qsim {
namespace {

struct GateSpec {
  int params;
  int targets;
  int controls;
};

// Controlled gates are spelled with one leading 'c' per control: "ccx" = X with 2 controls.
const std::unordered_map<std::string, GateSpec>& gate_table() {
  static const std::unordered_map<std::string, GateSpec> table = {
      {"h", {0, 1, 0}},    {"x", {0, 1, 0}},    {"y", {0, 1, 0}},   {"z", {0, 1, 0}},
      {"s", {0, 1, 0}},    {"sdg", {0, 1, 0}},  {"t", {0, 1, 0}},   {"tdg", {0, 1, 0}},
      {"rx", {1, 1, 0}},   {"ry", {1, 1, 0}},   {"rz", {1, 1, 0}},  {"u3", {3, 1, 0}},
      {"p", {1, 1, 0}},    {"cp", {1, 1, 1}},
      {"swap", {0, 2, 0}}, {"cx", {0, 1, 1}},   {"cy", {0, 1, 1}},  {"cz", {0, 1, 1}},
      {"ch", {0, 1, 1}},   {"crz", {1, 1, 1}},  {"ccx", {0, 1, 2}}, {"cswap", {0, 2, 1}},
  };
  return table;
}

Matrix base_matrix(const std::string& base, const std::vector<double>& p) {
  using namespace gates;
  if (base == "h") return H();
  if (base == "x") return X();
  if (base == "y") return Y();
  if (base == "z") return Z();
  if (base == "s") return S();
  if (base == "sdg") return Sdg();
  if (base == "t") return T();
  if (base == "tdg") return Tdg();
  if (base == "rx") return RX(p[0]);
  if (base == "ry") return RY(p[0]);
  if (base == "rz") return RZ(p[0]);
  if (base == "u3") return U3(p[0], p[1], p[2]);
  if (base == "p") return U3(0, 0, p[0]);  // phase gate diag(1, e^{i lambda}), as in Qiskit
  if (base == "swap") return SWAP();
  throw std::logic_error("parser: no matrix for gate '" + base + "'");
}

[[noreturn]] void fail(int line, const std::string& msg) {
  throw std::runtime_error("line " + std::to_string(line) + ": " + msg);
}

int parse_int(const std::string& tok, int line) {
  std::size_t used = 0;
  int v = 0;
  try {
    v = std::stoi(tok, &used);
  } catch (const std::exception&) {
    fail(line, "expected an integer, got '" + tok + "'");
  }
  if (used != tok.size()) fail(line, "expected an integer, got '" + tok + "'");
  return v;
}

double parse_double(const std::string& tok, int line) {
  std::size_t used = 0;
  double v = 0;
  try {
    v = std::stod(tok, &used);
  } catch (const std::exception&) {
    fail(line, "expected a number, got '" + tok + "'");
  }
  if (used != tok.size()) fail(line, "expected a number, got '" + tok + "'");
  return v;
}

void check_distinct(const Op& op, int line) {
  std::vector<int> all = op.targets;
  all.insert(all.end(), op.controls.begin(), op.controls.end());
  std::sort(all.begin(), all.end());
  if (auto it = std::adjacent_find(all.begin(), all.end()); it != all.end())
    fail(line, "qubit " + std::to_string(*it) + " used twice in one op");
}

}  // namespace

Circuit parse_circuit(std::string_view text) {
  Circuit c;
  bool have_header = false;
  std::istringstream in{std::string(text)};
  std::string raw;
  int line = 0;

  while (std::getline(in, raw)) {
    ++line;
    if (auto hash = raw.find('#'); hash != std::string::npos) raw.resize(hash);
    std::istringstream ls(raw);
    std::vector<std::string> tok;
    for (std::string t; ls >> t;) tok.push_back(t);
    if (tok.empty()) continue;

    std::string name = tok[0];
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

    if (!have_header) {
      if (name != "qubits" || tok.size() != 2) fail(line, "the first statement must be 'qubits N'");
      c.num_qubits = parse_int(tok[1], line);
      if (c.num_qubits < 1 || c.num_qubits > 40) fail(line, "qubits must be 1..40");
      have_header = true;
      continue;
    }
    if (name == "qubits") fail(line, "'qubits' given twice");

    Op op;
    op.name = name;
    std::size_t pos = 1;
    auto take_qubit = [&]() {
      if (pos >= tok.size()) fail(line, "missing qubit index");
      const int q = parse_int(tok[pos++], line);
      if (q < 0 || q >= c.num_qubits)
        fail(line, "qubit " + std::to_string(q) + " out of range (circuit has " +
                       std::to_string(c.num_qubits) + " qubits)");
      return q;
    };

    if (name == "mat") {
      if (tok.size() < 2) fail(line, "mat needs an arity: mat <k> <qubits...> <entries...>");
      const int k = parse_int(tok[pos++], line);
      if (k < 1 || k > 6) fail(line, "mat arity must be 1..6");
      for (int i = 0; i < k; ++i) op.targets.push_back(take_qubit());
      check_distinct(op, line);
      const std::size_t dim = std::size_t{1} << k;
      const std::size_t want = 2 * dim * dim;
      if (tok.size() - pos != want)
        fail(line, "mat " + std::to_string(k) + " needs " + std::to_string(dim * dim) +
                       " complex entries (" + std::to_string(want) + " numbers), got " +
                       std::to_string(tok.size() - pos));
      op.matrix.resize(dim * dim);
      for (auto& m : op.matrix) {
        const double re = parse_double(tok[pos++], line);
        const double im = parse_double(tok[pos++], line);
        m = {re, im};
      }
    } else {
      const auto it = gate_table().find(name);
      if (it == gate_table().end()) fail(line, "unknown gate '" + name + "'");
      const GateSpec& g = it->second;
      const std::size_t need = 1 + std::size_t(g.params + g.controls + g.targets);
      if (tok.size() != need)
        fail(line, name + " takes " + std::to_string(g.params) + " parameter(s) and " +
                       std::to_string(g.controls + g.targets) + " qubit(s)");
      std::vector<double> params;
      for (int i = 0; i < g.params; ++i) params.push_back(parse_double(tok[pos++], line));
      for (int i = 0; i < g.controls; ++i) op.controls.push_back(take_qubit());
      for (int i = 0; i < g.targets; ++i) op.targets.push_back(take_qubit());
      check_distinct(op, line);

      op.matrix = base_matrix(name.substr(static_cast<std::size_t>(g.controls)), params);
      const std::size_t dim = std::size_t{1} << g.targets;
      if (op.matrix.size() != dim * dim)
        throw std::logic_error("gate '" + name + "': matrix has " + std::to_string(op.matrix.size()) +
                               " entries, expected " + std::to_string(dim * dim));
    }
    c.ops.push_back(std::move(op));
  }

  if (!have_header) throw std::runtime_error("empty circuit: missing 'qubits N'");
  return c;
}

Circuit load_circuit(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  try {
    return parse_circuit(ss.str());
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(path + ": " + e.what());
  }
}

}  // namespace qsim
