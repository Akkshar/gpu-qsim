// OpenQASM 2.0 subset -> Circuit. Each QASM statement is translated into .qc lines and the result goes
// through parse_circuit, so gate matrices and validation live in one place (circuit_io.cpp).
//
// Supported: OPENQASM 2.0; include "qelib1.inc"; qreg (several, concatenated in declaration order:
// first register = lowest qubits, like Qiskit); creg/barrier (ignored); measure (ignored: the
// simulator returns the pre-measurement state, sample it with --shots); gates
//   U u u3 u2 u1 p id u0 x y z h s sdg t tdg sx sxdg rx ry rz
//   CX cx cy cz ch crx cry crz cu1 cp cu3 swap ccx cswap rzz
// with parameter expressions (+ - * / ^, unary minus, pi, sin cos tan exp ln sqrt) and register
// broadcasting ("h q;", "cx a, b;" with equal-size registers).
// Rejected with a clear error: gate/opaque definitions, if(...), reset.

#include <cctype>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "qsim/circuit.hpp"

namespace qsim {
namespace {

[[noreturn]] void fail(int stmt, const std::string& msg) {
  throw std::runtime_error("qasm statement " + std::to_string(stmt) + ": " + msg);
}

std::string trim(std::string_view s) {
  std::size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return std::string(s.substr(a, b - a));
}

std::string num(double x) {
  char buf[40];
  std::snprintf(buf, sizeof buf, "%.17g", x);
  return buf;
}

// Recursive-descent evaluator for QASM 2 parameter expressions.
class Expr {
 public:
  Expr(const std::string& s, int stmt) : s_(s), stmt_(stmt) {}
  double eval() {
    const double v = sum();
    skip();
    if (i_ != s_.size()) fail(stmt_, "unexpected '" + s_.substr(i_) + "' in expression '" + s_ + "'");
    return v;
  }

 private:
  void skip() {
    while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
  }
  bool eat(char c) {
    skip();
    if (i_ < s_.size() && s_[i_] == c) {
      ++i_;
      return true;
    }
    return false;
  }
  double sum() {
    double v = product();
    for (;;) {
      if (eat('+')) v += product();
      else if (eat('-')) v -= product();
      else return v;
    }
  }
  double product() {
    double v = power();
    for (;;) {
      if (eat('*')) v *= power();
      else if (eat('/')) v /= power();
      else return v;
    }
  }
  double power() {
    const double base = unary();
    return eat('^') ? std::pow(base, power()) : base;
  }
  double unary() {
    if (eat('-')) return -unary();
    if (eat('+')) return unary();
    return primary();
  }
  double primary() {
    skip();
    if (eat('(')) {
      const double v = sum();
      if (!eat(')')) fail(stmt_, "missing ')' in expression '" + s_ + "'");
      return v;
    }
    if (i_ < s_.size() && std::isalpha(static_cast<unsigned char>(s_[i_]))) {
      const std::size_t start = i_;
      while (i_ < s_.size() && std::isalnum(static_cast<unsigned char>(s_[i_]))) ++i_;
      const std::string id = s_.substr(start, i_ - start);
      if (id == "pi") return std::numbers::pi;
      static const std::unordered_map<std::string, double (*)(double)> fns = {
          {"sin", [](double x) { return std::sin(x); }},   {"cos", [](double x) { return std::cos(x); }},
          {"tan", [](double x) { return std::tan(x); }},   {"exp", [](double x) { return std::exp(x); }},
          {"ln", [](double x) { return std::log(x); }},    {"sqrt", [](double x) { return std::sqrt(x); }}};
      const auto it = fns.find(id);
      if (it == fns.end()) fail(stmt_, "unknown identifier '" + id + "' in expression '" + s_ + "'");
      if (!eat('(')) fail(stmt_, "expected '(' after " + id);
      const double v = sum();
      if (!eat(')')) fail(stmt_, "missing ')' after " + id + "(...");
      return it->second(v);
    }
    std::size_t used = 0;
    double v = 0;
    try {
      v = std::stod(s_.substr(i_), &used);
    } catch (const std::exception&) {
      fail(stmt_, "bad number in expression '" + s_ + "'");
    }
    i_ += used;
    return v;
  }

  std::string s_;
  int stmt_;
  std::size_t i_ = 0;
};

// Split on top-level commas (not inside parentheses).
std::vector<std::string> split_args(const std::string& s) {
  std::vector<std::string> out;
  int depth = 0;
  std::string cur;
  for (char ch : s) {
    if (ch == '(') ++depth;
    if (ch == ')') --depth;
    if (ch == ',' && depth == 0) {
      out.push_back(trim(cur));
      cur.clear();
    } else {
      cur += ch;
    }
  }
  if (!trim(cur).empty()) out.push_back(trim(cur));
  return out;
}

struct Register {
  int offset, size;
};

// name -> (.qc name, params, qubits). Special cases (u2, sx, sxdg, rzz, id/u0) are handled in code.
struct Mapping {
  const char* qc;
  int params, qubits;
};
const std::unordered_map<std::string, Mapping>& mappings() {
  static const std::unordered_map<std::string, Mapping> m = {
      {"U", {"u3", 3, 1}},   {"u", {"u3", 3, 1}},     {"u3", {"u3", 3, 1}},   {"u1", {"p", 1, 1}},
      {"p", {"p", 1, 1}},    {"x", {"x", 0, 1}},      {"y", {"y", 0, 1}},     {"z", {"z", 0, 1}},
      {"h", {"h", 0, 1}},    {"s", {"s", 0, 1}},      {"sdg", {"sdg", 0, 1}}, {"t", {"t", 0, 1}},
      {"tdg", {"tdg", 0, 1}}, {"rx", {"rx", 1, 1}},   {"ry", {"ry", 1, 1}},   {"rz", {"rz", 1, 1}},
      {"CX", {"cx", 0, 2}},  {"cx", {"cx", 0, 2}},    {"cy", {"cy", 0, 2}},   {"cz", {"cz", 0, 2}},
      {"ch", {"ch", 0, 2}},  {"crx", {"crx", 1, 2}},  {"cry", {"cry", 1, 2}}, {"crz", {"crz", 1, 2}},
      {"cu1", {"cp", 1, 2}}, {"cp", {"cp", 1, 2}},    {"cu3", {"cu3", 3, 2}}, {"swap", {"swap", 0, 2}},
      {"ccx", {"ccx", 0, 3}}, {"cswap", {"cswap", 0, 3}},
      // translated specially below
      {"u2", {"", 2, 1}},    {"sx", {"", 0, 1}},      {"sxdg", {"", 0, 1}},   {"rzz", {"", 1, 2}},
      {"id", {"", 0, 1}},    {"u0", {"", 1, 1}},
  };
  return m;
}

}  // namespace

bool looks_like_qasm(std::string_view text) {
  std::size_t i = 0;
  for (;;) {
    while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
    if (text.substr(i, 2) == "//") {  // skip comment lines before the header
      while (i < text.size() && text[i] != '\n') ++i;
      continue;
    }
    return text.substr(i, 8) == "OPENQASM";
  }
}

Circuit parse_qasm(std::string_view text) {
  // Strip // comments, then split into ';'-terminated statements.
  std::string clean;
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '/' && i + 1 < text.size() && text[i + 1] == '/') {
      while (i < text.size() && text[i] != '\n') ++i;
      clean += '\n';
    } else {
      clean += text[i];
    }
  }

  std::unordered_map<std::string, Register> regs;
  int total = 0;
  std::vector<std::string> body;  // .qc lines after the header
  int stmt = 0;
  bool saw_header = false;

  std::size_t pos = 0;
  while (pos < clean.size()) {
    std::size_t end = clean.find(';', pos);
    const std::string raw = trim(clean.substr(pos, end == std::string::npos ? std::string::npos : end - pos));
    if (raw.find('{') != std::string::npos)
      fail(stmt + 1, "custom gate definitions ('gate ... { }') are not supported");
    if (end == std::string::npos) {
      if (!raw.empty()) fail(stmt + 1, "missing ';' after '" + raw + "'");
      break;
    }
    pos = end + 1;
    if (raw.empty()) continue;
    ++stmt;

    // keyword / gate name
    std::size_t k = 0;
    while (k < raw.size() && (std::isalnum(static_cast<unsigned char>(raw[k])) || raw[k] == '_')) ++k;
    const std::string word = raw.substr(0, k);
    const std::string rest = trim(raw.substr(k));

    if (word == "OPENQASM") {
      if (rest.empty() || rest[0] != '2') fail(stmt, "only OpenQASM 2.x is supported, got '" + rest + "'");
      saw_header = true;
      continue;
    }
    if (!saw_header) fail(stmt, "expected 'OPENQASM 2.0;' first");
    if (word == "include") {
      if (rest != "\"qelib1.inc\"") fail(stmt, "only include \"qelib1.inc\" is supported");
      continue;
    }
    if (word == "creg" || word == "barrier" || word == "measure") continue;
    if (word == "qreg") {
      const auto lb = rest.find('['), rb = rest.find(']');
      if (lb == std::string::npos || rb == std::string::npos) fail(stmt, "expected qreg name[size]");
      const std::string name = trim(rest.substr(0, lb));
      const int size = std::stoi(rest.substr(lb + 1, rb - lb - 1));
      if (size < 1) fail(stmt, "register size must be >= 1");
      if (regs.count(name)) fail(stmt, "register '" + name + "' declared twice");
      regs[name] = {total, size};
      total += size;
      continue;
    }
    if (word == "gate" || word == "opaque") fail(stmt, "custom gate definitions are not supported");
    if (word == "if") fail(stmt, "classically controlled gates (if) are not supported");
    if (word == "reset") fail(stmt, "reset is not supported");

    const auto mit = mappings().find(word);
    if (mit == mappings().end()) fail(stmt, "unsupported gate '" + word + "'");
    const Mapping& mp = mit->second;

    // parameters
    std::vector<double> params;
    std::string args = rest;
    if (!args.empty() && args[0] == '(') {
      int depth = 0;
      std::size_t close = 0;
      for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] == '(') ++depth;
        if (args[i] == ')' && --depth == 0) {
          close = i;
          break;
        }
      }
      if (close == 0) fail(stmt, "unbalanced parentheses in '" + raw + "'");
      for (const auto& e : split_args(args.substr(1, close - 1))) params.push_back(Expr(e, stmt).eval());
      args = trim(args.substr(close + 1));
    }
    if (static_cast<int>(params.size()) != mp.params)
      fail(stmt, word + " takes " + std::to_string(mp.params) + " parameter(s), got " + std::to_string(params.size()));

    // qubit arguments, with register broadcasting
    struct Arg {
      int offset, size;  // size 1 for an indexed qubit
      bool whole;
    };
    std::vector<Arg> qargs;
    for (const auto& a : split_args(args)) {
      const auto lb = a.find('[');
      const std::string name = trim(lb == std::string::npos ? a : a.substr(0, lb));
      const auto rit = regs.find(name);
      if (rit == regs.end()) fail(stmt, "unknown register '" + name + "'");
      if (lb == std::string::npos) {
        qargs.push_back({rit->second.offset, rit->second.size, true});
      } else {
        const int idx = std::stoi(a.substr(lb + 1));
        if (idx < 0 || idx >= rit->second.size)
          fail(stmt, "index " + std::to_string(idx) + " out of range for " + name + "[" +
                         std::to_string(rit->second.size) + "]");
        qargs.push_back({rit->second.offset + idx, 1, false});
      }
    }
    if (static_cast<int>(qargs.size()) != mp.qubits)
      fail(stmt, word + " takes " + std::to_string(mp.qubits) + " qubit(s), got " + std::to_string(qargs.size()));
    int reps = 1;
    for (const auto& q : qargs)
      if (q.whole) {
        if (reps != 1 && q.size != reps) fail(stmt, "broadcast registers must have the same size");
        reps = q.size;
      }

    for (int r = 0; r < reps; ++r) {
      std::vector<std::string> qs;
      for (const auto& q : qargs) qs.push_back(std::to_string(q.offset + (q.whole ? r : 0)));
      auto qlist = [&] {
        std::string s;
        for (const auto& q : qs) s += " " + q;
        return s;
      };
      std::string ps;
      for (double p : params) ps += " " + num(p);

      if (word == "id" || word == "u0") continue;  // identity
      if (word == "u2") {
        body.push_back("u3 " + num(std::numbers::pi / 2) + ps + qlist());
      } else if (word == "sx" || word == "sxdg") {
        const double s = word == "sx" ? 0.5 : -0.5;  // sx = [[1+i, 1-i], [1-i, 1+i]] / 2
        body.push_back("mat 1" + qlist() + " 0.5 " + num(s) + " 0.5 " + num(-s) + " 0.5 " + num(-s) + " 0.5 " +
                       num(s));
      } else if (word == "rzz") {  // exp(-i theta/2 Z(x)Z) = cx; rz(theta) on the target; cx
        body.push_back("cx " + qs[0] + " " + qs[1]);
        body.push_back("rz" + ps + " " + qs[1]);
        body.push_back("cx " + qs[0] + " " + qs[1]);
      } else {
        body.push_back(std::string(mp.qc) + ps + qlist());
      }
    }
  }

  if (!saw_header) throw std::runtime_error("qasm: missing 'OPENQASM 2.0;'");
  if (total == 0) throw std::runtime_error("qasm: no qreg declared");
  std::string qc = "qubits " + std::to_string(total) + "\n";
  for (const auto& l : body) qc += l + "\n";
  return parse_circuit(qc);
}

}  // namespace qsim
