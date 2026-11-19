// qsim_cli — run a circuit file on the CPU reference simulator.
//
//   qsim_cli <circuit.qc> [--f32 | --f64] [--reps N] [--dump out.bin]
//
// --dump writes the final state as raw little-endian complex128 (2^n values, real then imaginary),
// which tests/qiskit_check.py compares against Qiskit. Timing is the median over --reps runs.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "qsim/circuit.hpp"
#include "qsim/cpu_statevector.hpp"

namespace {

template <class T>
void run(const qsim::Circuit& c, int reps, const char* dump) {
  qsim::CpuStateVector<T> sv(c.num_qubits);
  std::vector<double> ms;
  for (int r = 0; r < reps; ++r) {
    sv.reset();
    const auto t0 = std::chrono::steady_clock::now();
    sv.run(c);
    const auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  const double med = ms[ms.size() / 2];

  std::printf("qubits %d   ops %zu   precision %s\n", c.num_qubits, c.ops.size(),
              sizeof(T) == 4 ? "f32" : "f64");
  std::printf("time   %.3f ms (median of %d, min %.3f, max %.3f)\n", med, reps, ms.front(), ms.back());
  if (!c.ops.empty()) {
    // Each op touches (at most) the whole state once: read + write.
    const double bytes_per_pass = 2.0 * double(sv.size()) * sizeof(typename qsim::CpuStateVector<T>::amp_t);
    const double per_op = med / double(c.ops.size());
    std::printf("per op %.4f ms   full-pass equivalent %.1f GB/s\n", per_op,
                bytes_per_pass / (per_op * 1e-3) / 1e9);
  }

  if (dump) {
    std::ofstream f(dump, std::ios::binary);
    for (const auto& a : sv.amps()) {
      const double re = a.real(), im = a.imag();
      f.write(reinterpret_cast<const char*>(&re), sizeof re);
      f.write(reinterpret_cast<const char*>(&im), sizeof im);
    }
    if (!f) throw std::runtime_error(std::string("failed writing ") + dump);
  }
}

int usage() {
  std::fprintf(stderr, "usage: qsim_cli <circuit.qc> [--f32 | --f64] [--reps N] [--dump out.bin]\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = nullptr;
  const char* dump = nullptr;
  bool f32 = false;
  int reps = 1;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--f32")) f32 = true;
    else if (!std::strcmp(argv[i], "--f64")) f32 = false;
    else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
    else if (argv[i][0] == '-') return usage();
    else if (!path) path = argv[i];
    else return usage();
  }
  if (!path) return usage();

  try {
    const qsim::Circuit c = qsim::load_circuit(path);
    if (f32) run<float>(c, reps, dump);
    else run<double>(c, reps, dump);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
