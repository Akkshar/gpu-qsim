// qsim_cli — run a circuit file on the CPU reference or the GPU backend.
//
//   qsim_cli <circuit.qc> [--gpu] [--f32 | --f64] [--reps N] [--dump out.bin]
//
// --dump writes the final state as raw little-endian complex128 (2^n values, real then imaginary),
// which tests/qiskit_check.py compares against Qiskit. Timing is the median over --reps runs; on the
// GPU it covers launching every op and waiting for the last one (state reset excluded).

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
#include "qsim/cuda_statevector.hpp"

namespace {

constexpr double kGpuCeilingGBs = 358.0;  // measured on this laptop, on AC (BENCH.md, M0)

template <class T> void sync(qsim::CpuStateVector<T>&) {}
template <class T> void sync(qsim::CudaStateVector<T>& sv) { sv.synchronize(); }

template <class T>
std::vector<std::complex<T>> amplitudes(const qsim::CpuStateVector<T>& sv) {
  return {sv.amps().begin(), sv.amps().end()};
}
template <class T>
std::vector<std::complex<T>> amplitudes(const qsim::CudaStateVector<T>& sv) {
  return sv.download();
}

template <class SV>
void run(const qsim::Circuit& c, int reps, const char* dump, bool gpu) {
  using amp_t = typename SV::amp_t;
  SV sv(c.num_qubits);
  std::vector<double> ms;
  for (int r = 0; r < reps; ++r) {
    sv.reset();
    sync(sv);
    const auto t0 = std::chrono::steady_clock::now();
    sv.run(c);
    sync(sv);
    const auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  const double med = ms[ms.size() / 2];

  std::printf("qubits %d   ops %zu   precision %s   backend %s\n", c.num_qubits, c.ops.size(),
              sizeof(amp_t) == 8 ? "f32" : "f64", gpu ? "gpu" : "cpu");
  std::printf("time   %.3f ms (median of %d, min %.3f, max %.3f)\n", med, reps, ms.front(), ms.back());
  if (!c.ops.empty()) {
    // Bytes each op must move: read + write of every amplitude it touches. An op with c controls
    // touches only the 1/2^c of the state where all controls are 1.
    double bytes = 0;
    for (const auto& op : c.ops)
      bytes += 2.0 * double(sv.size()) * sizeof(amp_t) / double(std::uint64_t{1} << op.controls.size());
    const double per_op = med / double(c.ops.size());
    const double gbs = bytes / (med * 1e-3) / 1e9;
    std::printf("per op %.4f ms   bandwidth %.1f GB/s (amplitudes touched, read+write)", per_op, gbs);
    if (gpu) std::printf("   (%.0f%% of the %.0f GB/s ceiling)", 100.0 * gbs / kGpuCeilingGBs, kGpuCeilingGBs);
    std::printf("\n");
  }

  if (dump) {
    std::ofstream f(dump, std::ios::binary);
    for (const auto& a : amplitudes(sv)) {
      const double re = a.real(), im = a.imag();
      f.write(reinterpret_cast<const char*>(&re), sizeof re);
      f.write(reinterpret_cast<const char*>(&im), sizeof im);
    }
    if (!f) throw std::runtime_error(std::string("failed writing ") + dump);
  }
}

int usage() {
  std::fprintf(stderr, "usage: qsim_cli <circuit.qc> [--gpu] [--f32 | --f64] [--reps N] [--dump out.bin]\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = nullptr;
  const char* dump = nullptr;
  bool f32 = false, gpu = false;
  int reps = 1;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--f32")) f32 = true;
    else if (!std::strcmp(argv[i], "--f64")) f32 = false;
    else if (!std::strcmp(argv[i], "--gpu")) gpu = true;
    else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
    else if (argv[i][0] == '-') return usage();
    else if (!path) path = argv[i];
    else return usage();
  }
  if (!path) return usage();

  try {
    const qsim::Circuit c = qsim::load_circuit(path);
    if (gpu) {
      if (f32) run<qsim::CudaStateVector<float>>(c, reps, dump, true);
      else run<qsim::CudaStateVector<double>>(c, reps, dump, true);
    } else {
      if (f32) run<qsim::CpuStateVector<float>>(c, reps, dump, false);
      else run<qsim::CpuStateVector<double>>(c, reps, dump, false);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
