// qsim_cli — run a circuit file on the CPU reference or the GPU backend.
//
//   qsim_cli <circuit.qc> [--gpu] [--f32 | --f64] [--fuse K] [--kernel auto|shuffle|direct]
//            [--reps N] [--dump out.bin] [--profile-ops]
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
#include "qsim/fusion.hpp"

namespace {

constexpr double kGpuCeilingGBs = 358.0;  // measured on this laptop, on AC (BENCH.md, M0)

template <class T> void sync(qsim::CpuStateVector<T>&) {}
template <class T> void sync(qsim::CudaStateVector<T>& sv) { sv.synchronize(); }
template <class T> void configure(qsim::CpuStateVector<T>&, qsim::MultiTargetKernel) {}
template <class T> void configure(qsim::CudaStateVector<T>& sv, qsim::MultiTargetKernel m) { sv.set_multi_target_kernel(m); }

template <class T>
std::vector<std::complex<T>> amplitudes(const qsim::CpuStateVector<T>& sv) {
  return {sv.amps().begin(), sv.amps().end()};
}
template <class T>
std::vector<std::complex<T>> amplitudes(const qsim::CudaStateVector<T>& sv) {
  return sv.download();
}

// --profile-ops: run the circuit once more, waiting after every op, and report where the time goes,
// grouped by op shape (name, targets, controls). Each op's time includes one host-device round trip
// (~10-20 us), negligible next to the multi-millisecond ops this is meant for.
template <class SV>
void profile_ops(SV& sv, const qsim::Circuit& c) {
  using amp_t = typename SV::amp_t;
  struct Row {
    std::string key;
    int count = 0;
    double ms = 0, bytes = 0;
  };
  std::vector<Row> rows;
  auto key_of = [](const qsim::Op& op) {
    std::string k = op.name + " t{";
    for (std::size_t i = 0; i < op.targets.size(); ++i) k += (i ? "," : "") + std::to_string(op.targets[i]);
    k += "}";
    if (!op.controls.empty()) {
      k += " c{";
      for (std::size_t i = 0; i < op.controls.size(); ++i) k += (i ? "," : "") + std::to_string(op.controls[i]);
      k += "}";
    }
    return k;
  };
  sv.reset();
  sync(sv);
  double total = 0;
  for (const auto& op : c.ops) {
    const auto t0 = std::chrono::steady_clock::now();
    sv.apply(op);
    sync(sv);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    total += ms;
    const std::string k = key_of(op);
    auto it = std::find_if(rows.begin(), rows.end(), [&](const Row& r) { return r.key == k; });
    if (it == rows.end()) it = rows.insert(rows.end(), Row{k});
    it->count += 1;
    it->ms += ms;
    it->bytes += 2.0 * double(sv.size()) * sizeof(amp_t) / double(std::uint64_t{1} << op.controls.size());
  }
  std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.ms > b.ms; });
  std::printf("\nper-op profile (one synchronised run, %.1f ms total), top %zu of %zu shapes:\n", total,
              std::min<std::size_t>(rows.size(), 20), rows.size());
  std::printf("  %-40s %6s %10s %10s %8s %7s\n", "op", "count", "total ms", "avg ms", "GB/s", "share");
  for (std::size_t i = 0; i < rows.size() && i < 20; ++i) {
    const Row& r = rows[i];
    std::printf("  %-40s %6d %10.1f %10.3f %8.1f %6.1f%%\n", r.key.c_str(), r.count, r.ms, r.ms / r.count,
                r.bytes / (r.ms * 1e-3) / 1e9, 100.0 * r.ms / total);
  }
}

template <class SV>
void run(const qsim::Circuit& c, int reps, const char* dump, bool gpu, std::size_t input_ops,
         qsim::MultiTargetKernel mtk, bool prof) {
  using amp_t = typename SV::amp_t;
  SV sv(c.num_qubits);
  configure(sv, mtk);
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
    if (input_ops != c.ops.size() && input_ops > 0)
      std::printf("per input op %.4f ms (%zu ops before fusion)\n", med / double(input_ops), input_ops);
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
  if (prof) profile_ops(sv, c);
}

int usage() {
  std::fprintf(stderr,
               "usage: qsim_cli <circuit.qc> [--gpu] [--f32 | --f64] [--fuse K] [--kernel auto|shuffle|direct]\n"
               "                [--reps N] [--dump out.bin] [--profile-ops]\n"
               "  --fuse K   merge consecutive gates into dense blocks of at most K qubits (1..6)\n"
               "  --kernel   GPU kernel for 2-5 target ops: warp-shuffle or direct (default auto)\n"
               "  --profile-ops  also time every op individually, grouped by shape\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = nullptr;
  const char* dump = nullptr;
  bool f32 = false, gpu = false, prof = false;
  int reps = 1, fuse = 0;
  qsim::MultiTargetKernel mtk = qsim::MultiTargetKernel::Auto;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--kernel") && i + 1 < argc) {
      const char* v = argv[++i];
      if (!std::strcmp(v, "auto")) mtk = qsim::MultiTargetKernel::Auto;
      else if (!std::strcmp(v, "shuffle")) mtk = qsim::MultiTargetKernel::Shuffle;
      else if (!std::strcmp(v, "direct")) mtk = qsim::MultiTargetKernel::Direct;
      else return usage();
      continue;
    }
    if (!std::strcmp(argv[i], "--f32")) f32 = true;
    else if (!std::strcmp(argv[i], "--f64")) f32 = false;
    else if (!std::strcmp(argv[i], "--gpu")) gpu = true;
    else if (!std::strcmp(argv[i], "--profile-ops")) prof = true;
    else if (!std::strcmp(argv[i], "--fuse") && i + 1 < argc) fuse = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::max(1, std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
    else if (argv[i][0] == '-') return usage();
    else if (!path) path = argv[i];
    else return usage();
  }
  if (!path) return usage();

  try {
    qsim::Circuit c = qsim::load_circuit(path);
    const std::size_t input_ops = c.ops.size();
    if (fuse > 0) {
      const auto t0 = std::chrono::steady_clock::now();
      c = qsim::fuse_circuit(c, fuse);
      const auto t1 = std::chrono::steady_clock::now();
      std::printf("fusion %zu ops -> %zu ops (blocks of <= %d qubits), %.2f ms on the host\n", input_ops,
                  c.ops.size(), fuse, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    if (gpu) {
      if (f32) run<qsim::CudaStateVector<float>>(c, reps, dump, true, input_ops, mtk, prof);
      else run<qsim::CudaStateVector<double>>(c, reps, dump, true, input_ops, mtk, prof);
    } else {
      if (f32) run<qsim::CpuStateVector<float>>(c, reps, dump, false, input_ops, mtk, prof);
      else run<qsim::CpuStateVector<double>>(c, reps, dump, false, input_ops, mtk, prof);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
