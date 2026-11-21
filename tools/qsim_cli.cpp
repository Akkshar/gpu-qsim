// qsim_cli — run a circuit (.qc or OpenQASM 2.0) on the CPU reference or the GPU backend.
//
//   qsim_cli <circuit.qc|circuit.qasm> [--gpu] [--f32 | --f64] [--fuse K] [--fuse-mode M]
//            [--kernel auto|shuffle|direct] [--shots N] [--seed S] [--reps N] [--dump out.bin]
//            [--profile-ops]
//
// --shots samples N measurements of all qubits from the final state and prints the counts as
// bitstrings, qubit n-1 first (Qiskit's order). --dump writes the final state as raw little-endian
// complex128 (2^n values, real then imaginary), which tests/qiskit_check.py compares against Qiskit.
// Timing is the median over --reps runs; on the GPU it covers launching every op and waiting for the
// last one (state reset excluded).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "qsim/circuit.hpp"
#include "qsim/cpu_statevector.hpp"
#include "qsim/fusion.hpp"
#ifdef QSIM_HAS_CUDA
#include "qsim/cuda_statevector.hpp"
#endif

namespace {

constexpr double kGpuCeilingGBs = 358.0;  // measured on this laptop, on AC (BENCH.md, M0)

enum class KernelChoice { Auto, Shuffle, Direct };

struct Options {
  const char* dump = nullptr;
  bool gpu = false, prof = false;
  int reps = 1, shots = 0;
  std::uint64_t seed = 1;
  KernelChoice kernel = KernelChoice::Auto;
};

template <class T> void sync(qsim::CpuStateVector<T>&) {}
template <class T> void configure(qsim::CpuStateVector<T>&, KernelChoice) {}
template <class T>
std::vector<std::complex<T>> amplitudes(const qsim::CpuStateVector<T>& sv) {
  return {sv.amps().begin(), sv.amps().end()};
}

#ifdef QSIM_HAS_CUDA
template <class T> void sync(qsim::CudaStateVector<T>& sv) { sv.synchronize(); }
template <class T> void configure(qsim::CudaStateVector<T>& sv, KernelChoice k) {
  sv.set_multi_target_kernel(k == KernelChoice::Shuffle  ? qsim::MultiTargetKernel::Shuffle
                             : k == KernelChoice::Direct ? qsim::MultiTargetKernel::Direct
                                                         : qsim::MultiTargetKernel::Auto);
}
template <class T>
std::vector<std::complex<T>> amplitudes(const qsim::CudaStateVector<T>& sv) {
  return sv.download();
}
#endif

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
void print_counts(const SV& sv, int n, int shots, std::uint64_t seed) {
  std::map<std::uint64_t, int> counts;
  for (std::uint64_t x : sv.sample(shots, seed)) ++counts[x];
  std::vector<std::pair<std::uint64_t, int>> rows(counts.begin(), counts.end());
  std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
    return a.second != b.second ? a.second > b.second : a.first < b.first;
  });
  std::printf("\ncounts (%d shots, seed %llu, %zu distinct outcomes; bitstring = qubit %d ... qubit 0)\n", shots,
              static_cast<unsigned long long>(seed), rows.size(), n - 1);
  for (std::size_t i = 0; i < rows.size() && i < 16; ++i) {
    std::string bits(static_cast<std::size_t>(n), '0');
    for (int q = 0; q < n; ++q)
      if ((rows[i].first >> q) & 1u) bits[static_cast<std::size_t>(n - 1 - q)] = '1';
    std::printf("  %s  %6d  (%.4f)\n", bits.c_str(), rows[i].second, double(rows[i].second) / shots);
  }
  if (rows.size() > 16) std::printf("  ... %zu more\n", rows.size() - 16);
}

template <class SV>
void run(const qsim::Circuit& c, const Options& o, std::size_t input_ops) {
  using amp_t = typename SV::amp_t;
  SV sv(c.num_qubits);
  configure(sv, o.kernel);
  std::vector<double> ms;
  for (int r = 0; r < o.reps; ++r) {
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
              sizeof(amp_t) == 8 ? "f32" : "f64", o.gpu ? "gpu" : "cpu");
  std::printf("time   %.3f ms (median of %d, min %.3f, max %.3f)\n", med, o.reps, ms.front(), ms.back());
  if (!c.ops.empty()) {
    // Bytes each op must move: read + write of every amplitude it touches. An op with c controls
    // touches only the 1/2^c of the state where all controls are 1.
    double bytes = 0;
    for (const auto& op : c.ops)
      bytes += 2.0 * double(sv.size()) * sizeof(amp_t) / double(std::uint64_t{1} << op.controls.size());
    const double per_op = med / double(c.ops.size());
    const double gbs = bytes / (med * 1e-3) / 1e9;
    std::printf("per op %.4f ms   bandwidth %.1f GB/s (amplitudes touched, read+write)", per_op, gbs);
    if (o.gpu) std::printf("   (%.0f%% of the %.0f GB/s ceiling)", 100.0 * gbs / kGpuCeilingGBs, kGpuCeilingGBs);
    std::printf("\n");
    if (input_ops != c.ops.size() && input_ops > 0)
      std::printf("per input op %.4f ms (%zu ops before fusion)\n", med / double(input_ops), input_ops);
  }

  if (o.dump) {
    std::ofstream f(o.dump, std::ios::binary);
    for (const auto& a : amplitudes(sv)) {
      const double re = a.real(), im = a.imag();
      f.write(reinterpret_cast<const char*>(&re), sizeof re);
      f.write(reinterpret_cast<const char*>(&im), sizeof im);
    }
    if (!f) throw std::runtime_error(std::string("failed writing ") + o.dump);
  }
  if (o.shots > 0) print_counts(sv, c.num_qubits, o.shots, o.seed);
  if (o.prof) profile_ops(sv, c);
}

int usage() {
  std::fprintf(stderr,
               "usage: qsim_cli <circuit.qc|circuit.qasm> [--gpu] [--f32 | --f64] [--fuse K] [--fuse-mode M]\n"
               "                [--kernel auto|shuffle|direct] [--shots N] [--seed S] [--reps N]\n"
               "                [--dump out.bin] [--profile-ops]\n"
               "  --gpu        run on the CUDA backend (default: CPU reference)\n"
               "  --fuse K     merge gates into dense blocks of at most K qubits (1..6)\n"
               "  --fuse-mode  best (default) | commute | inorder\n"
               "  --kernel     GPU kernel for 2-5 target ops: warp-shuffle or direct (default auto)\n"
               "  --shots N    sample N measurements of all qubits and print the counts\n"
               "  --profile-ops  also time every op individually, grouped by shape\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = nullptr;
  bool f32 = false;
  int fuse = 0;
  qsim::FusionStrategy fuse_mode = qsim::FusionStrategy::Best;
  Options o;
  for (int i = 1; i < argc; ++i) {
    const bool has_value = i + 1 < argc;
    if (!std::strcmp(argv[i], "--kernel") && has_value) {
      const char* v = argv[++i];
      if (!std::strcmp(v, "auto")) o.kernel = KernelChoice::Auto;
      else if (!std::strcmp(v, "shuffle")) o.kernel = KernelChoice::Shuffle;
      else if (!std::strcmp(v, "direct")) o.kernel = KernelChoice::Direct;
      else return usage();
    } else if (!std::strcmp(argv[i], "--fuse-mode") && has_value) {
      const char* v = argv[++i];
      if (!std::strcmp(v, "best")) fuse_mode = qsim::FusionStrategy::Best;
      else if (!std::strcmp(v, "commute")) fuse_mode = qsim::FusionStrategy::Commuting;
      else if (!std::strcmp(v, "inorder")) fuse_mode = qsim::FusionStrategy::InOrder;
      else return usage();
    } else if (!std::strcmp(argv[i], "--f32")) f32 = true;
    else if (!std::strcmp(argv[i], "--f64")) f32 = false;
    else if (!std::strcmp(argv[i], "--gpu")) o.gpu = true;
    else if (!std::strcmp(argv[i], "--profile-ops")) o.prof = true;
    else if (!std::strcmp(argv[i], "--fuse") && has_value) fuse = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--shots") && has_value) o.shots = std::max(0, std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--seed") && has_value) o.seed = std::strtoull(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--reps") && has_value) o.reps = std::max(1, std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--dump") && has_value) o.dump = argv[++i];
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
      c = qsim::fuse_circuit(c, fuse, fuse_mode);
      const auto t1 = std::chrono::steady_clock::now();
      const char* mode = fuse_mode == qsim::FusionStrategy::Best        ? "best of both"
                         : fuse_mode == qsim::FusionStrategy::Commuting ? "commute"
                                                                         : "inorder";
      std::printf("fusion %zu ops -> %zu ops (blocks of <= %d qubits, %s, cost %.1f passes), %.2f ms on the host\n",
                  input_ops, c.ops.size(), fuse, mode, qsim::pass_cost(c),
                  std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    if (o.gpu) {
#ifdef QSIM_HAS_CUDA
      if (f32) run<qsim::CudaStateVector<float>>(c, o, input_ops);
      else run<qsim::CudaStateVector<double>>(c, o, input_ops);
#else
      throw std::runtime_error("this build has no CUDA backend (configure with QSIM_ENABLE_CUDA=ON)");
#endif
    } else {
      if (f32) run<qsim::CpuStateVector<float>>(c, o, input_ops);
      else run<qsim::CpuStateVector<double>>(c, o, input_ops);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
