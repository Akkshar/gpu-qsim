// nanobind bindings: the low-level `qsim._qsim` module. The friendly API (Circuit builder,
// statevector(), sample_counts()) is in python/qsim/__init__.py on top of this.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <complex>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "qsim/circuit.hpp"
#include "qsim/cpu_statevector.hpp"
#include "qsim/fusion.hpp"
#ifdef QSIM_HAS_CUDA
#include "qsim/cuda_statevector.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Hand a std::vector to NumPy without copying: the array owns the vector through a capsule.
template <class T>
nb::ndarray<nb::numpy, T, nb::ndim<1>> to_numpy(std::vector<T>&& v) {
  auto* heap = new std::vector<T>(std::move(v));
  nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  return nb::ndarray<nb::numpy, T, nb::ndim<1>>(heap->data(), {heap->size()}, owner);
}

template <class T>
std::vector<std::complex<T>> amplitudes(const qsim::CpuStateVector<T>& sv) {
  return {sv.amps().begin(), sv.amps().end()};
}
#ifdef QSIM_HAS_CUDA
template <class T>
std::vector<std::complex<T>> amplitudes(const qsim::CudaStateVector<T>& sv) {
  return sv.download();
}
#endif

template <class SV>
void bind_statevector(nb::module_& m, const char* name, const char* doc) {
  nb::class_<SV>(m, name, doc)
      .def(nb::init<int>(), "num_qubits"_a)
      .def_prop_ro("num_qubits", &SV::num_qubits)
      .def("reset", &SV::reset, "Back to |0...0>.")
      .def("run", &SV::run, "circuit"_a, nb::call_guard<nb::gil_scoped_release>(),
           "Apply every op of a compiled circuit (state is not reset first).")
      .def("amplitudes", [](const SV& s) { return to_numpy(amplitudes(s)); },
           "All 2^n amplitudes as a NumPy array; index bit i = qubit i (Qiskit order).")
      .def("probabilities", [](const SV& s) { return to_numpy(s.probabilities()); })
      .def("norm_squared", &SV::norm_squared)
      .def("sample", [](const SV& s, int shots, std::uint64_t seed) { return to_numpy(s.sample(shots, seed)); },
           "shots"_a, "seed"_a = 1, "Basis-state indices drawn from |amplitude|^2; the state is unchanged.")
      .def("measure",
           [](SV& s, int qubit, std::uint64_t seed) {
             std::mt19937_64 rng(seed);
             return s.measure(qubit, rng);
           },
           "qubit"_a, "seed"_a = 1, "Measure one qubit: returns 0 or 1, collapses and renormalises.");
}

}  // namespace

NB_MODULE(_qsim, m) {
  m.doc() = "qsim: hand-written CUDA state-vector quantum circuit simulator (low-level bindings)";

  nb::class_<qsim::Circuit>(m, "CompiledCircuit", "A parsed circuit: qubit count plus a list of ops.")
      .def_ro("num_qubits", &qsim::Circuit::num_qubits)
      .def("__len__", [](const qsim::Circuit& c) { return c.ops.size(); })
      .def("pass_cost", [](const qsim::Circuit& c) { return qsim::pass_cost(c); },
           "Estimated full passes over the state (1 per op, 1/2^c for an op with c controls).")
      .def("__repr__", [](const qsim::Circuit& c) {
        return "<CompiledCircuit " + std::to_string(c.num_qubits) + " qubits, " + std::to_string(c.ops.size()) +
               " ops>";
      });

  m.def("parse_qc", [](const std::string& text) { return qsim::parse_circuit(text); }, "text"_a,
        "Parse qsim's .qc text format.");
  m.def("parse_qasm", [](const std::string& text) { return qsim::parse_qasm(text); }, "text"_a,
        "Parse an OpenQASM 2.0 program (qelib1.inc gates; measure/barrier ignored).");
  m.def("load", &qsim::load_circuit, "path"_a, "Load a .qc or OpenQASM 2.0 file (detected by content).");
  m.def(
      "fuse",
      [](const qsim::Circuit& c, int max_qubits, const std::string& mode) {
        qsim::FusionStrategy s = qsim::FusionStrategy::Best;
        if (mode == "commute") s = qsim::FusionStrategy::Commuting;
        else if (mode == "inorder") s = qsim::FusionStrategy::InOrder;
        else if (mode != "best") throw std::invalid_argument("mode must be 'best', 'commute' or 'inorder'");
        return qsim::fuse_circuit(c, max_qubits, s);
      },
      "circuit"_a, "max_qubits"_a = 5, "mode"_a = "best",
      "Merge gates into dense blocks of at most max_qubits qubits (exact; fewer passes over the state).");

  bind_statevector<qsim::CpuStateVector<float>>(m, "CpuStateVectorF32", "CPU reference simulator, complex64.");
  bind_statevector<qsim::CpuStateVector<double>>(m, "CpuStateVectorF64", "CPU reference simulator, complex128.");
#ifdef QSIM_HAS_CUDA
  bind_statevector<qsim::CudaStateVector<float>>(m, "CudaStateVectorF32", "CUDA simulator, complex64.");
  bind_statevector<qsim::CudaStateVector<double>>(m, "CudaStateVectorF64", "CUDA simulator, complex128.");
  m.attr("has_cuda") = true;
#else
  m.attr("has_cuda") = false;
#endif
}
