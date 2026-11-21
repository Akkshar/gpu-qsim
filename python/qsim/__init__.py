"""qsim: a hand-written CUDA state-vector quantum circuit simulator.

    import qsim

    c = qsim.Circuit(3).h(0).cx(0, 1).rz(0.3, 2)
    psi = qsim.statevector(c)                        # NumPy complex array, Qiskit qubit order
    counts = qsim.sample_counts(c, shots=1000)       # {"011": 512, "000": 488, ...}

    c = qsim.Circuit.from_qasm(open("bell.qasm").read())   # OpenQASM 2.0 works too

Amplitude index bit i is qubit i (little-endian, like Qiskit); count bitstrings print qubit n-1
first, also like Qiskit. Defaults: GPU when the build has CUDA, single precision, gate fusion with
blocks of up to 5 qubits ("best" strategy). The low-level module is qsim._qsim.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from . import _qsim
from ._qsim import CompiledCircuit, fuse, has_cuda, load, parse_qasm, parse_qc

__all__ = ["Circuit", "CompiledCircuit", "statevector", "sample_counts", "simulator", "fuse", "load",
           "parse_qc", "parse_qasm", "has_cuda"]


def _num(x: float) -> str:
    return repr(float(x))


class Circuit:
    """Chainable circuit builder. Every gate method returns the circuit, so calls can be chained."""

    def __init__(self, num_qubits: int):
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self.num_qubits = num_qubits
        self._lines: list[str] = []
        self._qasm: str | None = None

    @classmethod
    def from_qasm(cls, text: str) -> "Circuit":
        compiled = parse_qasm(text)  # validate now, so errors point at the QASM
        c = cls(compiled.num_qubits)
        c._qasm = text
        return c

    def _add(self, line: str) -> "Circuit":
        if self._qasm is not None:
            raise ValueError("this circuit came from QASM; build a new Circuit to add gates")
        self._lines.append(line)
        return self

    # 1-qubit gates
    def h(self, q): return self._add(f"h {q}")
    def x(self, q): return self._add(f"x {q}")
    def y(self, q): return self._add(f"y {q}")
    def z(self, q): return self._add(f"z {q}")
    def s(self, q): return self._add(f"s {q}")
    def sdg(self, q): return self._add(f"sdg {q}")
    def t(self, q): return self._add(f"t {q}")
    def tdg(self, q): return self._add(f"tdg {q}")
    def rx(self, theta, q): return self._add(f"rx {_num(theta)} {q}")
    def ry(self, theta, q): return self._add(f"ry {_num(theta)} {q}")
    def rz(self, theta, q): return self._add(f"rz {_num(theta)} {q}")
    def p(self, lam, q): return self._add(f"p {_num(lam)} {q}")
    def u3(self, theta, phi, lam, q): return self._add(f"u3 {_num(theta)} {_num(phi)} {_num(lam)} {q}")

    # controlled and multi-qubit gates (controls first, like Qiskit)
    def cx(self, c, t): return self._add(f"cx {c} {t}")
    def cy(self, c, t): return self._add(f"cy {c} {t}")
    def cz(self, c, t): return self._add(f"cz {c} {t}")
    def ch(self, c, t): return self._add(f"ch {c} {t}")
    def crx(self, theta, c, t): return self._add(f"crx {_num(theta)} {c} {t}")
    def cry(self, theta, c, t): return self._add(f"cry {_num(theta)} {c} {t}")
    def crz(self, theta, c, t): return self._add(f"crz {_num(theta)} {c} {t}")
    def cp(self, lam, c, t): return self._add(f"cp {_num(lam)} {c} {t}")
    def swap(self, a, b): return self._add(f"swap {a} {b}")
    def ccx(self, c1, c2, t): return self._add(f"ccx {c1} {c2} {t}")
    def cswap(self, c, a, b): return self._add(f"cswap {c} {a} {b}")

    def unitary(self, matrix, qubits: Iterable[int]) -> "Circuit":
        """Arbitrary 2^k x 2^k unitary on k <= 6 qubits; qubits[0] is the matrix's lowest index bit."""
        qubits = list(qubits)
        m = np.asarray(matrix, dtype=complex)
        if m.shape != (2 ** len(qubits),) * 2:
            raise ValueError(f"matrix must be {2 ** len(qubits)}x{2 ** len(qubits)} for {len(qubits)} qubits")
        entries = " ".join(f"{_num(z.real)} {_num(z.imag)}" for z in m.flatten())
        return self._add(f"mat {len(qubits)} {' '.join(map(str, qubits))} {entries}")

    def to_qc(self) -> str:
        if self._qasm is not None:
            raise ValueError("this circuit came from QASM")
        return "\n".join([f"qubits {self.num_qubits}", *self._lines]) + "\n"

    def compile(self) -> CompiledCircuit:
        return parse_qasm(self._qasm) if self._qasm is not None else parse_qc(self.to_qc())

    def __len__(self) -> int:
        return len(self.compile()) if self._qasm is not None else len(self._lines)

    def __repr__(self) -> str:
        return f"<qsim.Circuit {self.num_qubits} qubits, {len(self)} gates>"


def _compiled(circuit) -> CompiledCircuit:
    return circuit.compile() if isinstance(circuit, Circuit) else circuit


def simulator(num_qubits: int, device: str | None = None, precision: str = "f32"):
    """A low-level state-vector object (CPU or GPU, complex64 or complex128)."""
    device = device or ("gpu" if has_cuda else "cpu")
    if precision not in ("f32", "f64"):
        raise ValueError("precision must be 'f32' or 'f64'")
    if device == "gpu":
        if not has_cuda:
            raise RuntimeError("this qsim build has no CUDA backend")
        cls = _qsim.CudaStateVectorF32 if precision == "f32" else _qsim.CudaStateVectorF64
    elif device == "cpu":
        cls = _qsim.CpuStateVectorF32 if precision == "f32" else _qsim.CpuStateVectorF64
    else:
        raise ValueError("device must be 'gpu' or 'cpu'")
    return cls(num_qubits)


def _run(circuit, device, precision, fuse_qubits):
    c = _compiled(circuit)
    if fuse_qubits:
        c = fuse(c, fuse_qubits)
    sv = simulator(c.num_qubits, device, precision)
    sv.run(c)
    return sv


def statevector(circuit, device: str | None = None, precision: str = "f32", fuse_qubits: int = 5) -> np.ndarray:
    """Final state of a circuit (Circuit or CompiledCircuit) as a NumPy array."""
    return _run(circuit, device, precision, fuse_qubits).amplitudes()


def sample_counts(circuit, shots: int = 1024, seed: int = 1, device: str | None = None, precision: str = "f32",
                  fuse_qubits: int = 5) -> dict[str, int]:
    """Measure every qubit `shots` times. Bitstrings list qubit n-1 first (Qiskit order)."""
    sv = _run(circuit, device, precision, fuse_qubits)
    n = sv.num_qubits
    return dict(Counter(format(int(x), f"0{n}b") for x in sv.sample(shots, seed)))
