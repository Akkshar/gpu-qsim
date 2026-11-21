"""Benchmark circuits and converters, shared by every benchmark script (Windows and WSL).

Generators return the .qc text format as a list of lines (see include/qsim/circuit.hpp):
  qft(n)            textbook QFT: h + controlled-phase ladder per qubit, then the reversal swaps
  layered(n, L)     random-circuit-sampling style: per layer a random u3 on every qubit, then cz on
                    neighbouring pairs (even pairs on even layers, odd pairs on odd layers)
  ghz(n)            h 0, then a cx chain
  qaoa(n, p)        QAOA for MaxCut on a ring: h on all, then p rounds of (cx-rz-cx on every ring
                    edge, rx on every qubit)
  random_mix(n)     tests/qiskit_check.py's mix of random gates on random qubits

Converters (no import-time dependency on Qiskit, so the cuStateVec venv can use this too):
  parse_qc(lines)   -> (n, [(name, params, controls, targets, matrix-or-None)])
  to_qiskit(lines)  -> qiskit.QuantumCircuit
  gate_matrix(name, params) -> numpy matrix of the *uncontrolled* gate, Qiskit conventions
"""

from __future__ import annotations

import cmath
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- generators

def qft(n: int) -> list[str]:
    lines = [f"qubits {n}"]
    for j in range(n):
        lines.append(f"h {j}")
        for k in range(j + 1, n):
            lines.append(f"cp {repr(math.pi / 2 ** (k - j))} {k} {j}")
    for j in range(n // 2):
        lines.append(f"swap {j} {n - 1 - j}")
    return lines


def layered(n: int, layers: int = 20, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    lines = [f"qubits {n}"]
    for layer in range(layers):
        for q in range(n):
            t, p, l = (rng.uniform(-math.pi, math.pi) for _ in range(3))
            lines.append(f"u3 {repr(t)} {repr(p)} {repr(l)} {q}")
        for q in range(layer % 2, n - 1, 2):
            lines.append(f"cz {q} {q + 1}")
    return lines


def ghz(n: int) -> list[str]:
    return [f"qubits {n}", "h 0"] + [f"cx {q} {q + 1}" for q in range(n - 1)]


def qaoa(n: int, p: int = 2, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    lines = [f"qubits {n}"] + [f"h {q}" for q in range(n)]
    for _ in range(p):
        gamma, beta = rng.uniform(0, math.pi), rng.uniform(0, math.pi)
        for q in range(n):
            a, b = q, (q + 1) % n
            lines += [f"cx {a} {b}", f"rz {repr(2 * gamma)} {b}", f"cx {a} {b}"]
        lines += [f"rx {repr(2 * beta)} {q}" for q in range(n)]
    return lines


def random_mix(n: int, depth: int = 200, seed: int = 7) -> list[str]:
    sys.path.insert(0, str(ROOT / "tests"))
    import qiskit_check  # noqa: E402

    _, lines = qiskit_check.random_ops(n, depth, random.Random(seed), with_qiskit=False)
    return lines


FAMILIES = {
    "qft": qft,
    "layered": layered,
    "ghz": ghz,
    "qaoa": qaoa,
    "random": random_mix,
}


# ---------------------------------------------------------------- parsing / matrices

# name -> (params, controls, targets); controlled names are "c" * controls + base name
SPEC = {
    "h": (0, 0, 1), "x": (0, 0, 1), "y": (0, 0, 1), "z": (0, 0, 1), "s": (0, 0, 1), "sdg": (0, 0, 1),
    "t": (0, 0, 1), "tdg": (0, 0, 1), "rx": (1, 0, 1), "ry": (1, 0, 1), "rz": (1, 0, 1), "u3": (3, 0, 1),
    "p": (1, 0, 1), "swap": (0, 0, 2), "cx": (0, 1, 1), "cy": (0, 1, 1), "cz": (0, 1, 1), "ch": (0, 1, 1),
    "crz": (1, 1, 1), "cp": (1, 1, 1), "ccx": (0, 2, 1), "cswap": (0, 1, 2),
}


def gate_matrix(name: str, params: list[float]) -> np.ndarray:
    """Uncontrolled gate matrix, row-major, Qiskit conventions (same as include/qsim/gates.hpp)."""
    if name == "h":
        return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    if name == "x":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if name == "y":
        return np.array([[0, -1j], [1j, 0]])
    if name == "z":
        return np.diag([1, -1]).astype(complex)
    if name == "s":
        return np.diag([1, 1j])
    if name == "sdg":
        return np.diag([1, -1j])
    if name == "t":
        return np.diag([1, cmath.exp(1j * math.pi / 4)])
    if name == "tdg":
        return np.diag([1, cmath.exp(-1j * math.pi / 4)])
    if name in ("rx", "ry", "rz"):
        c, s = math.cos(params[0] / 2), math.sin(params[0] / 2)
        if name == "rx":
            return np.array([[c, -1j * s], [-1j * s, c]])
        if name == "ry":
            return np.array([[c, -s], [s, c]], dtype=complex)
        return np.diag([cmath.exp(-1j * params[0] / 2), cmath.exp(1j * params[0] / 2)])
    if name == "u3":
        t, p, l = params
        c, s = math.cos(t / 2), math.sin(t / 2)
        return np.array([[c, -cmath.exp(1j * l) * s], [cmath.exp(1j * p) * s, cmath.exp(1j * (p + l)) * c]])
    if name == "p":
        return np.diag([1, cmath.exp(1j * params[0])])
    if name == "swap":
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    raise ValueError(name)


def parse_qc(lines: list[str]):
    n, ops = None, []
    for raw in lines:
        tok = raw.split("#")[0].split()
        if not tok:
            continue
        if tok[0] == "qubits":
            n = int(tok[1])
            continue
        name = tok[0]
        if name == "mat":
            k = int(tok[1])
            targets = [int(x) for x in tok[2:2 + k]]
            v = [float(x) for x in tok[2 + k:]]
            m = (np.array(v[0::2]) + 1j * np.array(v[1::2])).reshape(2**k, 2**k)
            ops.append(("mat", [], [], targets, m))
            continue
        npar, nc, nt = SPEC[name]
        params = [float(x) for x in tok[1:1 + npar]]
        qs = [int(x) for x in tok[1 + npar:]]
        ops.append((name, params, qs[:nc], qs[nc:], None))
    return n, ops


def op_matrix(op) -> np.ndarray:
    name, params, controls, _, m = op
    return m if m is not None else gate_matrix(name[len(controls):], params)


def to_qiskit(lines: list[str]):
    from qiskit import QuantumCircuit

    n, ops = parse_qc(lines)
    qc = QuantumCircuit(n)
    for name, params, controls, targets, m in ops:
        if name == "mat":
            qc.unitary(m, targets)
        elif name == "u3":
            qc.u(*params, *targets)
        else:
            getattr(qc, name)(*params, *controls, *targets)
    return qc


def gpu_preflight(label: str = "") -> dict:
    """Snapshot GPU load before/after a benchmark and warn if something else is using the GPU.

    A game running on the same GPU once made a whole A/B run 2-36x slow (BENCH.md, M3.4). Idle here is
    ~1-5% utilisation and ~1.1-1.4 GB used (desktop, browser, Steam client)."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,clocks.sm",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.strip()
        util, mem, temp, sm = (float(x) for x in out.split(","))
    except Exception as e:  # no nvidia-smi: don't block the benchmark
        print(f"[gpu {label}] nvidia-smi unavailable: {e}", file=sys.stderr)
        return {}
    snap = {"util_pct": util, "mem_mib": mem, "temp_c": temp, "sm_mhz": sm}
    print(f"[gpu {label}] {util:.0f}% busy, {mem:.0f} MiB used, {temp:.0f} C, SM {sm:.0f} MHz", file=sys.stderr)
    if util > 10 or mem > 2500:
        print(f"[gpu {label}] WARNING: something else is using the GPU; timings will be unreliable",
              file=sys.stderr)
    return snap


def write_qc(lines: list[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path
