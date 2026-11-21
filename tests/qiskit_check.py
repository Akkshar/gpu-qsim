"""Cross-check qsim against Qiskit on random circuits, and emit random circuits for benchmarking.

  check: random circuits -> Qiskit Statevector (ground truth) vs `qsim_cli --dump`, amplitude by amplitude
      python tests/qiskit_check.py check [--circuits 200] [--max-qubits 12] [--max-depth 40]
                                         [--precision f64|f32] [--backend cpu|gpu] [--fuse K] [--seed 1]
                                         [--exe path/to/qsim_cli]
  emit: write one random circuit to a .qc file (no Qiskit simulation, so any size works)
      python tests/qiskit_check.py emit --qubits 24 --depth 200 --seed 7 --out bench/circuits/rand24.qc

Failing circuits are saved to tests/failures/ so you can rerun them with qsim_cli directly.
Comparison is exact (global phase included): conventions in include/qsim/gates.hpp match Qiskit's.
"""

from __future__ import annotations

import argparse
import math
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "build" / "Release" / ("qsim_cli.exe" if sys.platform == "win32" else "qsim_cli")
FAIL_DIR = ROOT / "tests" / "failures"

FIXED_1Q = ["h", "x", "y", "z", "s", "sdg", "t", "tdg"]
ROT_1Q = ["rx", "ry", "rz", "p"]
CTRL_1Q = ["cx", "cy", "cz", "ch"]


def num(x: float) -> str:
    return repr(float(x))  # shortest string that round-trips exactly


def random_ops(n: int, depth: int, rng: random.Random, with_qiskit: bool, allow_mat: bool = True):
    """Yield (qc_or_None, text lines). Builds the Qiskit circuit in step when with_qiskit is set."""
    qc = None
    if with_qiskit:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(n)
    lines = [f"qubits {n}"]

    kinds = ["fixed", "rot", "u3"]
    if n >= 2:
        kinds += ["ctrl", "crz", "swap"] + (["mat"] if allow_mat else [])
    if n >= 3:
        kinds += ["ccx", "cswap"]

    for _ in range(depth):
        kind = rng.choice(kinds)
        if kind == "fixed":
            g, q = rng.choice(FIXED_1Q), rng.randrange(n)
            lines.append(f"{g} {q}")
            if qc is not None:
                getattr(qc, g)(q)
        elif kind == "rot":
            g, q, t = rng.choice(ROT_1Q), rng.randrange(n), rng.uniform(-2 * math.pi, 2 * math.pi)
            lines.append(f"{g} {num(t)} {q}")
            if qc is not None:
                getattr(qc, g)(t, q)
        elif kind == "u3":
            q = rng.randrange(n)
            t, p, l = (rng.uniform(-2 * math.pi, 2 * math.pi) for _ in range(3))
            lines.append(f"u3 {num(t)} {num(p)} {num(l)} {q}")
            if qc is not None:
                qc.u(t, p, l, q)  # Qiskit's U gate == u3
        elif kind == "ctrl":
            g = rng.choice(CTRL_1Q)
            c, t = rng.sample(range(n), 2)
            lines.append(f"{g} {c} {t}")
            if qc is not None:
                getattr(qc, g)(c, t)
        elif kind == "crz":
            g = rng.choice(["crz", "cp"])
            c, t = rng.sample(range(n), 2)
            a = rng.uniform(-2 * math.pi, 2 * math.pi)
            lines.append(f"{g} {num(a)} {c} {t}")
            if qc is not None:
                getattr(qc, g)(a, c, t)
        elif kind == "swap":
            a, b = rng.sample(range(n), 2)
            lines.append(f"swap {a} {b}")
            if qc is not None:
                qc.swap(a, b)
        elif kind == "ccx":
            a, b, t = rng.sample(range(n), 3)
            lines.append(f"ccx {a} {b} {t}")
            if qc is not None:
                qc.ccx(a, b, t)
        elif kind == "cswap":
            c, a, b = rng.sample(range(n), 3)
            lines.append(f"cswap {c} {a} {b}")
            if qc is not None:
                qc.cswap(c, a, b)
        elif kind == "mat":
            k = rng.randint(1, min(3, n))
            qubits = rng.sample(range(n), k)
            u = random_unitary_matrix(2**k, rng)
            entries = " ".join(f"{num(z.real)} {num(z.imag)}" for z in u.flatten())  # row-major
            lines.append(f"mat {k} {' '.join(map(str, qubits))} {entries}")
            if qc is not None:
                qc.unitary(u, qubits)  # Qiskit: qubits[0] is the low bit of the matrix index, same as ours
    return qc, lines


def random_unitary_matrix(d: int, rng: random.Random) -> np.ndarray:
    """Haar-random unitary via QR of a complex Gaussian matrix (numpy only, seeded from rng)."""
    g = np.random.default_rng(rng.randrange(2**63))
    z = (g.standard_normal((d, d)) + 1j * g.standard_normal((d, d))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def cmd_check(args) -> int:
    from qiskit.quantum_info import Statevector

    exe = Path(args.exe)
    if not exe.exists():
        print(f"qsim_cli not found at {exe}; build first (cmake --build build --config Release)")
        return 2
    tol = 1e-9 if args.precision == "f64" else 2e-4
    rng = random.Random(args.seed)
    FAIL_DIR.mkdir(parents=True, exist_ok=True)

    failures, worst = [], 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(args.circuits):
            n = rng.randint(1, args.max_qubits)
            depth = rng.randint(args.min_depth, args.max_depth)
            qc, lines = random_ops(n, depth, rng, with_qiskit=True, allow_mat=not args.qasm)
            dump = Path(tmp) / f"c{i}.bin"
            if args.qasm:  # feed qsim Qiskit's own OpenQASM 2 export instead of the .qc text
                from qiskit import qasm2

                qc_path = Path(tmp) / f"c{i}.qasm"
                qc_path.write_text(qasm2.dumps(qc))
                lines = qc_path.read_text().splitlines()
            else:
                qc_path = Path(tmp) / f"c{i}.qc"
                qc_path.write_text("\n".join(lines) + "\n")

            cmd = [str(exe), str(qc_path), f"--{args.precision}", "--dump", str(dump)]
            if args.backend == "gpu":
                cmd.append("--gpu")
            if args.fuse:
                cmd += ["--fuse", str(args.fuse)]
            if args.backend == "gpu":
                cmd += ["--kernel", args.kernel]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if proc.returncode != 0:
                err = proc.stderr.strip()
                if "TODO(M1)" in err:
                    print(f"not implemented yet -> {err}")
                    return 1
                failures.append((i, n, depth, f"qsim_cli failed: {err}", lines))
                continue

            got = np.fromfile(dump, dtype=np.complex128)
            want = Statevector(qc).data
            if got.shape != want.shape:
                failures.append((i, n, depth, f"wrong size {got.size}, expected {want.size}", lines))
                continue
            diff = float(np.max(np.abs(got - want)))
            worst = max(worst, diff)
            if diff > tol:
                failures.append((i, n, depth, f"max |amp diff| = {diff:.3e} (tol {tol:g})", lines))

    for i, n, depth, why, lines in failures[:10]:
        path = FAIL_DIR / f"fail_{args.seed}_{i}.{'qasm' if args.qasm else 'qc'}"
        path.write_text("\n".join(lines) + "\n")
        print(f"FAIL circuit {i}: {n} qubits, {depth} ops: {why}\n     saved to {path}")
    passed = args.circuits - len(failures)
    fused = (" qasm" if args.qasm else "") + (f", fuse {args.fuse}" if args.fuse else "") + (f", kernel {args.kernel}" if args.backend == "gpu" else "")
    print(f"{passed}/{args.circuits} passed ({args.backend}, {args.precision}{fused}, seed {args.seed}); "
          f"worst max |amp diff| {worst:.2e}")
    return 0 if not failures else 1


def cmd_emit(args) -> int:
    _, lines = random_ops(args.qubits, args.depth, random.Random(args.seed), with_qiskit=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}: {args.qubits} qubits, {args.depth} ops")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="compare against Qiskit on random circuits")
    c.add_argument("--circuits", type=int, default=200)
    c.add_argument("--max-qubits", type=int, default=12)
    c.add_argument("--min-depth", type=int, default=1)
    c.add_argument("--max-depth", type=int, default=40)
    c.add_argument("--precision", choices=["f64", "f32"], default="f64")
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    c.add_argument("--fuse", type=int, default=0, help="pass --fuse K to qsim_cli (0 = off)")
    c.add_argument("--qasm", action="store_true", help="give qsim Qiskit's OpenQASM 2 export (no `mat` gates)")
    c.add_argument("--kernel", choices=["auto", "shuffle", "direct"], default="auto", help="GPU kernel for 2-5 target ops")
    c.add_argument("--exe", default=str(DEFAULT_EXE))

    e = sub.add_parser("emit", help="write a random circuit file")
    e.add_argument("--qubits", type=int, required=True)
    e.add_argument("--depth", type=int, required=True)
    e.add_argument("--seed", type=int, default=1)
    e.add_argument("--out", required=True)

    args = ap.parse_args()
    return cmd_check(args) if args.cmd == "check" else cmd_emit(args)


if __name__ == "__main__":
    sys.exit(main())
