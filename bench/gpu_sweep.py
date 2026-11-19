"""GPU bandwidth sweep: how close does each kind of op get to the memory ceiling?

For each target qubit t, runs 64 copies of one op on t and reports time per op and effective
bandwidth (read + write of every amplitude the op touches; a controlled op touches half the state)
as a % of the measured ceiling.

    python bench/gpu_sweep.py [--qubits 28] [--precision f32] [--reps 5] [--ops h,cx,mat2,mat3]
                              [--targets 0,1,2,...]

Prints a markdown table, ready for BENCH.md. Benchmark on AC power.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "Release" / ("qsim_cli.exe" if sys.platform == "win32" else "qsim_cli")
CEILING_GBS = 358.0  # BENCH.md, M0


def mat_text(k: int, qubits: list[int]) -> str:
    # Dense k-qubit matrix (all entries 1/sqrt(dim), Hadamard-like magnitude); contents don't affect speed.
    dim = 2**k
    v = repr(1 / dim**0.5)
    return f"mat {k} {' '.join(map(str, qubits))} " + " ".join(f"{v} 0" for _ in range(dim * dim))


def spread_qubits(t: int, n: int, k: int) -> list[int]:
    """t first, then qubits spread around the register: t + n/2, t + n/3, t + n/5, ... (distinct)."""
    qs = [t]
    for step in (n // 2, n // 3, n // 5, n // 7, n // 11, 1, 2, 3):
        q = (t + step) % n
        if q not in qs:
            qs.append(q)
        if len(qs) == k:
            return qs
    raise ValueError(f"cannot pick {k} distinct qubits from {n}")


def op_line(kind: str, t: int, n: int) -> str:
    if kind == "h":
        return f"h {t}"
    if kind == "cx":
        return f"cx {(t + n // 2) % n} {t}"  # control far away, target t
    if kind.startswith("mat"):
        k = int(kind[3:])
        return mat_text(k, spread_qubits(t, n, k))
    raise ValueError(kind)


def run(path: Path, precision: str, reps: int) -> tuple[float, float]:
    out = subprocess.run([str(EXE), str(path), "--gpu", f"--{precision}", "--reps", str(reps)],
                         capture_output=True, text=True, check=True).stdout
    m = re.search(r"per op ([\d.]+) ms\s+bandwidth ([\d.]+) GB/s", out)
    if not m:
        raise RuntimeError(out)
    return float(m.group(1)), float(m.group(2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qubits", type=int, default=28)
    ap.add_argument("--precision", choices=["f32", "f64"], default="f32")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ops", default="h,cx,mat2,mat3")
    ap.add_argument("--targets", default=None, help="comma-separated; default: every qubit")
    ap.add_argument("--count", type=int, default=64, help="ops per circuit")
    args = ap.parse_args()

    n = args.qubits
    kinds = args.ops.split(",")
    targets = [int(x) for x in args.targets.split(",")] if args.targets else list(range(n))

    print(f"{n} qubits, {args.precision}, {args.count} ops per run, median of {args.reps}; "
          f"bandwidth = bytes of touched amplitudes, read+write; % of {CEILING_GBS:.0f} GB/s")
    print()
    print("| target | " + " | ".join(f"{k} ms/op | {k} %" for k in kinds) + " |")
    print("|---|" + "---|---|" * len(kinds))
    with tempfile.TemporaryDirectory() as tmp:
        for t in targets:
            cells = []
            for kind in kinds:
                path = Path(tmp) / f"{kind}_{t}.qc"
                line = op_line(kind, t, n)
                path.write_text(f"qubits {n}\n" + (line + "\n") * args.count)
                ms, gbs = run(path, args.precision, args.reps)
                cells += [f"{ms:.3f}", f"{100 * gbs / CEILING_GBS:.0f}%"]
            print(f"| {t} | " + " | ".join(cells) + " |", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
