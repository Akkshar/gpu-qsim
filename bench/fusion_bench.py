"""Gate-fusion benchmark: the same circuits on the GPU with fusion off and at several block sizes.

Circuits (all written as .qc into bench/circuits/):
  qft      textbook QFT: h + controlled-phase ladder per qubit, then the reversal swaps
  layered  random-circuit-sampling style: each layer = a random u3 on every qubit, then cz on
           neighbouring pairs (even pairs on even layers, odd pairs on odd layers)
  ghz      h 0, then a cx chain
  random   tests/qiskit_check.py mix: random gates on random (far-apart) qubits; little to fuse

    python bench/fusion_bench.py [--qubits 28] [--precision f32] [--reps 3] [--fuse 0,2,3,4,5]
                                 [--circuits qft,layered,ghz,random] [--layers 20]

Prints a markdown table. Benchmark on AC power.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / "Release" / ("qsim_cli.exe" if sys.platform == "win32" else "qsim_cli")
OUT = ROOT / "bench" / "circuits"
sys.path.insert(0, str(ROOT / "tests"))


def qft(n: int) -> list[str]:
    lines = [f"qubits {n}"]
    for j in range(n):
        lines.append(f"h {j}")
        for k in range(j + 1, n):
            lines.append(f"cp {repr(math.pi / 2 ** (k - j))} {k} {j}")
    for j in range(n // 2):
        lines.append(f"swap {j} {n - 1 - j}")
    return lines


def layered(n: int, layers: int, seed: int = 7) -> list[str]:
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


def random_mix(n: int, depth: int = 200, seed: int = 7) -> list[str]:
    import qiskit_check  # tests/qiskit_check.py

    _, lines = qiskit_check.random_ops(n, depth, random.Random(seed), with_qiskit=False)
    return lines


def run(path: Path, precision: str, reps: int, fuse: int) -> tuple[float, int]:
    cmd = [str(EXE), str(path), "--gpu", f"--{precision}", "--reps", str(reps)]
    if fuse:
        cmd += ["--fuse", str(fuse)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    ms = float(re.search(r"time\s+([\d.]+) ms", out).group(1))
    m = re.search(r"fusion \d+ ops -> (\d+) ops", out)
    ops = int(m.group(1)) if m else int(re.search(r"ops (\d+)", out).group(1))
    return ms, ops


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qubits", type=int, default=28)
    ap.add_argument("--precision", choices=["f32", "f64"], default="f32")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--fuse", default="0,2,3,4,5")
    ap.add_argument("--circuits", default="qft,layered,ghz,random")
    ap.add_argument("--layers", type=int, default=20)
    args = ap.parse_args()

    n = args.qubits
    makers = {"qft": lambda: qft(n), "layered": lambda: layered(n, args.layers), "ghz": lambda: ghz(n),
              "random": lambda: random_mix(n)}
    fuses = [int(x) for x in args.fuse.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"{n} qubits, {args.precision}, GPU, median of {args.reps}; ops = kernel launches after fusion")
    print()
    print("| circuit | gates | " + " | ".join(f"K={k} ops | K={k} ms" if k else "unfused ms" for k in fuses)
          + " | best speedup |")
    print("|---|---|" + "---|" * (sum(2 if k else 1 for k in fuses)) + "---|")
    for name in args.circuits.split(","):
        lines = makers[name]()
        path = OUT / f"{name}{n}.qc"
        path.write_text("\n".join(lines) + "\n")
        cells, base, best = [], None, None
        for k in fuses:
            ms, ops = run(path, args.precision, args.reps, k)
            if k == 0:
                base = ms
                cells.append(f"{ms:.0f}")
            else:
                cells += [str(ops), f"{ms:.0f}"]
            best = ms if best is None else min(best, ms)
        speedup = f"{base / best:.2f}x" if base else "-"
        print(f"| {name} | {len(lines) - 1} | " + " | ".join(cells) + f" | {speedup} |", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
