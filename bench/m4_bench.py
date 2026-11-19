"""M4: qsim vs Qiskit Aer (CPU, GPU) and NVIDIA cuStateVec on the same circuits.

Runs every (family, qubits) on every backend in interleaved rounds (round 1: every backend once,
round 2: every backend once, ...) so thermal drift hits all backends alike, and saves JSON results
after each circuit. Plot with bench/m4_plot.py.

    python bench/m4_bench.py [--families qft,layered,qaoa,ghz] [--qubits 16,20,24,26,28]
                             [--rounds 3] [--out bench/results/m4.json] [--backends ...]

Backends (all single precision unless --precision f64):
  qsim             this project, GPU, no fusion                      (Windows, native)
  qsim-fused       this project, GPU, --fuse 4                       (Windows, native)
  custatevec       NVIDIA cuStateVec, one custatevecApplyMatrix per gate, no fusion   (WSL)
  aer-gpu-cusv     Qiskit Aer GPU with cuStateVec_enable + Aer's default fusion: Aer's fastest
                   GPU configuration on this machine (WSL)
  aer-gpu          Qiskit Aer GPU, its own kernels, default fusion   (WSL; capped by --aer-gpu-max)
  aer-cpu          Qiskit Aer CPU, all 20 threads, default fusion    (WSL; capped by --aer-cpu-max)

Timing: qsim = min of 2 runs per round (the first includes lazy kernel loading); WSL backends =
one timed run after an untimed warm-up in the same process. See bench/baselines_wsl.py for exactly
what each baseline's number covers. WSL device bandwidth was checked equal to Windows (357 vs 358 GB/s).
Run on AC power.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuits  # noqa: E402

ROOT = circuits.ROOT
EXE = ROOT / "build" / "Release" / "qsim_cli.exe"
WSL_REPO = "/mnt/c/Users/akksh/Claude/gpu-qsim"
VENVS = "/home/akksh/gpu-qsim-baselines"
ALL_BACKENDS = ["qsim", "qsim-fused", "custatevec", "aer-gpu-cusv", "aer-gpu", "aer-cpu"]


def run_qsim(path: Path, precision: str, fuse: int) -> float:
    cmd = [str(EXE), str(path), "--gpu", f"--{precision}", "--reps", "2"]
    if fuse:
        cmd += ["--fuse", str(fuse)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return float(re.search(r"min ([\d.]+)", out).group(1))


def run_wsl(backend: str, path: Path, precision: str) -> float:
    venv = ".venv-cusv" if backend == "custatevec" else ".venv-aer"
    rel = path.relative_to(ROOT).as_posix()
    cmd = ["wsl", "-d", "Ubuntu-24.04", "--cd", WSL_REPO, "--", f"{VENVS}/{venv}/bin/python",
           "bench/baselines_wsl.py", "--backend", backend, "--circuit", rel, "--precision", precision, "--reps", "1"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    line = next(l for l in out.splitlines() if l.startswith("{"))
    return json.loads(line)["ms"][0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", default="qft,layered,qaoa,ghz")
    ap.add_argument("--qubits", default="16,20,24,26,28")
    ap.add_argument("--backends", default=",".join(ALL_BACKENDS))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--precision", choices=["f32", "f64"], default="f32")
    ap.add_argument("--fuse", type=int, default=4, help="block size for qsim-fused")
    ap.add_argument("--aer-cpu-max", type=int, default=26, help="largest n for aer-cpu (slow)")
    ap.add_argument("--aer-gpu-max", type=int, default=26, help="largest n for aer-gpu native kernels (slow)")
    ap.add_argument("--out", default=str(ROOT / "bench" / "results" / "m4.json"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else {"runs": []}
    results["meta"] = {"precision": args.precision, "fuse": args.fuse, "rounds": args.rounds,
                       "date": time.strftime("%Y-%m-%d %H:%M")}
    backends = args.backends.split(",")

    for fam in args.families.split(","):
        for n in [int(x) for x in args.qubits.split(",")]:
            lines = circuits.FAMILIES[fam](n)
            path = circuits.write_qc(lines, ROOT / "bench" / "circuits" / "m4" / f"{fam}{n}.qc")
            active = [b for b in backends
                      if not (b == "aer-cpu" and n > args.aer_cpu_max) and not (b == "aer-gpu" and n > args.aer_gpu_max)]
            times = {b: [] for b in active}
            for _ in range(args.rounds):
                for b in active:
                    try:
                        if b == "qsim":
                            ms = run_qsim(path, args.precision, 0)
                        elif b == "qsim-fused":
                            ms = run_qsim(path, args.precision, args.fuse)
                        else:
                            ms = run_wsl(b, path, args.precision)
                        times[b].append(ms)
                    except subprocess.CalledProcessError as e:
                        print(f"  {fam}{n} {b} FAILED: {e.stderr.strip()[-300:]}", flush=True)
            results["runs"] = [r for r in results["runs"] if not (r["family"] == fam and r["qubits"] == n)]
            for b, ts in times.items():
                if ts:
                    results["runs"].append({"family": fam, "qubits": n, "gates": len(lines) - 1, "backend": b,
                                            "ms": ts, "median_ms": sorted(ts)[len(ts) // 2]})
            out.write_text(json.dumps(results, indent=1))
            summary = "  ".join(f"{b} {sorted(t)[len(t) // 2]:.1f}" for b, t in times.items() if t)
            print(f"{fam:8} n={n:2}  {len(lines) - 1:5} gates  {summary}", flush=True)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
