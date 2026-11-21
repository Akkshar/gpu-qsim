"""Run one circuit on a Linux-only baseline and print timings as JSON. Runs inside WSL2.

    python bench/baselines_wsl.py --backend aer-gpu|aer-gpu-nofuse|aer-cpu|custatevec \
        --circuit bench/circuits/m4/qft24.qc [--precision f32|f64] [--reps 3]

Venvs (see docs/dev-notes/PLAN.md, Toolchain): Aer backends run in ~/gpu-qsim-baselines/.venv-aer,
cuStateVec in ~/gpu-qsim-baselines/.venv-cusv (they need different cuQuantum builds; override the
location with QSIM_BASELINES when running bench/m4_bench.py).

What's timed (one warm-up run first, not counted):
  aer-*       Aer's own experiment time (result.results[0].time_taken): state init + simulation,
              excluding Python, transpile and result conversion. The circuit ends in measure_all
              with shots=1 so Aer can't skip execution; that adds one sampling pass.
              aer-gpu / aer-cpu use Aer's default gate fusion (up to 5 qubits); aer-gpu-nofuse turns
              it off, for a kernel-vs-kernel comparison with unfused qsim and cuStateVec.
  custatevec  CUDA events around applying every gate with custatevecApplyMatrix to a CuPy state
              (reset excluded). No fusion: cuStateVec is a gate-application library. Includes one
              Python call per gate (~10-20 us), which matters only for small states.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuits  # noqa: E402

CUDA_C_32F, CUDA_C_64F = 4, 5


def run_aer(lines, backend: str, precision: str, reps: int, extra: dict | None = None) -> list[float]:
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    device = "CPU" if backend == "aer-cpu" else "GPU"
    opts = dict(method="statevector", device=device, precision="single" if precision == "f32" else "double")
    if backend == "aer-gpu-nofuse":
        opts["fusion_enable"] = False
    if backend == "aer-gpu-cusv":
        opts["cuStateVec_enable"] = True
    opts.update(extra or {})
    sim = AerSimulator(**opts)
    qc = circuits.to_qiskit(lines)
    qc.measure_all()
    tqc = transpile(qc, sim, optimization_level=0)
    times = []
    for i in range(reps + 1):
        res = sim.run(tqc, shots=1).result()
        if not res.success:
            raise RuntimeError(res.status)
        if i:  # first run is warm-up
            times.append(res.results[0].time_taken * 1e3)
    return times


def run_custatevec(lines, precision: str, reps: int, dump: str | None = None) -> list[float]:
    import cupy as cp
    import numpy as np
    import cuquantum.bindings.custatevec as cusv

    n, ops = circuits.parse_qc(lines)
    dt_np = np.complex64 if precision == "f32" else np.complex128
    dt_cuda = CUDA_C_32F if precision == "f32" else CUDA_C_64F
    compute = cusv.ComputeType.COMPUTE_32F if precision == "f32" else cusv.ComputeType.COMPUTE_64F

    handle = cusv.create()
    prepared = []
    ws_size = 0
    for op in ops:
        m = np.ascontiguousarray(circuits.op_matrix(op), dtype=dt_np)
        _, _, controls, targets, _ = op
        ws_size = max(ws_size, cusv.apply_matrix_get_workspace_size(
            handle, dt_cuda, n, m.ctypes.data, dt_cuda, cusv.MatrixLayout.ROW, 0,
            len(targets), len(controls), compute))
        prepared.append((m, list(targets), list(controls)))
    ws = cp.cuda.alloc(ws_size) if ws_size else None
    ws_ptr = ws.ptr if ws is not None else 0

    sv = cp.zeros(2**n, dtype=dt_np)
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    times = []
    for i in range(reps + 1):
        sv.fill(0)
        sv[0] = 1
        cp.cuda.Device().synchronize()
        start.record()
        for m, targets, controls in prepared:
            cusv.apply_matrix(handle, sv.data.ptr, dt_cuda, n, m.ctypes.data, dt_cuda, cusv.MatrixLayout.ROW, 0,
                              targets, len(targets), controls, 0, len(controls), compute, ws_ptr, ws_size)
        stop.record()
        stop.synchronize()
        if i:
            times.append(cp.cuda.get_elapsed_time(start, stop))
    if dump:
        cp.asnumpy(sv).astype(np.complex128).tofile(dump)  # same raw format as qsim_cli --dump
    cusv.destroy(handle)
    return times


def bandwidth_probe(qubits: int) -> dict:
    """Device-to-device copy bandwidth from inside WSL, to compare with tools/bwprobe.cu on Windows."""
    import cupy as cp

    a = cp.zeros(2**qubits, dtype=cp.complex64)
    b = cp.empty_like(a)
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    times = []
    for i in range(11):
        start.record()
        cp.copyto(b, a)
        stop.record()
        stop.synchronize()
        if i:
            times.append(cp.cuda.get_elapsed_time(start, stop))
    ms = sorted(times)[len(times) // 2]
    return {"qubits": qubits, "copy_ms": ms, "GBps": 2 * a.nbytes / (ms * 1e-3) / 1e9}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True,
                    choices=["aer-gpu", "aer-gpu-nofuse", "aer-gpu-cusv", "aer-cpu", "custatevec", "bwprobe"])
    ap.add_argument("--aer-opt", action="append", default=[], metavar="KEY=VALUE",
                    help="extra AerSimulator option, e.g. fusion_max_qubit=3 (ints/floats/bools parsed)")
    ap.add_argument("--circuit")
    ap.add_argument("--precision", choices=["f32", "f64"], default="f32")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dump", help="custatevec only: write the final state like qsim_cli --dump")
    ap.add_argument("--qubits", type=int, default=28, help="bwprobe only")
    args = ap.parse_args()
    if args.backend == "bwprobe":
        print(json.dumps(bandwidth_probe(args.qubits)))
        return 0
    lines = Path(args.circuit).read_text().splitlines()
    if args.backend == "custatevec":
        times = run_custatevec(lines, args.precision, args.reps, args.dump)
    else:
        extra = {}
        for kv in args.aer_opt:
            k, v = kv.split("=", 1)
            extra[k] = {"true": True, "false": False}.get(v.lower(), int(v) if v.lstrip("-").isdigit() else v)
        times = run_aer(lines, args.backend, args.precision, args.reps, extra)
    print(json.dumps({"backend": args.backend, "aer_opts": args.aer_opt, "ms": times}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
