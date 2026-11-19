"""M0 smoke test for the baselines (run inside WSL2; Aer-GPU and cuStateVec are Linux-only).

Builds an n-qubit GHZ circuit (H on q0, then a CX chain) and runs it on:
  - Qiskit Aer, CPU statevector
  - Qiskit Aer, GPU statevector
  - cuStateVec directly (CuPy buffer + custatevecApplyMatrix)
checks each against the exact GHZ state, and prints rough timings. Not a benchmark — M4 does that.

Aer-GPU 0.18 needs cuQuantum for CUDA 12 and cuquantum-python 26 is CUDA 13; they overwrite each
other's libraries, so they live in two venvs (~/gpu-qsim-baselines/.venv-aer, .venv-cusv):

    .venv-aer/bin/python  bench/m0_smoke.py aer  [n_qubits]
    .venv-cusv/bin/python bench/m0_smoke.py cusv [n_qubits]
"""

import sys
import time

import numpy as np

WHICH = sys.argv[1] if len(sys.argv) > 1 else "aer"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
CUDA_C_32F = 4  # cudaDataType enum value for complex64


def ghz(n: int):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    qc.save_statevector()
    return qc


def run_aer(device: str, n: int) -> tuple[np.ndarray, float]:
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = AerSimulator(method="statevector", device=device, precision="single")
    qc = transpile(ghz(n), sim)
    sim.run(qc).result()  # warm-up (GPU context, allocations)
    t0 = time.perf_counter()
    res = sim.run(qc).result()
    dt = time.perf_counter() - t0
    return np.asarray(res.get_statevector(), dtype=np.complex64), dt


def run_custatevec(n: int) -> tuple[np.ndarray, float]:
    import cupy as cp
    import cuquantum.bindings.custatevec as cusv

    h = (np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)).astype(np.complex64)
    x = np.array([[0, 1], [1, 0]], dtype=np.complex64)
    handle = cusv.create()

    def apply(sv, mat, target, controls=()):
        cusv.apply_matrix(
            handle, sv.data.ptr, CUDA_C_32F, n,
            mat.ctypes.data, CUDA_C_32F, cusv.MatrixLayout.ROW, 0,
            [target], 1, list(controls), 0, len(controls),
            cusv.ComputeType.COMPUTE_32F, 0, 0,
        )

    def circuit():
        sv = cp.zeros(1 << n, dtype=cp.complex64)
        sv[0] = 1
        apply(sv, h, 0)
        for q in range(n - 1):
            apply(sv, x, q + 1, controls=[q])
        cp.cuda.Device().synchronize()
        return sv

    circuit()  # warm-up
    t0 = time.perf_counter()
    sv = circuit()
    dt = time.perf_counter() - t0
    out = cp.asnumpy(sv)
    cusv.destroy(handle)
    return out, dt


def main() -> None:
    print(f"GHZ, {N} qubits, complex64\n")
    expected = np.zeros(1 << N, dtype=np.complex64)
    expected[0] = expected[-1] = 1 / np.sqrt(2)

    backends = {
        "aer": [("Aer CPU", lambda: run_aer("CPU", N)), ("Aer GPU", lambda: run_aer("GPU", N))],
        "cusv": [("cuStateVec", lambda: run_custatevec(N))],
    }[WHICH]

    ok = True
    for name, fn in backends:
        try:
            sv, dt = fn()
            fid = abs(np.vdot(expected, sv)) ** 2
            good = fid > 1 - 1e-5
            ok &= good
            print(f"{name:<11} {dt * 1e3:9.2f} ms   fidelity {fid:.7f}   {'OK' if good else 'MISMATCH'}")
        except Exception as e:  # report and keep going so one broken backend doesn't hide the others
            ok = False
            print(f"{name:<11} FAILED: {type(e).__name__}: {e}")

    print("\nall backends agree" if ok else "\nsomething is wrong — see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
