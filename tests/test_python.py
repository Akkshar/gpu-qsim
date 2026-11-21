"""Python bindings, checked against Qiskit.  Run:  python -m pytest tests/test_python.py -q
(after building with the Python module: .\\build.ps1, which turns it on when .venv has nanobind)."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import qsim  # noqa: E402

DEVICES = ["cpu"] + (["gpu"] if qsim.has_cuda else [])


def test_bell_builder():
    psi = qsim.statevector(qsim.Circuit(2).h(0).cx(0, 1), device="cpu", precision="f64")
    assert psi.dtype == np.complex128 and psi.shape == (4,)
    np.testing.assert_allclose(psi, [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)], atol=1e-15)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("precision", ["f32", "f64"])
def test_random_circuits_match_qiskit(device, precision):
    from qiskit.quantum_info import Statevector
    import qiskit_check

    rng = random.Random(5)
    tol = 2e-4 if precision == "f32" else 1e-10
    for _ in range(20):
        n = rng.randint(2, 10)
        qc, lines = qiskit_check.random_ops(n, rng.randint(5, 60), rng, with_qiskit=True)
        compiled = qsim.parse_qc("\n".join(lines))
        for fuse_qubits in (0, 4):
            got = qsim.statevector(compiled, device=device, precision=precision, fuse_qubits=fuse_qubits)
            np.testing.assert_allclose(got, Statevector(qc).data, atol=tol)


def test_qasm_from_qiskit():
    from qiskit import qasm2
    from qiskit.quantum_info import Statevector
    import qiskit_check

    rng = random.Random(9)
    for _ in range(10):
        qc, _ = qiskit_check.random_ops(6, 40, rng, with_qiskit=True, allow_mat=False)
        c = qsim.Circuit.from_qasm(qasm2.dumps(qc))
        np.testing.assert_allclose(qsim.statevector(c, device="cpu", precision="f64"), Statevector(qc).data,
                                   atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_counts(device):
    counts = qsim.sample_counts(qsim.Circuit(3).h(0).cx(0, 1).cx(1, 2), shots=4000, seed=3, device=device)
    assert set(counts) == {"000", "111"}
    assert sum(counts.values()) == 4000
    assert abs(counts["000"] - 2000) < 200
    assert counts == qsim.sample_counts(qsim.Circuit(3).h(0).cx(0, 1).cx(1, 2), shots=4000, seed=3, device=device)
    # bitstring order: qubit n-1 first
    assert qsim.sample_counts(qsim.Circuit(3).x(0), shots=10, device=device) == {"001": 10}


def test_unitary_and_low_level_api():
    # unitary(): qubits[0] is the matrix's low bit. X on bit 0 of a 2-qubit matrix, targets [2, 0] -> flips qubit 2.
    m = np.kron(np.eye(2), [[0, 1], [1, 0]])
    psi = qsim.statevector(qsim.Circuit(3).unitary(m, [2, 0]), device="cpu", precision="f64")
    assert abs(psi[0b100]) == pytest.approx(1)

    c = qsim.Circuit(4).h(0).h(1).h(2).h(3).cx(0, 1).cx(2, 3).compile()
    assert len(c) == 6 and c.num_qubits == 4
    assert len(qsim.fuse(c, 2)) < len(c)
    sv = qsim.simulator(4, device="cpu", precision="f64")
    sv.run(c)
    assert sv.norm_squared() == pytest.approx(1)
    assert sv.probabilities().sum() == pytest.approx(1)
    assert sv.measure(0, seed=1) in (0, 1)


def test_errors():
    with pytest.raises(RuntimeError, match="unsupported gate"):
        qsim.Circuit.from_qasm('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nfoo q[0];')
    with pytest.raises(RuntimeError, match="out of range"):
        qsim.parse_qc("qubits 2\nh 5")
    with pytest.raises(ValueError):
        qsim.fuse(qsim.parse_qc("qubits 1\nh 0"), 9)
    with pytest.raises(ValueError):
        qsim.simulator(2, device="tpu")
