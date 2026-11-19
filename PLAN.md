# gpu-qsim — plan

A CUDA state-vector quantum circuit simulator, written by hand, profiled and benchmarked against
Qiskit Aer and NVIDIA cuStateVec. Planned 2026-11-19. Nothing built yet.

## What the project is meant to show

"I wrote GPU kernels, found the bottleneck with a profiler, removed it step by step, and can
prove each step with numbers." The simulator is the vehicle; the optimisation log is the product.

Target resume line (fill in real numbers at the end):
> Built a CUDA state-vector quantum simulator (C++20/CUDA, 29 qubits on an 8 GB GPU). Gate fusion and
> shared-memory blocking reached X% of peak DRAM bandwidth; Y× faster than Qiskit Aer (CPU) and within
> Z% of NVIDIA cuStateVec on QFT/random circuits. Python bindings, OpenQASM input, 400+ correctness tests.

## Ground rules

- I write every kernel and the core simulator. Claude may explain, review, and write plumbing
  (CMake, bench harness, plotting, CI) but not the kernels.
- Every optimisation gets a before/after measurement in `BENCH.md` with the Nsight Compute numbers.
  No measurement, no claim.
- Correctness first: nothing is "faster" until it matches the CPU reference.

## Hardware and limits

- RTX 3070 Ti Laptop, 8 GB GDDR6, 256-bit. Ampere, sm_86. This unit's memory maxes at 6001 MHz
  (12 Gbps) → **384 GB/s peak** per the driver, not the 448 GB/s spec sheet. **Measured ceiling on AC:
  358 GB/s** (BENCH.md, 2026-11-19). On battery it drops to ~282 GB/s.
- State size = 2^n amplitudes × 8 B (complex64) or 16 B (complex128).
  - complex64: 28 q = 2 GiB, **29 q = 4 GiB (max)**, 30 q = 8 GiB (does not fit).
  - complex128: **28 q = 4 GiB (max)**.
- A 1-qubit gate reads and writes the whole state once. At 28 q complex64 that is 4 GiB of traffic,
  which takes ≈ 12.0 ms at the measured 358 GB/s. That is the speed limit every kernel is measured against.
- Laptop caveats: benchmark only on AC, warm up, report median of ≥10 runs plus spread, log the power
  state. The GPU also drives the display, so leave ~1 GB headroom.

## Toolchain (set up 2026-11-19, M0)

- CUDA Toolkit **13.1.1** on Windows (matches driver 592.00, which supports up to 13.1; no driver
  installed with it), Nsight Compute 2025.4.1, MSVC 19.44, CMake 4.4.2 (WinLibs build).
- Build: `cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T cuda=13.1` then
  `cmake --build build --config Release`. Shells opened before the CUDA install lack `CUDA_PATH_V13_1`.
- WSL2 baselines live in `~/gpu-qsim-baselines`, as **two venvs** because Aer-GPU 0.18.4 needs
  cuQuantum for CUDA 12 (<24.11) and cuquantum-python 26 is CUDA 13; they overwrite each other's libs:
  `.venv-aer` (qiskit 2.6.2, qiskit-aer-gpu 0.18.4) and `.venv-cusv` (cuquantum-python-cu13 26.11, cupy 14.2).
- Peak bandwidth depends on the power state: 358 GB/s on AC, ~282 GB/s on battery. Always log the
  power state with a result (see BENCH.md).

- **Main dev: native Windows** (CMake + MSVC + nvcc). Nsight Compute works there.
- **Baselines in WSL2**: `qiskit-aer-gpu` and `cuquantum-python` (cuStateVec) are Linux-only.
- Keep the CMake build portable; the Arch dual-boot can take over later.
- GPU performance counters on Windows need "Allow access to the GPU performance counters to all
  users" in NVIDIA Control Panel (Developer → Manage GPU Performance Counters). That is a setting
  the user changes, not Claude.

## Architecture

```
gpu-qsim/
  include/qsim/        public headers: StateVector<T>, Circuit, Gate, Executor
  src/cpu/             reference simulator (plain C++, then OpenMP)
  src/cuda/            kernels: apply_1q, apply_controlled, apply_dense_k, measure, blocked
  src/fusion/          circuit → fused blocks pass
  src/qasm/            OpenQASM 2.0 subset parser
  python/              nanobind bindings: qsim.Circuit, qsim.run(...)
  tests/               Catch2 (C++) + pytest (vs Qiskit)
  bench/               circuit generators, runner → results/*.json, plots
  BENCH.md             optimisation log, one entry per change
```

- `Circuit` is a flat list of ops (gate matrix + target qubits + control qubits).
- `Executor` interface with three backends: `CpuRef`, `CudaNaive`, `CudaOpt`. Same tests run on all.
- Qubit order is little-endian (qubit 0 = least significant bit), the same as Qiskit, so results can be
  compared directly.
- Precision is a template parameter (`float` / `double`).

## Milestones

### M0 — Setup and spike (½ day)
- Install CUDA 12.x on Windows. Check the MSVC C++ workload.
- CMake project with a vector-add kernel; run it under `ncu` and read DRAM throughput.
- WSL2: `nvidia-smi` works inside; install qiskit-aer-gpu and cuquantum-python in a uv venv.
- **Done when:** a kernel is profiled on Windows, and Aer-GPU runs a 20-qubit circuit in WSL.

### M1 — CPU reference (weekend 1) — DONE 2026-11-19 (v1 by you, v2 fast paths by Claude on request; see BENCH.md)
- Gates: H X Y Z S Sdg T Tdg RX RY RZ U3, CX CZ SWAP, CCX, generic controlled-U, generic 2×2 / 4×4.
- Index math: for target t, iterate k over 2^(n-1); i0 = insert a 0 bit at position t in k; i1 = i0 | (1<<t).
- Measurement: probabilities, sampling (cumulative sum + binary search), collapse and renormalise.
- Tests: 200 random circuits up to 12 qubits; fidelity |<ψ_qiskit|ψ_ours>|² > 1 − 1e-10 (f64).
- **Done when:** all tests pass and the CPU timings for 16–24 q are recorded as the first baseline.

### M2 — Naive GPU (weekend 1–2) — DONE 2026-11-20 (written by Claude on request; see BENCH.md)
- One thread per amplitude pair, the same bit-insert trick. Gate matrix passed as kernel arguments.
- Controlled gates: thread only over indices whose control bits are 1 (insert those bits too).
- Measurement on GPU: CUB reduce (norm), CUB scan (cumulative prob), batched binary search per shot.
- Report the **effective bandwidth** = bytes moved / kernel time, as a % of 448 GB/s.
- **Done when:** results match CPU (1e-5 f32 / 1e-12 f64) for every test, and the first % of peak is logged.

### M3 — Optimise (weekends 2–4; the core of the project) — M3.1 multi-target kernel DONE 2026-11-20
Each step: hypothesis → change → `ncu` before/after → entry in BENCH.md.
1. **Memory access per target qubit.** Sweep t = 0..n-1. Low t has pairs that are adjacent and
   interleaved; fix with vectorised loads (float4 / two pairs per thread). Goal: a flat line across t.
2. **Launch tuning.** Block size and grid-stride loops; CUDA Graphs so launch overhead stops
   dominating small circuits (≤20 q).
3. **Gate fusion.** Merge runs of gates on the same qubits into one dense 2^k×2^k matrix (k ≤ 4–5),
   so 10 gates become one pass over memory. A greedy pass first; measure how many passes it saves per circuit.
4. **Shared-memory blocking.** Load a 2^m-amplitude chunk (m ≈ 10–11) into shared memory, apply every
   gate that only touches the low m qubits, write back once. Many gates for the price of one pass.
5. **Qubit remapping.** Insert swaps so high qubits move into the local window, which lets step 4
   cover more gates. Measure when the swap pays for itself.
6. **Precision.** complex64 vs complex128: speed against accuracy (fidelity drift over deep circuits).
- **Done when:** 1q-gate kernel at ≥80% of peak bandwidth, and the fused/blocked path beats per-gate
  application by a measured factor on QFT and random circuits.

### M4 — Benchmarks (weekend 4)
- Circuits: GHZ, QFT, Grover, random layered circuits (1q random rotations + CZ ladder), one QAOA layer.
- Sweep n = 16 … 29 (f32) / 28 (f64). Depth fixed per circuit family.
- Compare against: own CPU (OpenMP), Qiskit Aer CPU, Qiskit Aer GPU, cuStateVec — same precision,
  same circuits, warm-up runs, median of 10, host↔device copy reported separately.
- Output: JSON results → plots (time vs n, speedup vs Aer, % of peak bandwidth, roofline).
- **Done when:** the charts are in the README and every number can be regenerated by `bench/run_all`.

### M5 — Polish (weekend 5)
- OpenQASM 2.0 subset input; `qsim run circuit.qasm --shots 1000` CLI.
- nanobind Python package: `qsim.Circuit().h(0).cx(0,1).run(shots=1000)`.
- GitHub repo with CI: CPU build + tests on every push (GitHub runners have no GPU; GPU tests run locally).
- README: what it is, architecture diagram, results charts, "what I tried that didn't work",
  how to reproduce. A short write-up post to link from the resume.

### Stretch (pick at most one)
- Noise: Monte Carlo trajectories (depolarising/amplitude damping) on top of the state vector.
- Past 8 GB: out-of-core 30–31 qubits with chunks in pinned host memory, overlapping copy and compute
  with streams.
- MPS (matrix product state) backend for 50+ qubit low-entanglement circuits, with a comparison of where
  it beats the state vector.

## Interview questions to be able to answer by the end
- Why is gate application memory-bound, and what is its arithmetic intensity?
- Why does the target qubit's position change the access pattern, and how did you fix it?
- How does gate fusion trade FLOPs for memory traffic, and where does it stop helping (k too big)?
- Why can't you just run 35 qubits? What would multi-GPU need (all-to-all for high qubits)?
- How did you verify correctness, and how big is float32 error after 1,000 gates?
- Where is cuStateVec still faster than yours, and why?

## Rough timeline
5 weekends for M0–M5, plus one for a stretch goal. M3 is where the time goes, and that's intended.
