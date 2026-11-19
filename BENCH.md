# BENCH.md — measurement log

One entry per measurement or optimisation: what changed, the numbers, and what they mean.
Always note the power state; this laptop's GPU behaves very differently on battery.

---

## 2026-11-19 — M0: toolchain + bandwidth ceiling (provisional)

**Setup:** CUDA 13.1.1 (driver 592.00 supports up to 13.1), MSVC 19.44 (VS Build Tools 2022),
CMake 4.4.2, Visual Studio generator, sm_86. **On battery, Windows "Silent" profile → GPU capped at
55 W (max 120 W), memory clock 6001 MHz.**

`bwprobe` (tools/bwprobe.cu), median of 20, complex64:

| Test | 28 q (2 GiB) | 24 q (128 MiB) |
|---|---|---|
| cudaMemcpy D2D | 286.5 GB/s | 283.7 GB/s |
| copy kernel (float4, grid-stride) | 281.5 GB/s | 273.1 GB/s |
| in-place read-modify-write (float4) | 282.2 GB/s | 199.2 GB/s |

- The driver reports a **384 GB/s** peak in this power state (12 Gbps × 256-bit), not the 448 GB/s
  spec (14 Gbps). We reach ~74% of it, so 282 GB/s is the working ceiling *for now*.
- At 28 qubits a full-state pass (read + write 2 GiB) takes ~15.2 ms → that's the floor for one
  1-qubit gate in this power state.
- ~~Open question: in-place RMW dropped to 199 GB/s at 24 q.~~ Answered below: a one-off.

**Superseded** by the on-AC run above (358 GB/s).

## M1: CPU reference v1 — general `apply` (scan all 2^n indices, skip non-matching)

**Setup:** single thread, Release, on AC. `qsim_cli <file> --reps 3`, median. Circuits from
`tests/qiskit_check.py emit --seed 7` (mixed gates incl. controlled and 1–3 qubit `mat`).

| Circuit | f64 per op | f32 per op | f64 "full-pass GB/s" |
|---|---|---|---|
| 16 q, 200 random ops | 0.24 ms | 0.25 ms | 8.7 |
| 20 q, 200 random ops | 4.30 ms | 4.30 ms | 7.8 |
| 24 q, 50 random ops | 105 ms (spread 74–108) | 100 ms | 5.1 |
| 20 q, 200 × `h` only | 5.09 ms | — | 6.6 |

Correctness: all 39 Catch2 cases; qiskit_check 200/200 f64 (worst 9.9e-16), 200/200 f32 (1.2e-7),
500/500 f64 depth ≤ 100 (2.0e-15).

- **f32 is no faster than f64** → this loop is bound by per-index overhead (2^n iterations, branch
  per index, gather/scatter through small vectors, a dense 2^k matmul in double), not by memory.
- Next: a dedicated 1-target path (pair counter + `insert_zero_bit`, matrix converted to `amp_t`
  once, no gather vectors) and controls by inserting bits instead of skipping. Re-measure.

## 2026-11-19 — M0: the real ceiling (on AC) ← use this one

**Setup:** plugged in, Windows scheme still "Silent", GPU in P0, memory at its **6001 MHz max**,
SM 1620 MHz, 66 W, 73 °C. Three runs each:

| Test | 28 q (2 GiB) | 24 q (128 MiB) |
|---|---|---|
| cudaMemcpy D2D | 358.7–359.2 GB/s | 356.7–357.1 GB/s |
| copy kernel (float4) | 357.5–358.1 GB/s | 355.7–356.2 GB/s |
| in-place RMW (float4) | 357.9–358.5 GB/s | 355.2–356.2 GB/s |

- **Working ceiling: 358 GB/s**, 93% of the 384 GB/s driver peak. Every gate kernel is scored as a
  % of this number.
- This GPU's memory maxes out at 6001 MHz (12 Gbps × 256-bit = 384 GB/s) even on AC in P0, so the
  448 GB/s spec-sheet figure doesn't apply to this laptop. A Turbo profile raises power and core clocks,
  not the memory clock, so it shouldn't move a memory-bound ceiling.
- Floor for one full-state pass (read + write): **12.0 ms at 28 q**, 24.0 ms at 29 q (complex64).
- Battery vs AC: 282 → 358 GB/s (+27%). **Benchmark only on AC**, and log the power state.

## 2026-11-19 — M0: first Nsight Compute profile

**How:** GPU counters stay admin-only (Windows default, by choice), so profiling goes through
`tools/ncu-admin.ps1`, which runs ncu elevated (one UAC prompt) and saves `profiles/<name>.ncu-rep`.
Reading a report needs no admin. Still on battery / Silent.

```
.\tools\ncu-admin.ps1 -Name m0-probe24 --set full --kernel-id "::regex:copy_f4|scale_inplace_f4:4" `
    C:\Users\akksh\Claude\gpu-qsim\build\Release\bwprobe.exe 24
ncu --import profiles\m0-probe24.ncu-rep --page raw --csv --metrics gpu__time_duration.sum,dram__bytes_read.sum,...
```

| Kernel (24 q, 128 MiB) | Time | DRAM read | DRAM write | Effective | L2 hit | Warps active |
|---|---|---|---|---|---|---|
| copy_f4 | 944.8 µs | 134.2 MB | 133.7 MB | 283 GB/s | 50.0% | 93.8% |
| scale_inplace_f4 | 946.3 µs | 134.2 MB | 133.6 MB | 283 GB/s | 50.0% | 93.5% |

- Both kernels move exactly the buffer size in and out: no extra DRAM traffic, so in-place
  read-modify-write costs the same as a copy. Gate kernels can work in place without penalty.
- L2 hit rate of 50% is the expected pattern for streaming: reads miss, writes hit (they're allocated in L2).
- ncu reported the DRAM clock at **5.49 GHz** during profiling, and nvidia-smi showed 5501 MHz current /
  6001 MHz max on battery. If the memory really runs at 5.5 GHz on battery, the true peak is ~352 GB/s
  and 283 GB/s is ~80% of it, which is normal. (ncu locks clocks to base by default, so this is a hint,
  not proof; recheck plugged in.)
- **The 199 GB/s result was a one-off.** Four unprofiled reruns of `bwprobe 24`: memcpy 283.7–284.0,
  copy 280.1–281.6, in-place 280.7–281.3 GB/s. Most likely a clock/power-state transition on battery.
  Lesson: always run benchmarks several times and report the spread.

## 2026-11-19 — M0: baselines (WSL2)

GHZ, 20 qubits, complex64, `bench/m0_smoke.py` — correctness check, not a benchmark:

| Backend | Time | Fidelity vs exact |
|---|---|---|
| Qiskit Aer 0.18.4, CPU | 66.5 ms | 0.9999999 |
| Qiskit Aer 0.18.4, GPU | 41.6 ms | 0.9999999 |
| cuStateVec (cuquantum-python 26.11) | 0.96 ms | 0.9999999 |

Aer numbers include Python/transpile-cache/result-copy overhead; M4 must time the simulation alone.
