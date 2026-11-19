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

## M3.2: gate fusion

**Pass** (`src/fusion/fuse.cpp`, `qsim_cli --fuse K`): greedy and order-preserving. Walk the ops, keep
adding to the current block while the union of touched qubits (targets + controls) is ≤ K, else
close it. A block of ≥ 2 ops becomes one dense 2^q × 2^q matrix (built in double on the host by
applying each op to the block's columns); a single-op block is emitted unchanged, so a lone cx keeps
its control and its half-state cost. Host cost is small (QFT-28: 420 → 103 ops in a few ms).

**Correctness:** 5 new [fusion] cases (fused == unfused for K = 1..6 on 8 random circuits; the block
rules; GPU running fused circuits, f32 and f64). qiskit_check (now also with `p`/`cp`): CPU fuse 3
200/200 (6.2e-16); GPU f64 fuse 5 200/200 (8.9e-16); GPU f32 fuse 4 200/200 (**8.2e-8**, lower than
unfused 2.6e-7 because fused matrices are built in double); GPU f64 fuse 6, ≤ 16 q 150/150 (1.1e-15).

**Results** — `bench/fusion_bench.py`, 28 q f32 GPU, median of 3, on AC. ms = whole circuit;
ops = kernel launches after fusion.

| circuit | gates | unfused | K=2 | K=3 | K=4 | K=5 | best |
|---|---|---|---|---|---|---|---|
| QFT (h + cp ladder + swaps) | 420 | 2793 ms | 392 ops / 2612 | 209 / 2532 | 140 / 1725 | **103 / 1472** | **1.90×** (K=5) |
| layered (u3 on all + cz pairs, 20 layers) | 830 | 8521 | 550 / 5162 | **450 / 4371** | 275 / 5005 | 245 / 5288 | **1.95×** (K=3) |
| GHZ | 28 | 189 | 27 / 176 | 14 / 169 | **9 / 122** | 7 / 134 | **1.55×** (K=4) |
| random mix (far-apart qubits) | 200 | 1837 | 177 / 1565 | 145 / 1275 | **104 / 1104** | 81 / 1167 | **1.66×** (K=4) |

Launch counts drop 3–4×, run time only 1.6–1.95×: fused launches cost more than the 12 ms floor
(layered K=4: 18 ms per launch; QFT K=5: 14 ms). Bigger K stops helping (layered peaks at K=3).

**Why: dense blocks on the lowest qubits are slow.** Same op, 64 copies, 28 q f32:

| block targets | ms/op | % of ceiling |
|---|---|---|
| mat3 {0,1,2} / {12,13,14} | 14.3 / 12.2 | 84% / 98% |
| mat4 {0,1,2,3} / {2..5} / {12..15} / {24..27} | 21.9 / 14.0 / 12.4 / 12.1 | **55%** / 85% / 97% / 99% |
| mat5 {0..4} / {12..16} | 26.7 / 12.2 | **45%** / 98% |

With targets 0..k-1 each thread's group is 2^k *contiguous* amplitudes (128–256 B in f32), so the
32 lanes of a warp sit 128–256 B apart: every load instruction uses 8 B from each of 32 different
lines and relies on L1 keeping them for the thread's other 2^k − 1 loads. With many warps in flight,
L1 thrashes. Fusion makes this common: blocks are naturally on neighbouring qubits.

**Next (M3.3):** stage low-qubit blocks through shared memory: a thread block loads a contiguous tile
with fully coalesced loads, applies the matrix from shared memory, writes the tile back coalesced
(the plan's "shared-memory blocking"; also the fix for low-qubit controls). Then a per-op timing
mode to attribute each fused launch's cost, and a cost-aware fusion (choose K per block).

## M3.1: multi-target kernel — compile-time k, matrix in the argument block

### Found on the way: everything had been compiled for sm_75
`ptxas -v` (now always on, `QSIM_PTXAS_INFO`) printed `for 'sm_75'`. `CMAKE_CUDA_ARCHITECTURES`
was set *after* `project()`, which had already filled in nvcc's default (75), so since M0 every kernel
was a Turing build that the driver JIT-compiled from PTX at load. Fixed by setting it before
`project()`. Re-measured the M2 kernels as native sm_86: within ±1% of the M2 numbers (h 98–100%,
cx 48–100%, mat2 61–68%, mat3 46–58%), so earlier results stand.

### Diagnosis (hypothesis 3 from M2: confirmed)
ptxas: `apply_kt` has a **512-byte stack frame (f32), 1024 (f64)**: the `in[64]` array indexed by a
runtime `dim` can't live in registers. ncu, one `mat3` op at 24 q f32 (`profiles/m3-kt-before`), ideal
traffic = 134.2 MB read + 134.2 MB written:

| | M2 `apply_kt` | M3 `apply_kt_fixed<f,3>` |
|---|---|---|
| DRAM read / write | 159.5 / **264.8 MB** | **134.3 / 133.5 MB** (= ideal) |
| local-memory load / store sectors | 33.6 M / 4.2 M (1.07 GB of L1 traffic) | **0 / 0** |
| instructions executed | 60.2 M | 32.1 M |
| registers / warps active | 41 / 69% | 53 / 64% |
| kernel time (ncu, base clocks) | 1.61 ms | **0.96 ms** |

Writes were 2× ideal: spilled `in[]` lines get evicted to DRAM. Local loads were 8 per amplitude
because the inner loop re-reads `in[c]` for every output row.

### Fix
`apply_kt_fixed<T, K>` for K = 2..4: K is a template parameter so every loop fully unrolls and
`in[]` is registers; matrix + member offsets are passed by value in the kernel arguments
(`__grid_constant__`, up to 16.6 KB, fine since CUDA 12.1 on sm_70+), so every matrix read is a
constant-bank broadcast. K = 6 keeps the M2 kernel.

K = 5 needed two more steps:
1. Same kernel: **25% of ceiling**. The 8 KB (f32) matrix exceeds the per-SM constant cache, and 167
   registers leave 8 warps per SM.
2. Matrix copied to shared memory per block, all loops unrolled: **5%** (215–224 ms/op). ptxas:
   **7.5 KB spills** per thread (f32): the compiler hoisted all 1024 shared loads into registers.
3. Shared memory + only the inner loop unrolled (`#pragma unroll 1` on rows): **82–97%**, 91 registers,
   no spills.

### Results — 28 q f32, `bench/gpu_sweep.py`, median of 5 (k=5: 3), on AC

| Op | M2 | **M3.1** | Notes |
|---|---|---|---|
| mat2, all 28 targets | 58–67% | **97–100%** | |
| mat3, all 28 targets | 46–58% | **99–101%**; dips 81–92% at t = 14, 15, 19 | dips: qubit 0 or 1 among the targets |
| mat4, all 28 targets | — | **95–98%**; dips 68–86% at t = 14, 18, 19, 23, 24 | same low-qubit effect |
| mat5, t = 0/7/14/21/27 | — (generic: ~25% est.) | **82–97%** | shared-memory variant |

f64, t = 0/7/27: mat2 40–47% (was 36%), mat3 21–23% (was 19%), mat4 11%, mat5 6%. Time doubles per
extra target (51 → 105 → 215 → 410 ms): FP64-bound. mat5 = 2^28 × 32 complex MACs × 8 flops in 0.41 s
≈ 168 GFLOP/s, ~60% of this GeForce's FP64 peak (1/64 of FP32). Nothing to gain without FP32.

Correctness: new [cuda] case with one op of every k = 2..6 (scattered, all-low, all-high, controlled);
53/53 Catch2 cases, qiskit_check gpu f64 200/200 (9.2e-16), f32 200/200 (2.6e-7; seed 9: 2.55e-7),
18 q 300/300 (1.7e-15).

**Remaining M3 targets:** low qubits (0/1) as controls (48%) or as non-first targets (68–92%): every
32-byte sector fetched is partly waste. Then gate fusion: every kernel is now at the memory floor,
so fewer passes over the state is the only way left to go faster.

## M2: naive GPU kernels — one launch per op

**Kernels** (`src/cuda/statevector.cu`): `apply_1t` (one thread per amplitude pair, 2x2 matrix in
kernel arguments) and `apply_kt` (one thread per group of 2^k, runtime `dim`, matrix + offsets in
global memory, gathered amplitudes in a per-thread array). Both use the same insert-zero-bit group
enumeration as the CPU, so no thread skips work. 256-thread blocks, grid capped at 32 × SMs,
grid-stride loops. Measurement: Thrust transform_reduce / transform_inclusive_scan / vectorised
upper_bound, uniforms drawn on the host exactly like the CPU (same seed → same samples).

**Correctness:** 12 new [cuda] Catch2 cases (GPU vs CPU reference: every target position, every
control/multi-target kind incl. a 6-target `mat`, 12 random circuits up to 15 q, sampling and
measurement identical to the CPU for the same seed). qiskit_check --backend gpu: 200/200 f64
(9.2e-16), 200/200 f32 (2.6e-7), 300/300 f64 up to 18 q / depth 80 (1.7e-15).

**Bandwidth sweep** — `bench/gpu_sweep.py`, 28 qubits, 64 identical ops per run, median of 5, on AC.
Bandwidth counts the amplitudes an op actually touches (read + write; a controlled op touches half),
as a % of the 358 GB/s ceiling. Floor for a full pass at 28 q f32 = 12.0 ms.

| Op (f32, 28 q) | ms/op | % of ceiling | Notes |
|---|---|---|---|
| `h t`, every t = 0..27 | 11.98–12.25 | **98–100%** | already at the memory limit, at every target position |
| `cx c t`, c ≥ 4 | 6.00–6.28 | **96–100%** | touches half the state, takes half the time |
| `cx c t`, c = 2, 3 | 6.60–6.67 | 90–91% | |
| `cx c t`, **c = 0, 1** | 12.1–12.5 | **48–49%** | control on a low qubit: touched amplitudes are interleaved with untouched ones, so every 32-byte sector fetched is half waste |
| `mat2` (dense 4x4), every t | 17.8–20.7 | **58–67%** | not memory-bound: see below |
| `mat3` (dense 8x8), every t | 20.7–26.3 | **46–58%** | worst when qubit 0/1 is among the targets in some positions |

| Op (f64, 28 q, t = 0/14/27) | ms/op | % of ceiling |
|---|---|---|
| `h` | 29.0–29.7 | 81–83% |
| `mat2` | 66.6–67.4 | 36% |
| `mat3` | 124–126 | 19% |

What this says (hypotheses for M3, to confirm with ncu):
1. **1-target f32 is done**: 98–100% at every target, including t = 0 where pairs are adjacent.
   Gate fusion (fewer passes), not a faster kernel, is the only way to beat 12 ms/gate here.
2. **Low-qubit controls waste half the bandwidth.** Fix candidates: have each thread handle both
   halves of the control-0 region (i.e. treat low controls as targets with an identity block), or
   remap qubits so controls never sit on bits 0–1.
3. **`apply_kt` is far from the memory limit even in f32.** Arithmetic is ~32 flops/amplitude for
   mat2, a few % of FP32 peak, so it's not compute. Likely culprits: `in[64]` indexed with a runtime
   `dim` lives in local memory (spills), and matrix/offset reads from global memory every iteration.
   Fix: template the kernel on k so the arrays become registers and the loops unroll; matrix in
   shared or constant memory.
4. **f64 multi-target is FP64-bound**: GeForce Ampere runs FP64 at 1/64 of FP32. mat3 is ~64
   flops/amplitude → ~140 GFLOP/s, roughly half the card's FP64 peak. f64 is for validation; f32 is
   the performance target (as for cuStateVec benchmarks).
5. Metric note: the first sweep run used "full-pass equivalent" bandwidth, which showed controlled
   gates at "200%". The CLI now counts touched bytes only; all numbers above use that.

## M1: CPU reference v2 — insert-bit enumeration + 1-target fast path

**Change:** groups are enumerated directly (insert a 0 at every target/control position, ascending,
then OR in the control mask) instead of scanning 2^n indices and skipping. One-target ops get a
register-only 2x2 path in the state's precision with a plain complex multiply (`cmul`, avoiding
std::complex's Annex G NaN handling). Multi-target ops keep v1's gather/multiply/scatter.
Same circuits, same machine, on AC, median of 5 (3 for 24 q):

| Circuit | v1 f64 | **v2 f64** | v1 f32 | **v2 f32** | speedup (f64 / f32) |
|---|---|---|---|---|---|
| 20 q, 200 × `h` | 5.09 ms/op | **1.67 ms/op** (20.1 GB/s) | — | **1.41 ms/op** | **3.1×** / — |
| 24 q, 200 × `h` | — | **39.3 ms/op** (13.7 GB/s) | — | — | — |
| 16 q, 200 random | 0.24 | **0.14** | 0.25 | **0.15** | 1.7× / 1.7× |
| 20 q, 200 random | 4.30 | **3.32** | 4.30 | **2.49** | 1.3× / 1.7× |
| 24 q, 50 random | 105 | **84** (spread 49–86) | 100 | **52** | 1.3× / 1.9× |

Correctness unchanged: 39/39 Catch2 cases; qiskit_check f64 200/200 (9.9e-16) and 500/500 (2.0e-15);
f32 200/200 (2.6e-7, was 1.2e-7: 1-target gates now compute in float) and 300/300 at ≤ 16 q (3.1e-7).

- 1-target gates now run at ~20 GB/s at 20 q, about what one core can stream from DRAM, and f32 is
  finally faster than f64. That path is memory-bound now.
- Random circuits only gain 1.3–1.9× because ~1/3 of their ops have 2–3 targets (swap, cswap,
  2–3 qubit `mat`), which still go through the generic gather/scatter loop. A 2-target register path
  would be the next CPU win, but the CPU code is the reference, not the product: the GPU work in
  M2/M3 is where speed matters.
- 24 q timings vary a lot run to run (49–86 ms/op for v2 f64): 256 MiB states, page faults/thermal.
  Use median of ≥ 5 at this size.

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
