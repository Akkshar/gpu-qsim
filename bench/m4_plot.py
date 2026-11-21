"""Charts + markdown tables from bench/results/m4.json (written by bench/m4_bench.py).

    python bench/m4_plot.py [--in bench/results/m4.json] [--out docs/img]

Writes:
  docs/img/m4_scaling.png   time vs qubits, one panel per circuit family, every backend (log scale)
  docs/img/m4_at28.png      time at the largest size, per family and backend (log scale)
and prints markdown tables (medians + speedups) for BENCH.md / README.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Fixed entity -> colour + marker (colour follows the backend, never its rank). Palette validated with
# the dataviz validator (light surface): CVD/normal-vision separation pass; three colours are under
# 3:1 contrast, so every series also has its own marker shape and the data is in tables too.
STYLE = {
    "qsim-fused": ("#2a78d6", "o", "qsim, fused (K=4)"),
    "qsim": ("#eb6834", "s", "qsim, unfused"),
    "custatevec": ("#1baf7a", "^", "cuStateVec (per gate)"),
    "aer-gpu-cusv": ("#eda100", "D", "Aer GPU + cuStateVec + Aer fusion"),
    "aer-gpu": ("#e87ba4", "v", "Aer GPU, native kernels"),
    "aer-cpu": ("#008300", "P", "Aer CPU (20 threads)"),
}
ORDER = list(STYLE)
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TITLES = {"qft": "QFT", "layered": "Layered random (20 layers)", "qaoa": "QAOA ring (p=2)", "ghz": "GHZ",
          "random": "Random mix"}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load(path: Path):
    data = json.loads(path.read_text())
    meta = data["meta"]
    STYLE["qsim-fused"] = (STYLE["qsim-fused"][0], STYLE["qsim-fused"][1],
                           f"qsim, fused (K={meta['fuse']}{', ' + meta['fuse_mode'] if 'fuse_mode' in meta else ''})")
    table = {}
    for r in data["runs"]:
        table[(r["family"], r["qubits"], r["backend"])] = r["median_ms"]
    fams = [f for f in TITLES if any(k[0] == f for k in table)]
    ns = sorted({k[1] for k in table})
    return data, table, fams, ns


def plot_scaling(table, fams, ns, out: Path):
    cols = 2
    rows = (len(fams) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.2 * rows), facecolor=SURFACE, squeeze=False)
    for i, fam in enumerate(fams):
        ax = axes[i // cols][i % cols]
        style_axes(ax)
        for b in ORDER:
            pts = [(n, table[(fam, n, b)]) for n in ns if (fam, n, b) in table]
            if not pts:
                continue
            color, marker, label = STYLE[b]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker=marker, markersize=6,
                    linewidth=2, label=label, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_yscale("log")
        ax.set_xticks(ns)
        ax.set_title(TITLES[fam], color=INK, fontsize=11, loc="left")
        ax.set_xlabel("qubits", color=MUTED, fontsize=9)
        ax.set_ylabel("time (ms, log scale)", color=MUTED, fontsize=9)
    for j in range(len(fams), rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / (4.2 * rows)))
    fig.suptitle("Simulation time vs qubits, single precision, RTX 3070 Ti Laptop (lower is faster)",
                 color=INK, fontsize=12, x=0.01, ha="left", y=1.0 + 0.55 / (4.2 * rows))
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, 1.0 - 0.7 / (4.2 * rows)), ncol=3,
               frameon=False, fontsize=9, labelcolor=INK)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_at_max(table, fams, ns, out: Path):
    n = max(ns)
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.28 * len(fams) * len(ORDER)), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    y, yticks, ylabels = 0.0, [], []
    for fam in fams:
        start = y
        for b in ORDER:
            if (fam, n, b) not in table:
                continue
            color, marker, label = STYLE[b]
            ms = table[(fam, n, b)]
            # Dot plot, not bars: on a log axis bar length isn't proportional to the value.
            ax.plot([ms], [y], marker=marker, markersize=9, color=color, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, linestyle="none")
            txt = f"{ms:,.0f} ms" if ms >= 100 else (f"{ms:.1f} ms" if ms >= 1 else f"{ms:.2f} ms")
            ax.text(ms * 1.12, y, txt, va="center", fontsize=8, color=INK)
            y += 1
        yticks.append((start + y - 1) / 2)
        ylabels.append(TITLES[fam])
        y += 0.8
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, color=INK, fontsize=10)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel(f"time at {n} qubits (ms, log scale; further left is faster)", color=MUTED, fontsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.5, linestyle=":")
    shown = [b for b in ORDER if any((f, n, b) in table for f in fams)]
    missing = [STYLE[b][2] for b in ORDER if b not in shown and any(k[2] == b for k in table)]
    handles = [plt.Line2D([], [], marker=STYLE[b][1], color=STYLE[b][0], linestyle="none", markersize=8)
               for b in shown]
    ax.legend(handles, [STYLE[b][2] for b in shown], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
              frameon=False, fontsize=8, labelcolor=INK)
    note = f" — not run at this size: {', '.join(missing)}" if missing else ""
    fig.suptitle(f"Time at {n} qubits, single precision (further left is faster){note}", color=INK, fontsize=11,
                 x=0.01, ha="left")
    ax.set_xlim(right=ax.get_xlim()[1] * 2.5)  # room for the value labels
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def tables(data, table, fams, ns) -> str:
    out = [f"Medians in ms ({data['meta']['rounds']} interleaved rounds, {data['meta']['precision']}, "
           f"qsim-fused = --fuse {data['meta']['fuse']} {data['meta'].get('fuse_mode', 'inorder')}, "
           f"{data['meta']['date']})", ""]
    present = [b for b in ORDER if any(k[2] == b for k in table)]
    for fam in fams:
        out.append(f"**{TITLES[fam]}**")
        out.append("")
        out.append("| qubits | " + " | ".join(STYLE[b][2] for b in present) + " |")
        out.append("|---|" + "---|" * len(present))
        for n in ns:
            cells = [f"{table[(fam, n, b)]:,.1f}" if (fam, n, b) in table else "—" for b in present]
            out.append(f"| {n} | " + " | ".join(cells) + " |")
        out.append("")
    n = max(ns)
    out.append(f"**Speedup of qsim-fused at {n} qubits** (other backend's time / qsim-fused time; >1 = qsim faster)")
    out.append("")
    others = [b for b in present if b != "qsim-fused"]
    out.append("| family | " + " | ".join(STYLE[b][2] for b in others) + " |")
    out.append("|---|" + "---|" * len(others))
    for fam in fams:
        base = table.get((fam, n, "qsim-fused"))
        cells = [f"{table[(fam, n, b)] / base:.2f}×" if base and (fam, n, b) in table else "—" for b in others]
        out.append(f"| {TITLES[fam]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "bench" / "results" / "m4.json"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "img"))
    args = ap.parse_args()
    data, table, fams, ns = load(Path(args.inp))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_scaling(table, fams, ns, out / "m4_scaling.png")
    plot_at_max(table, fams, ns, out / "m4_at28.png")
    print(tables(data, table, fams, ns))
    print(f"\ncharts: {out / 'm4_scaling.png'}, {out / 'm4_at28.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
