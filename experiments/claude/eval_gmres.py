"""
eval_gmres.py
══════════════════════════════════════════════════════════════════════════════
Post-run evaluation for FGMRES v4 preconditioned vs unpreconditioned GMRES.

Reads results_v4.json from all 3 frequency pairs and produces:
  1. Iteration count comparison table (A vs D vs E, per pair)
  2. Speedup summary (geometric mean across problems)
  3. Convergence curve overlay (best problem from each pair)
  4. Per-preconditioner-call timing vs convergence wall-time tradeoff

Output:
  experiments/claude/results_transfer/eval_gmres/
    summary_table.txt
    iter_counts.png      — bar chart: iterations per variant per pair
    convergence.png      — residual curves (best problem per pair)
    speedup_summary.png  — speedup vs baseline

Usage:
  python experiments/claude/eval_gmres.py
══════════════════════════════════════════════════════════════════════════════
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[2]
RESDIR = ROOT / "experiments/claude/results_transfer"
OUTDIR = ROOT / "experiments/claude/results_transfer/eval_gmres"
OUTDIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("16_32",  16,  32),
    ("32_64",  32,  64),
    ("64_128", 64, 128),
]

VARIANT_LABELS = {
    "A": "A: Unpreconditioned GMRES",
    "D": "D: FGMRES + interior-restrict",
    "E": "E: FGMRES + full raw",
}
VARIANT_COLORS = {"A": "#4878CF", "D": "#2CA02C", "E": "#D62728"}


def load_pair(pair_id: str) -> list | None:
    p = RESDIR / f"precond_gmres_v4_{pair_id}" / "results_v4.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def geom_mean(values):
    return float(np.exp(np.mean(np.log(np.array(values, dtype=float) + 1e-9))))


def main():
    print("=" * 70)
    print("  GMRES v4 EVALUATION — preconditioned vs unpreconditioned")
    print("=" * 70)

    all_data = {}
    for pair_id, omega_l, omega_h in PAIRS:
        data = load_pair(pair_id)
        if data is None:
            print(f"  WARNING: {pair_id} results not found — skipping")
        else:
            all_data[pair_id] = data
            print(f"  {pair_id}: {len(data)} problems loaded")

    if not all_data:
        print("\nNo results found. Are the GMRES runs complete?")
        return

    # ── 1. Print summary table ─────────────────────────────────────────────
    print()
    print(f"  {'Pair':<8}  {'Prob':>5}  {'A iters':>8}  {'D iters':>8}  "
          f"{'E iters':>8}  {'SpeedD':>8}  {'SpeedE':>8}")
    print("  " + "-" * 64)

    for pair_id, omega_l, omega_h in PAIRS:
        data = all_data.get(pair_id)
        if data is None:
            continue
        su_Ds, su_Es = [], []
        for r in data:
            print(f"  {pair_id:<8}  {r['problem']:>5}  "
                  f"{r['A']['iters']:>8}  {r['D']['iters']:>8}  "
                  f"{r['E']['iters']:>8}  "
                  f"{r['speedup_D']:>7.2f}x  {r['speedup_E']:>7.2f}x")
            su_Ds.append(r["speedup_D"])
            su_Es.append(r["speedup_E"])
        gm_D = geom_mean(su_Ds)
        gm_E = geom_mean(su_Es)
        print(f"  {pair_id:<8}  {'GEOMEAN':>5}  {'':>8}  {'':>8}  {'':>8}  "
              f"{gm_D:>7.2f}x  {gm_E:>7.2f}x")
        print("  " + "-" * 64)

    # ── 2. Convergence curve overlay ───────────────────────────────────────
    n_pairs = len(all_data)
    if n_pairs == 0:
        return

    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]
    fig.suptitle(
        "FGMRES v4: Residual Convergence\n"
        "A=Unpreconditioned  D=Interior-restrict  E=Full-raw",
        fontsize=11,
    )

    for ax, (pair_id, omega_l, omega_h) in zip(axes, PAIRS):
        data = all_data.get(pair_id)
        if data is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        # Show best-converging problem for D (lowest D iters) as representative
        best_prob = min(data, key=lambda r: r["D"]["iters"])

        ax.set_title(f"ω: {omega_l}→{omega_h}  (Problem {best_prob['problem']}, "
                     f"{best_prob['n_sources']} src)", fontsize=9)
        ax.set_xlabel("GMRES iteration")
        ax.set_ylabel("Residual norm")

        for key in ["A", "D", "E"]:
            res = best_prob[key]["residuals"]
            ax.semilogy(res, color=VARIANT_COLORS[key], lw=1.8,
                        label=f"{VARIANT_LABELS[key]}  ({len(res)-1} iters)")

        ax.axhline(1e-4, color="gray", ls=":", lw=1, label="tol=1e-4")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    p = OUTDIR / "convergence.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"\n  Convergence plot → {p}")
    plt.close(fig)

    # ── 3. Bar chart: mean iters per pair per variant ─────────────────────
    pairs_present = [(pid, ol, oh) for pid, ol, oh in PAIRS if pid in all_data]
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.set_title("Mean iteration counts — A vs D vs E", fontsize=11)
    ax2.set_ylabel("Mean GMRES iterations")
    ax2.set_xlabel("Frequency pair")

    x      = np.arange(len(pairs_present))
    width  = 0.25
    labels = [f"ω {ol}→{oh}" for _, ol, oh in pairs_present]

    for i, key in enumerate(["A", "D", "E"]):
        means = []
        for pid, ol, oh in pairs_present:
            data  = all_data[pid]
            iters = [r[key]["iters"] for r in data]
            means.append(np.mean(iters))
        bars = ax2.bar(x + (i - 1) * width, means, width,
                       label=VARIANT_LABELS[key], color=VARIANT_COLORS[key],
                       alpha=0.85)
        for bar, val in zip(bars, means):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    p2 = OUTDIR / "iter_counts.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"  Bar chart → {p2}")
    plt.close(fig2)

    # ── 4. Speedup summary ─────────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.set_title("Speedup: FGMRES (D/E) vs Unpreconditioned (A)\n"
                  "(speedup>1 means preconditioner reduced iteration count)",
                  fontsize=10)
    ax3.set_ylabel("Speedup vs A (iter_A / iter_X)")
    ax3.set_xlabel("Frequency pair")
    ax3.axhline(1.0, color="gray", ls="--", lw=1, label="no speedup")

    for i, key in enumerate(["D", "E"]):
        speedups = []
        for pid, _, _ in pairs_present:
            data = all_data[pid]
            su   = [r[f"speedup_{key}"] for r in data]
            speedups.append(geom_mean(su))

        ax3.plot(labels, speedups, "o-", color=VARIANT_COLORS[key],
                 lw=2, ms=8, label=f"{key}: {VARIANT_LABELS[key]}")
        for lbl, su in zip(labels, speedups):
            ax3.annotate(f"{su:.2f}x", xy=(lbl, su),
                         xytext=(0, 8), textcoords="offset points",
                         ha="center", fontsize=8, color=VARIANT_COLORS[key])

    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = OUTDIR / "speedup_summary.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    print(f"  Speedup chart → {p3}")
    plt.close(fig3)

    # ── 5. Save summary text ───────────────────────────────────────────────
    summary_path = OUTDIR / "summary_table.txt"
    with open(summary_path, "w") as f:
        f.write("GMRES v4 Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Variants:\n")
        f.write("  A: Unpreconditioned GMRES (baseline)\n")
        f.write("  D: FGMRES + interior restriction (288×288)\n")
        f.write("  E: FGMRES + full raw residual (512×512)\n\n")
        f.write("Weights: VORONOI-LOOKaLIKE-1703, kernel=7, ~65% val RelL2\n")
        f.write("Problems: 5 per pair, seed=12345\n\n")

        for pair_id, omega_l, omega_h in PAIRS:
            data = all_data.get(pair_id)
            f.write(f"Pair: ω {omega_l}→{omega_h}\n")
            if data is None:
                f.write("  NOT FOUND\n\n")
                continue
            su_Ds, su_Es = [], []
            for r in data:
                f.write(
                    f"  Prob {r['problem']} ({r['n_sources']} src): "
                    f"A={r['A']['iters']}  D={r['D']['iters']}  E={r['E']['iters']}  "
                    f"su_D={r['speedup_D']:.2f}x  su_E={r['speedup_E']:.2f}x  "
                    f"t_A={r['A']['time_s']}s  t_D={r['D']['time_s']}s  t_E={r['E']['time_s']}s\n"
                )
                su_Ds.append(r["speedup_D"])
                su_Es.append(r["speedup_E"])
            f.write(f"  GEOMEAN speedup: D={geom_mean(su_Ds):.2f}x  E={geom_mean(su_Es):.2f}x\n\n")

    print(f"  Summary text → {summary_path}")
    print(f"\n  All outputs in: {OUTDIR}/")


if __name__ == "__main__":
    main()
