"""
eval_gmres_v5.py
══════════════════════════════════════════════════════════════════════════════
Post-run evaluation for the 5-way GMRES preconditioner benchmark (v5).

Reads results_v5.json from all 3 frequency pairs and produces:

  1. Summary table: mean iters and geomean speedup per variant per pair
  2. Convergence curve overlay (best problem per pair, all 5 variants)
  3. Bar chart: mean iters per variant across all pairs
  4. Wall-clock breakdown: setup time vs. per-problem solve time
  5. Scientific verdict for each question (printed + saved)

Output:
  experiments/claude/results_transfer/eval_gmres_v5/
    summary_table.txt     — human-readable summary
    convergence.png       — residual curves
    iter_counts.png       — grouped bar chart
    timing_breakdown.png  — setup vs. solve times
    speedup_vs_freq.png   — speedup as function of ω

Usage:
  python experiments/claude/eval_gmres_v5.py
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
OUTDIR = ROOT / "experiments/claude/results_transfer/eval_gmres_v5"
OUTDIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("16_32",  16,  32),
    ("32_64",  32,  64),
    ("64_128", 64, 128),
]

VARIANT_KEYS   = ["A", "B", "C", "D", "E"]
VARIANT_LABELS = {
    "A": "A: Unpreconditioned",
    "B": "B: Jacobi",
    "C": "C: ILU(0)",
    "D": "D: CSL (β=0.5)",
    "E": "E: Neural (interior)",
}
VARIANT_COLORS = {
    "A": "#4878CF",
    "B": "#999999",
    "C": "#FF7F0E",
    "D": "#9467BD",
    "E": "#2CA02C",
}

# ── helpers ─────────────────────────────────────────────────────────────────

def gm(values):
    return float(np.exp(np.mean(np.log(np.array(values, dtype=float) + 1e-9))))


def load_pair(pair_id: str) -> dict | None:
    p = RESDIR / f"precond_gmres_v5_{pair_id}" / "results_v5.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  GMRES v5 EVALUATION — 5-way preconditioner benchmark")
    print("=" * 72)

    all_data = {}
    for pair_id, omega_l, omega_h in PAIRS:
        data = load_pair(pair_id)
        if data is None:
            print(f"  WARNING: {pair_id} results_v5.json not found — skipping")
        else:
            all_data[pair_id] = data
            print(f"  {pair_id}: {len(data['problems'])} problems loaded")

    if not all_data:
        print("\nNo results found. Are the GMRES v5 runs complete?")
        return

    # ── 1. Summary table ──────────────────────────────────────────────────
    print()
    print(f"  {'Pair':<8}  {'Var':>3}  {'Mean iters':>12}  {'Geomean speedup':>16}  "
          f"{'Setup (s)':>10}  {'Call (ms)':>10}")
    print("  " + "-" * 68)

    verdict = {}
    for pair_id, omega_l, omega_h in PAIRS:
        data = all_data.get(pair_id)
        if data is None:
            continue
        probs = data["problems"]
        setup = data.get("setup_times", {})
        calls = data.get("avg_call_times_ms", {})

        iters_A = [r["A"]["iters"] for r in probs]

        for key in VARIANT_KEYS:
            iters_X = [r[key]["iters"] for r in probs]
            sp_key  = f"speedup_{key}"
            speedups = [r[sp_key] for r in probs if sp_key in r]
            if key == "A":
                speedups = [1.0] * len(probs)
            gm_su    = gm(speedups)
            setup_s  = setup.get(key, setup.get(f"{key}_lu", 0.0))
            call_ms  = calls.get(key, 0.0)
            print(f"  {pair_id:<8}  {key:>3}  {np.mean(iters_X):>12.1f}  "
                  f"{gm_su:>16.2f}x  {setup_s:>10.1f}  {call_ms:>10.1f}")

        print("  " + "·" * 68)
        verdict[pair_id] = {
            "iters_A": float(np.mean(iters_A)),
            "iters": {
                k: float(np.mean([r[k]["iters"] for r in probs]))
                for k in VARIANT_KEYS
            },
            "speedups": {
                k: gm([r.get(f"speedup_{k}", 1.0) for r in probs])
                for k in ["B", "C", "D", "E"]
            },
        }

    # ── 2. Scientific verdict ─────────────────────────────────────────────
    print()
    print("  SCIENTIFIC VERDICT")
    print("  " + "=" * 68)

    lines = []
    for pair_id, omega_l, omega_h in PAIRS:
        if pair_id not in verdict:
            continue
        v = verdict[pair_id]
        best_classical_key = min(["B","C","D"], key=lambda k: v["iters"][k])
        best_classical_su  = v["speedups"][best_classical_key]
        neural_su          = v["speedups"]["E"]
        iters_A = v["iters_A"]
        iters_E = v["iters"]["E"]
        iters_best = v["iters"][best_classical_key]

        q1 = "YES" if neural_su > best_classical_su else "NO"
        q4 = f"Neural: {iters_E:.0f} vs {best_classical_key}: {iters_best:.0f}"
        line = (
            f"  ω {omega_l}→{omega_h}:  "
            f"A={iters_A:.0f}  "
            f"Best-classical={best_classical_key}({iters_best:.0f}, {best_classical_su:.2f}x)  "
            f"E={iters_E:.0f}({neural_su:.2f}x)  "
            f"[Q1 neural>classical: {q1}]"
        )
        print(line)
        lines.append(line)

    # ── 3. Convergence plot ───────────────────────────────────────────────
    pairs_present = [(pid, ol, oh) for pid, ol, oh in PAIRS if pid in all_data]
    n_pairs = len(pairs_present)

    fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]
    fig.suptitle(
        "GMRES v5: Residual Convergence (best-converging problem per pair)\n"
        "A=Unprecond  B=Jacobi  C=ILU(0)  D=CSL(β=0.5)  E=Neural",
        fontsize=11,
    )

    for ax, (pair_id, omega_l, omega_h) in zip(axes, pairs_present):
        data  = all_data[pair_id]
        probs = data["problems"]
        # show the problem where D (CSL) converges fastest vs A
        best_prob = min(probs, key=lambda r: r["D"]["iters"])

        ax.set_title(
            f"ω: {omega_l}→{omega_h}  (Prob {best_prob['problem']}, "
            f"{best_prob['n_sources']} src)",
            fontsize=9,
        )
        ax.set_xlabel("GMRES iteration")
        if ax is axes[0]:
            ax.set_ylabel("Residual norm")

        for key in VARIANT_KEYS:
            res = best_prob[key]["residuals"]
            ax.semilogy(res, color=VARIANT_COLORS[key], lw=1.8,
                        label=f"{VARIANT_LABELS[key]}  ({len(res)-1})")

        ax.axhline(1e-4, color="black", ls=":", lw=1, label="tol=1e-4")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    p = OUTDIR / "convergence.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"\n  Convergence → {p}")
    plt.close(fig)

    # ── 4. Grouped bar chart: mean iters ──────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.set_title("Mean iteration counts across problems — all pairs", fontsize=11)
    ax2.set_ylabel("Mean iterations to convergence")
    ax2.set_xlabel("Frequency pair")

    labels     = [f"ω {ol}→{oh}" for _, ol, oh in pairs_present]
    x          = np.arange(len(labels))
    n_variants = len(VARIANT_KEYS)
    width      = 0.15

    for i, key in enumerate(VARIANT_KEYS):
        means = []
        for pair_id, _, _ in pairs_present:
            data  = all_data[pair_id]
            probs = data["problems"]
            means.append(np.mean([r[key]["iters"] for r in probs]))
        offset = (i - n_variants / 2 + 0.5) * width
        bars = ax2.bar(x + offset, means, width,
                       label=VARIANT_LABELS[key],
                       color=VARIANT_COLORS[key], alpha=0.85)
        for bar, val in zip(bars, means):
            if val < 1000:
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 5,
                         f"{val:.0f}", ha="center", va="bottom",
                         fontsize=6, rotation=45)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p2 = OUTDIR / "iter_counts.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    print(f"  Iter counts → {p2}")
    plt.close(fig2)

    # ── 5. Speedup vs frequency ───────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.set_title(
        "Geomean speedup vs. Unpreconditioned GMRES — as ω increases\n"
        "(Does preconditioning help more on harder systems?)",
        fontsize=10,
    )
    ax3.set_ylabel("Speedup (iters_A / iters_X)")
    ax3.axhline(1.0, color="gray", ls="--", lw=1)

    pair_labels = [f"ω {ol}→{oh}" for _, ol, oh in pairs_present]
    for key in ["B", "C", "D", "E"]:
        speedups = []
        for pair_id, _, _ in pairs_present:
            data  = all_data[pair_id]
            probs = data["problems"]
            sp_k  = f"speedup_{key}"
            su    = gm([r[sp_k] for r in probs if sp_k in r]) if any(sp_k in r for r in probs) else 1.0
            speedups.append(su)
        ax3.plot(pair_labels, speedups, "o-",
                 color=VARIANT_COLORS[key], lw=2, ms=8,
                 label=VARIANT_LABELS[key])
        for lbl, su_val in zip(pair_labels, speedups):
            ax3.annotate(f"{su_val:.2f}x",
                         xy=(lbl, su_val),
                         xytext=(0, 8), textcoords="offset points",
                         ha="center", fontsize=8,
                         color=VARIANT_COLORS[key])

    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = OUTDIR / "speedup_vs_freq.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    print(f"  Speedup vs freq → {p3}")
    plt.close(fig3)

    # ── 6. Timing breakdown ───────────────────────────────────────────────
    fig4, axes4 = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
    if n_pairs == 1:
        axes4 = [axes4]
    fig4.suptitle("Wall-clock breakdown: setup time vs. per-problem solve time",
                  fontsize=11)

    for ax, (pair_id, omega_l, omega_h) in zip(axes4, pairs_present):
        data  = all_data[pair_id]
        setup = data.get("setup_times", {})
        probs = data["problems"]

        keys     = ["B", "C", "D", "E"]
        s_times  = [setup.get(k, 0.0) for k in keys]
        # mean solve time from json
        sv_times = [np.mean([r[k]["time_s"] for r in probs]) for k in keys]

        bar_lbl  = [VARIANT_LABELS[k].split(":")[1].strip() for k in keys]
        bcolors  = [VARIANT_COLORS[k] for k in keys]
        xp       = np.arange(len(keys))
        width_t  = 0.35

        ax.bar(xp - width_t/2, s_times, width_t,
               label="Setup (once)", color=bcolors, alpha=0.5, hatch="//")
        ax.bar(xp + width_t/2, sv_times, width_t,
               label="Mean solve/problem", color=bcolors, alpha=0.9)
        ax.set_title(f"ω {omega_l}→{omega_h}", fontsize=9)
        ax.set_xticks(xp)
        ax.set_xticklabels(bar_lbl, fontsize=8)
        ax.set_ylabel("Time (s)")
        if ax is axes4[0]:
            ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        # Add A baseline solve time for reference
        a_mean = np.mean([r["A"]["time_s"] for r in probs])
        ax.axhline(a_mean, color=VARIANT_COLORS["A"], ls="--", lw=1.5,
                   label=f"A solve: {a_mean:.0f}s")

    plt.tight_layout()
    p4 = OUTDIR / "timing_breakdown.png"
    fig4.savefig(p4, dpi=150, bbox_inches="tight")
    print(f"  Timing → {p4}")
    plt.close(fig4)

    # ── 7. Save summary text ──────────────────────────────────────────────
    summary_path = OUTDIR / "summary_table.txt"
    with open(summary_path, "w") as f:
        f.write("GMRES v5 Evaluation — 5-way Preconditioner Benchmark\n")
        f.write("=" * 72 + "\n\n")
        f.write("Variants:\n")
        for k, v in VARIANT_LABELS.items():
            f.write(f"  {v}\n")
        f.write("\n")

        for pair_id, omega_l, omega_h in PAIRS:
            data = all_data.get(pair_id)
            f.write(f"Frequency pair: ω {omega_l}→{omega_h}\n")
            if data is None:
                f.write("  NOT FOUND\n\n")
                continue

            probs  = data["problems"]
            setup  = data.get("setup_times", {})
            calls  = data.get("avg_call_times_ms", {})

            f.write(f"  {'Var':<4}  {'Mean iters':>12}  {'Speedup':>10}  "
                    f"{'Setup (s)':>10}  {'Call (ms)':>10}\n")
            f.write("  " + "-" * 52 + "\n")
            iters_A = float(np.mean([r["A"]["iters"] for r in probs]))
            for key in VARIANT_KEYS:
                iters_X = float(np.mean([r[key]["iters"] for r in probs]))
                sp_k = f"speedup_{key}"
                su   = gm([r[sp_k] for r in probs if sp_k in r]) if key != "A" else 1.0
                s_s  = setup.get(key, setup.get(f"{key}_lu", 0.0))
                c_ms = calls.get(key, 0.0)
                f.write(f"  {key:<4}  {iters_X:>12.1f}  {su:>10.2f}x  "
                        f"{s_s:>10.1f}  {c_ms:>10.1f}\n")
            f.write("\n")

        f.write("Scientific verdict:\n")
        for line in lines:
            f.write(line + "\n")

    print(f"  Summary text → {summary_path}")
    print(f"\n  All outputs in: {OUTDIR}/")


if __name__ == "__main__":
    main()
