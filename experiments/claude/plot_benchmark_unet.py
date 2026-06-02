"""
plot_benchmark_unet.py
━━━━━━━━━━━━━━━━━━━━━━
Publication-quality FGMRES benchmark figure comparing:
  A — Unpreconditioned FGMRES
  C — ILU(10)
  F — Neural UNet as direct solver (A^{-1})

Usage:
    python experiments/claude/plot_benchmark_unet.py \
        --results experiments/claude/results_transfer/benchmark_unet_omega32/results.json \
        --out     experiments/claude/results_transfer/benchmark_unet_omega32/benchmark_fig.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

STYLE = {
    "A": dict(color="#2166ac", lw=2.2, zorder=4),   # unpreconditioned — blue
    "C": dict(color="#d73027", lw=1.8, zorder=3),   # ILU             — red
    "F": dict(color="#f4a582", lw=1.5, zorder=2),   # UNet            — salmon
}
MARKER = {"A": "o", "C": "s", "F": "^"}


def load(path: Path) -> dict:
    txt = path.read_text()
    txt = txt.replace(": NaN", ": null")   # json chokes on bare NaN
    return json.loads(txt)


def make_figure(data: dict, out: Path):
    omega    = data["omega"]
    tol      = data["fgmres_tol"]
    problems = data["problems"]
    n_prob   = len(problems)

    fig = plt.figure(figsize=(7 * n_prob, 9))
    gs  = gridspec.GridSpec(2, n_prob, height_ratios=[3, 1.6],
                            hspace=0.5, wspace=0.38)

    for col, prob in enumerate(problems):
        ax_main = fig.add_subplot(gs[0, col])
        ax_zoom = fig.add_subplot(gs[1, col])
        n_src   = prob["n_src"]

        for key in ("A", "C", "F"):
            m   = prob.get(key, {})
            res = m.get("residuals") or []
            if not res:
                continue
            iters  = np.arange(len(res))
            conv   = m.get("converged", False)
            niters = m.get("iters", len(res) - 1)
            t_s    = m.get("time_s", 0)
            label  = (f"{m['label']}\n"
                      f"  {'OK' if conv else 'FAIL'} {niters} iter"
                      f"{'s' if niters != 1 else ''}  ({t_s:.1f}s)")

            ax_main.semilogy(iters, res, label=label, **STYLE[key])
            ax_zoom.semilogy(iters[:21], res[:21], **STYLE[key])

            # mark convergence point
            if conv and len(res) > 1:
                ax_main.semilogy(niters, res[niters], MARKER[key],
                                 color=STYLE[key]["color"], ms=8, zorder=6)

        # tolerance line
        for ax in (ax_main, ax_zoom):
            ax.axhline(tol, color="#444", ls="--", lw=1,
                       label=f"tol={tol:.0e}" if ax is ax_main else None)
            ax.set_ylabel(r"Relative residual $\|r_k\|/\|r_0\|$")
            ax.grid(True, which="both", ls=":", alpha=0.35)

        ax_main.set_title(
            f"Problem {col+1}  ({n_src} sources)\n"
            f"ω={omega:.0f},  512×512 grid",
            pad=8
        )
        ax_main.set_xlabel("FGMRES iteration")
        ax_main.legend(loc="upper right", framealpha=0.92,
                       handlelength=1.8, labelspacing=0.6)
        ax_main.set_xlim(left=0)

        ax_zoom.set_title("First 20 iterations (detail)")
        ax_zoom.set_xlabel("FGMRES iteration")
        ax_zoom.set_xlim(0, 20)

        # ── scientific interpretation box ─────────────────────────────────────
        # compute convergence rate for A (geometric mean per iteration)
        res_A = prob.get("A", {}).get("residuals", [])
        if len(res_A) >= 2:
            rate = (res_A[-1] / res_A[0]) ** (1 / (len(res_A) - 1))
            rate_txt = f"Rate(A) = {rate:.2e}/iter"
        else:
            rate_txt = ""

        res_F = prob.get("F", {}).get("residuals", [])
        if len(res_F) >= 2:
            rate_F = (res_F[-1] / res_F[0]) ** (1 / (len(res_F) - 1))
            slowdown = rate_F / rate if res_A and len(res_A) >= 2 else float("nan")
            rate_F_txt = f"Rate(F) = {rate_F:.4f}/iter"
        else:
            rate_F_txt = ""

        note = "\n".join(filter(None, [rate_txt, rate_F_txt]))
        if note:
            ax_main.text(0.98, 0.56, note,
                         transform=ax_main.transAxes,
                         fontsize=8.5, family="monospace",
                         va="top", ha="right",
                         bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                   ec="#cccccc", alpha=0.9))

    fig.suptitle(
        f"FGMRES benchmark  —  ω={omega:.0f}  N=512×512  tol={tol}",
        fontsize=13, y=1.01
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")

    # ── stdout summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"ω={omega:.0f}  tol={tol}")
    for prob in problems:
        print(f"\nProblem {prob['idx']+1} ({prob['n_src']} sources):")
        for key in ("A", "C", "F"):
            m = prob.get(key, {})
            if not m:
                continue
            conv   = "CONV" if m.get("converged") else "FAIL"
            niters = m.get("iters", "—")
            t_s    = m.get("time_s", 0)
            fres   = m.get("final_res")
            fres_s = f"{fres:.4e}" if fres is not None else "NaN"
            print(f"  {key}: {conv}  iters={str(niters):>5}  "
                  f"time={t_s:7.2f}s  final_res={fres_s}  | {m['label']}")
    print(f"{'='*60}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out",     required=True)
    args = ap.parse_args()
    data = load(Path(args.results))
    make_figure(data, Path(args.out))


if __name__ == "__main__":
    main()
