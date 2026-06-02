"""
make_slides_apr22.py
────────────────────────────────────────────────────────────────────────────
Weekly FreqTransfer update slides — April 22, 2026
Generates a PDF slide deck from existing results.

Usage:
    source .venv/bin/activate
    python experiments/claude/make_slides_apr22.py
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "claude" / "results_transfer"
OUT     = ROOT / "experiments" / "claude" / "slides_apr22_2026.pdf"

# ── colour palette ────────────────────────────────────────────────────────────
BLUE   = "#2E6DA4"
ORANGE = "#E07B39"
GREEN  = "#3A9E5F"
RED    = "#C0392B"
GREY   = "#7F8C8D"
DARK   = "#2C3E50"
LIGHT  = "#ECF0F1"
GOLD   = "#F39C12"

# Slide dimensions (16:9, inches)
W, H = 13.3, 7.5

# ── helpers ───────────────────────────────────────────────────────────────────
def new_slide(title: str = "", subtitle: str = "") -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    if title:
        fig.text(0.5, 0.95, title, ha="center", va="top",
                 fontsize=20, fontweight="bold", color=DARK)
    if subtitle:
        fig.text(0.5, 0.89, subtitle, ha="center", va="top",
                 fontsize=12, color=GREY, style="italic")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    # horizontal rule under title
    if title:
        ax.axhline(0.875, xmin=0.05, xmax=0.95, color=BLUE, lw=1.5)
    return fig, ax


def status_badge(ax, x, y, text, color, fontsize=9):
    ax.text(x, y, f"  {text}  ", ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=color, edgecolor="none"))


def add_footer(fig, text="FreqTransfer  ·  April 22, 2026  ·  Kees & Fenna"):
    fig.text(0.5, 0.015, text, ha="center", va="bottom",
             fontsize=8, color=GREY)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 1 — Title
# ─────────────────────────────────────────────────────────────────────────────
def slide_title(pdf):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(DARK)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # accent bar
    ax.axhline(0.52, xmin=0.1, xmax=0.9, color=BLUE, lw=4)

    fig.text(0.5, 0.72, "FreqTransfer — Weekly Update",
             ha="center", va="center", fontsize=30, fontweight="bold",
             color="white")
    fig.text(0.5, 0.60, "April 22, 2026",
             ha="center", va="center", fontsize=20, color=GOLD)
    fig.text(0.5, 0.40, "Stage I: Warm-Start Results",
             ha="center", va="center", fontsize=16, color=LIGHT,
             style="italic")
    fig.text(0.5, 0.31, "Helmholtz frequency-transfer as an FGMRES initial guess",
             ha="center", va="center", fontsize=12, color=GREY)
    fig.text(0.5, 0.06, "Fenna Kiewiet  ·  TU Delft  ·  Thesis update",
             ha="center", va="bottom", fontsize=10, color=GREY)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 2 — Last week's commitments
# ─────────────────────────────────────────────────────────────────────────────
def slide_commitments(pdf):
    fig, ax = new_slide("Status: Commitments from April 15")
    add_footer(fig)

    commitments = [
        ("Run warm-start experiment\n(CSL-FGMRES, zero vs network start)",
         "DONE", GREEN,
         "Results obtained for ω=32, 64, 128"),
        ("Produce qualitative visual\n(solution panels at iter 1, 2, 3)",
         "DONE", GREEN,
         "snapshots.png generated for ω=32"),
        ("Move training to ORCD\n(resolve SLURM persistence)",
         "DONE", GREEN,
         "precond_v3 configs rewired, 12h Slurm jobs"),
        ("Fix dataset paths to ORCD pool\n(N=9600 for all three pairs)",
         "DONE", GREEN,
         "Symlinks resolved to /orcd/pool/006/fkiewiet/"),
        ("Check CSL β value in benchmark code",
         "DONE", GREEN,
         "Confirmed β = 0.5  (per results.json)"),
        ("Why emergent structures?\n(Fourier analysis)",
         "PENDING", ORANGE,
         "Deferred — needs dedicated analysis"),
        ("Split method chapters for writing",
         "PENDING", ORANGE,
         "Warm-start + preconditioning currently combined"),
    ]

    y_start = 0.82
    dy = 0.103
    col_status = 0.62
    col_note   = 0.67
    pad_l = 0.07

    ax.text(pad_l,        y_start + 0.025, "Commitment",         fontsize=10, fontweight="bold", color=DARK)
    ax.text(col_status,   y_start + 0.025, "Status",             fontsize=10, fontweight="bold", color=DARK, ha="center")
    ax.text(col_note,     y_start + 0.025, "Delivered",          fontsize=10, fontweight="bold", color=DARK)
    ax.axhline(y_start - 0.005, xmin=0.05, xmax=0.95, color="#BDC3C7", lw=0.8)

    for i, (commitment, status, color, note) in enumerate(commitments):
        y = y_start - (i + 1) * dy
        # alternating row
        if i % 2 == 0:
            rect = FancyBboxPatch((0.055, y - 0.035), 0.89, dy - 0.005,
                                  boxstyle="round,pad=0.005",
                                  facecolor="#F8F9FA", edgecolor="none", zorder=0)
            ax.add_patch(rect)
        ax.text(pad_l, y + 0.005, commitment, fontsize=8.5, color=DARK, va="center",
                linespacing=1.35)
        status_badge(ax, col_status, y + 0.005, status, color, fontsize=8)
        ax.text(col_note, y + 0.005, note, fontsize=8, color=GREY, va="center")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 3 — Warm-start: the key metric explained
# ─────────────────────────────────────────────────────────────────────────────
def slide_metric_explained(pdf):
    fig, ax = new_slide("Warm-Start Quality — What We Measure",
                        "r₀/‖b‖ is the only pre-solver metric; goal < 1.0 to benefit FGMRES")
    add_footer(fig)

    # Left: diagram
    ax_l = fig.add_axes([0.05, 0.14, 0.43, 0.65])
    ax_l.axis("off")
    ax_l.set_xlim(0, 1); ax_l.set_ylim(0, 1)

    # Zero start box
    ax_l.add_patch(FancyBboxPatch((0.0, 0.55), 0.45, 0.28,
                                  boxstyle="round,pad=0.02",
                                  facecolor="#D6EAF8", edgecolor=BLUE, lw=1.5))
    ax_l.text(0.225, 0.69, "Zero start  $x_0 = 0$",
              ha="center", va="center", fontsize=11, fontweight="bold", color=BLUE)
    ax_l.text(0.225, 0.59, "$r_0 = b - A \\cdot 0 = b$\n$r_0/\\|b\\| = 1.0$",
              ha="center", va="center", fontsize=9.5, color=BLUE)

    # Warm start box
    ax_l.add_patch(FancyBboxPatch((0.54, 0.55), 0.45, 0.28,
                                  boxstyle="round,pad=0.02",
                                  facecolor="#FAD7A0", edgecolor=ORANGE, lw=1.5))
    ax_l.text(0.765, 0.69, "Warm start  $x_0 = T(u_L)$",
              ha="center", va="center", fontsize=11, fontweight="bold", color=ORANGE)
    ax_l.text(0.765, 0.59, "$r_0 = b - A \\cdot T(u_L)$\n$r_0/\\|b\\| = ?$",
              ha="center", va="center", fontsize=9.5, color=ORANGE)

    # Arrow down to FGMRES
    ax_l.annotate("", xy=(0.5, 0.28), xytext=(0.225, 0.55),
                  arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.5))
    ax_l.annotate("", xy=(0.5, 0.28), xytext=(0.765, 0.55),
                  arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.5))

    ax_l.add_patch(FancyBboxPatch((0.2, 0.10), 0.6, 0.20,
                                  boxstyle="round,pad=0.02",
                                  facecolor="#D5F5E3", edgecolor=GREEN, lw=1.5))
    ax_l.text(0.5, 0.20, "CSL-preconditioned FGMRES\n(same preconditioner, same tolerance)",
              ha="center", va="center", fontsize=9.5, color=GREEN)

    ax_l.text(0.5, 0.03, "Gain = fewer iterations to converge",
              ha="center", va="center", fontsize=9, color=GREY, style="italic")

    # Right: key numbers table
    ax_r = fig.add_axes([0.52, 0.14, 0.44, 0.65])
    ax_r.axis("off")
    ax_r.set_xlim(0, 1); ax_r.set_ylim(0, 1)

    ax_r.text(0.5, 0.97, "Current warm-start quality  $r_0/\\|b\\|$",
              ha="center", va="top", fontsize=12, fontweight="bold", color=DARK)

    # Table
    headers = ["ω",  "Model  RelL2",  "r₀/‖b‖ (W)",  "Verdict"]
    rows = [
        ["32",  "39.2% (ep 44)",  "~4.2×",   "FAIL  worse than zero"],
        ["64",  "38.8%",          "~1.2×",   "FAIL  slightly worse"],
        ["128", "old ckpt (~bad)","~36 000×", "FAIL  catastrophic"],
    ]
    colors_v = [RED, RED, RED]

    ys = [0.76, 0.62, 0.49, 0.36]
    xs = [0.01, 0.22, 0.46, 0.68]
    col_w = [0.18, 0.24, 0.23, 0.33]

    for j, (h, w) in enumerate(zip(headers, col_w)):
        ax_r.text(xs[j] + w/2, ys[0], h, ha="center", va="center",
                  fontsize=9.5, fontweight="bold", color=DARK)
    ax_r.axhline(ys[0] - 0.06, xmin=0.0, xmax=1.0, color="#BDC3C7", lw=0.8)

    for i, (row, vc) in enumerate(zip(rows, colors_v)):
        y = ys[i + 1]
        bg = "#F8F9FA" if i % 2 == 0 else "white"
        ax_r.add_patch(FancyBboxPatch((0, y - 0.065), 1.0, 0.13,
                                      boxstyle="round,pad=0.005",
                                      facecolor=bg, edgecolor="none", zorder=0))
        for j, (cell, w) in enumerate(zip(row, col_w)):
            c = vc if j == 3 else DARK
            fw = "bold" if j == 0 else "normal"
            ax_r.text(xs[j] + w/2, y, cell, ha="center", va="center",
                      fontsize=8.5, color=c, fontweight=fw)

    # Target box
    ax_r.add_patch(FancyBboxPatch((0.0, 0.02), 1.0, 0.22,
                                  boxstyle="round,pad=0.02",
                                  facecolor="#EAFAF1", edgecolor=GREEN, lw=1.5))
    ax_r.text(0.5, 0.17, "Target for warm-start benefit:",
              ha="center", va="center", fontsize=9, color=DARK)
    ax_r.text(0.5, 0.08, "$r_0/\\|b\\| < 1.0$   →   needs RelL2 $\\lesssim 20\\%$",
              ha="center", va="center", fontsize=10, fontweight="bold", color=GREEN)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 4 — Convergence curves (from saved results.json)
# ─────────────────────────────────────────────────────────────────────────────
def slide_convergence_curves(pdf):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.96, "FGMRES Convergence: Zero vs Warm Start",
             ha="center", va="top", fontsize=20, fontweight="bold", color=DARK)
    fig.text(0.5, 0.91, "CSL β=0.5  ·  Both methods converge in 3 iterations regardless of starting point",
             ha="center", va="top", fontsize=11, color=GREY, style="italic")
    fig.text(0.5, 0.015, "FreqTransfer  ·  April 22, 2026  ·  Kees & Fenna",
             ha="center", va="bottom", fontsize=8, color=GREY)

    omegas = [32, 64, 128]
    data = {}
    for om in omegas:
        p = RESULTS / f"warmstart_omega{om}" / "results.json"
        if p.exists():
            data[om] = json.loads(p.read_text())

    n_om = len(omegas)
    # Show first problem only for each omega (clean slide)
    gs = gridspec.GridSpec(1, n_om, figure=fig,
                           left=0.06, right=0.97, top=0.85, bottom=0.10,
                           wspace=0.30)

    for col, om in enumerate(omegas):
        ax = fig.add_subplot(gs[col])
        if om not in data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        prob = data[om]["problems"][0]
        norm_b = prob["Z"]["residuals"][0]  # first residual ≈ ‖b‖ for zero start

        resZ = [r / norm_b for r in prob["Z"]["residuals"]]
        resW = [r / norm_b for r in prob["W"]["residuals"]]
        xZ = list(range(len(resZ)))
        xW = list(range(len(resW)))

        ax.semilogy(xZ, resZ, color=BLUE,   lw=2.5, marker="o", markersize=7,
                    label="Zero start  $x_0=0$")
        ax.semilogy(xW, resW, color=ORANGE, lw=2.5, marker="s", markersize=7,
                    ls="--", label=f"Warm start  $T(u_{{\\omega/2}})$")
        ax.axhline(1e-8, color=GREY, ls=":", lw=1.2, label="tol $10^{-8}$")

        ax.set_xlabel("FGMRES iteration $k$", fontsize=10)
        ax.set_ylabel("$\\|r_k\\| / \\|b\\|$", fontsize=10)
        ax.set_title(f"$\\omega = {om}$", fontsize=13, fontweight="bold", color=DARK)
        ax.grid(True, which="both", alpha=0.2)
        ax.tick_params(labelsize=9)

        # Annotate start residuals
        if len(resW) > 0:
            ax.annotate(f"$r_0/\\|b\\|$ = {resW[0]:.2f}",
                        xy=(0, resW[0]), xytext=(0.5, resW[0] * 2.5),
                        fontsize=8, color=ORANGE,
                        arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8))

        # verdict box
        wq = prob.get("warm_prediction_quality", "?")
        verdict_color = RED
        ax.text(0.97, 0.97,
                f"Warm quality: {float(wq):.1f}×\nZ conv: {prob['Z']['conv_iter']} iters\nW conv: {prob['W']['conv_iter']} iters",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=verdict_color,
                bbox=dict(facecolor="white", edgecolor=verdict_color,
                          boxstyle="round,pad=0.3", alpha=0.9))

        ax.legend(fontsize=8, loc="upper right",
                  bbox_to_anchor=(1.0, 0.62))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 5 — Field snapshots at k=0 (embed existing PNG)
# ─────────────────────────────────────────────────────────────────────────────
def slide_field_snapshot(pdf):
    snap_path = RESULTS / "warmstart_report" / "fig3_field_snapshot.png"
    if not snap_path.exists():
        return

    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.97, "Initial Guess Quality — ω=32",
             ha="center", va="top", fontsize=20, fontweight="bold", color=DARK)
    fig.text(0.5, 0.92,
             "Network prediction vs ground truth at k=0  ·  "
             "Current model RelL2 ≈ 39%  →  large initial error",
             ha="center", va="top", fontsize=11, color=GREY, style="italic")
    fig.text(0.5, 0.015, "FreqTransfer  ·  April 22, 2026  ·  Kees & Fenna",
             ha="center", va="bottom", fontsize=8, color=GREY)

    ax = fig.add_axes([0.04, 0.10, 0.92, 0.78])
    img = plt.imread(str(snap_path))
    ax.imshow(img, aspect="auto")
    ax.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 6 — Diagnosis: why warm-start failed
# ─────────────────────────────────────────────────────────────────────────────
def slide_diagnosis(pdf):
    fig, ax = new_slide("Diagnosis — Why Warm-Start Did Not Help (Yet)",
                        "Three independent failure modes")
    add_footer(fig)

    panels = [
        (ORANGE, "1  Regime too easy (ω=32, ω=64)",
         [
             "Unpreconditioned FGMRES converges in 3 iters — even CSL is overkill",
             "Warm-start can't improve on what's already optimal",
             "Meaningful test is ω=128 where iteration count is large",
         ]),
        (RED, "2  Model quality insufficient",
         [
             "Need  r₀/‖b‖ < 1.0  →  RelL2 ≲ 20%  for warm-start to reduce residual",
             "Current best (ep 44):  39.2% RelL2  →  r₀/‖b‖ ≈ 4× (worse than zero!)",
             "Ep 44 is the checkpoint that matters — val diverges after ep 15",
         ]),
        (GREY, "3  ω=128 used wrong checkpoint",
         [
             "Checkpoint: precond_unet_v2_omega128  (old architecture)",
             "Prediction quality ~36 000×  →  catastrophic warm start",
             "Need proper TransferUNet checkpoint for this pair",
         ]),
    ]

    y_tops = [0.82, 0.57, 0.32]
    for (col, title, bullets), yt in zip(panels, y_tops):
        ax.add_patch(FancyBboxPatch((0.05, yt - 0.19), 0.90, 0.22,
                                    boxstyle="round,pad=0.01",
                                    facecolor="white",
                                    edgecolor=col, lw=2.0))
        ax.text(0.08, yt + 0.005, title, fontsize=12, fontweight="bold", color=col)
        for k, b in enumerate(bullets):
            ax.text(0.10, yt - 0.05 - k * 0.055, f"• {b}",
                    fontsize=9.5, color=DARK)

    # Bottom summary box
    ax.add_patch(FancyBboxPatch((0.05, 0.02), 0.90, 0.09,
                                boxstyle="round,pad=0.01",
                                facecolor="#EAF2FF", edgecolor=BLUE, lw=1.5))
    ax.text(0.5, 0.065, "All three issues are fixable — a better model at ω=128 is the critical next step",
            ha="center", va="center", fontsize=11, fontweight="bold", color=BLUE)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 7 — New training: TransferUNet
# ─────────────────────────────────────────────────────────────────────────────
def slide_new_training(pdf):
    fig, ax = new_slide("New Training — TransferUNet (precond_v3)",
                        "Architecture upgrade + N=9600  ·  Running on wave7b")
    add_footer(fig)

    # Left: architecture
    ax.text(0.08, 0.78, "Architecture comparison", fontsize=12,
            fontweight="bold", color=DARK)

    rows_arch = [
        ("Model",        "Old DilatedCNN",       "New TransferUNet"),
        ("Params",       "~1.5M",                "9.77M"),
        ("Skip conns",   "No",                   "Yes (UNet)"),
        ("Input ch",     "29 (Fourier feats)",   "6 (clean, no Fourier)"),
        ("Norm",         "InstanceNorm",         "InstanceNorm"),
        ("Activation",   "ReLU",                 "GELU"),
        ("Bottleneck",   "—",                    "32×32 spatial"),
        ("Best val RelL2","39% (16→32)",         "0.43% T_down* (running)"),
    ]

    ys = np.linspace(0.70, 0.15, len(rows_arch))
    for i, (label, old, new) in enumerate(rows_arch):
        y = ys[i]
        bg = "#F8F9FA" if i % 2 == 0 else "white"
        ax.add_patch(FancyBboxPatch((0.05, y - 0.025), 0.52, 0.052,
                                    boxstyle="round,pad=0.003",
                                    facecolor=bg, edgecolor="none"))
        fw = "bold" if i == 0 else "normal"
        ax.text(0.075, y,  label, fontsize=9, color=DARK, va="center", fontweight=fw)
        ax.text(0.255, y,  old,   fontsize=9, color=GREY, va="center")
        ax.text(0.400, y,  new,   fontsize=9,
                color=GREEN if i == len(rows_arch)-1 else DARK,
                fontweight="bold" if i == len(rows_arch)-1 else "normal",
                va="center")
    ax.text(0.255, 0.72, "Before",  fontsize=9, fontweight="bold", color=DARK)
    ax.text(0.400, 0.72, "Now",     fontsize=9, fontweight="bold", color=GREEN)

    ax.text(0.25, 0.08,
            "* T_down 32→16 running on wave7b (epoch 167)  ·  T_up starting next",
            fontsize=8, color=GREY, style="italic")

    # Right: training curve for T_down
    log_path = Path("/tmp/fkiewiet/precond_v3_N9600/pair_16_32/T_down/log.csv")
    if log_path.exists():
        data = np.genfromtxt(log_path, delimiter=",", skip_header=1)
        if data.ndim == 2 and len(data) > 5:
            epochs     = data[:, 0]
            train_loss = data[:, 1]
            val_loss   = data[:, 2]
            best_epoch = int(data[-1, 5])
            best_val   = float(data[-1, 4])

            ax_r = fig.add_axes([0.63, 0.14, 0.33, 0.60])
            ax_r.semilogy(epochs, train_loss, color=BLUE,   lw=2, label="Train")
            ax_r.semilogy(epochs, val_loss,   color=ORANGE, lw=2, label="Val")
            ax_r.axvline(best_epoch, color=GREEN, ls="--", lw=1.5,
                         label=f"Best ep {best_epoch} ({best_val:.4f})")
            ax_r.set_xlabel("Epoch", fontsize=9)
            ax_r.set_ylabel("Interior RelL2 loss", fontsize=9)
            ax_r.set_title("T_down 32→16  ·  N=9600\n(precond_v3 TransferUNet)",
                           fontsize=9)
            ax_r.legend(fontsize=8, loc="upper right")
            ax_r.grid(True, which="both", alpha=0.2)
            ax_r.tick_params(labelsize=8)

            # annotate best
            ax_r.text(0.97, 0.15,
                      f"Best val: {best_val:.4f}\n({best_val*100:.2f}%)",
                      transform=ax_r.transAxes, ha="right", va="bottom",
                      fontsize=9, color=GREEN, fontweight="bold",
                      bbox=dict(facecolor="white", edgecolor=GREEN,
                                boxstyle="round,pad=0.3", alpha=0.9))

            fig.text(0.63 + 0.165, 0.76,
                     f"val loss ↓ 0.43%\nvs. 39% before",
                     ha="center", va="bottom", fontsize=9,
                     color=GREEN, fontweight="bold")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 8 — Model quality threshold
# ─────────────────────────────────────────────────────────────────────────────
def slide_threshold(pdf):
    fig, ax = new_slide("The Threshold for Warm-Start to Work",
                        "Relating model RelL2 to initial residual reduction")
    add_footer(fig)

    # Simple illustrative plot: r0/‖b‖ vs RelL2
    ax_p = fig.add_axes([0.07, 0.14, 0.45, 0.62])
    rrl2 = np.linspace(0, 1.2, 200)
    # rough model: r0/||b|| ≈ RelL2 (linear region near 0, then constant)
    # For a linear operator A, if x0 ≈ A^-1 b with RelL2 error ε,
    # then ‖Ax0 - b‖/‖b‖ ≈ ‖A‖‖A^-1‖ * ε  (condition number scaling)
    # but roughly it's proportional to ε
    cond_est = 4.0   # rough effective condition
    r0_est = cond_est * rrl2

    ax_p.axhline(1.0, color="#BDC3C7", ls="--", lw=1.2)
    ax_p.fill_between(rrl2, 0, 1.0, alpha=0.12, color=GREEN,
                      label="Warm-start helps (r₀/‖b‖ < 1)")
    ax_p.fill_between(rrl2, 1.0, np.maximum(r0_est, 1.0), alpha=0.12, color=RED,
                      label="Warm-start hurts (r₀/‖b‖ > 1)")
    ax_p.plot(rrl2, r0_est, color=DARK, lw=2, label="Rough estimate")

    # Mark current models
    models = [
        (0.392, 4.2,  "ω=32\n(current)", RED),
        (0.388, 4.1,  "ω=64\n(current)", RED),
        (0.25,  1.0,  "~threshold",      GREEN),
    ]
    for (x, y, label, col) in models:
        ax_p.scatter([x], [y], color=col, s=80, zorder=5)
        ax_p.annotate(label, (x, y), xytext=(x + 0.04, y + 0.3),
                      fontsize=8, color=col,
                      arrowprops=dict(arrowstyle="-", color=col, lw=0.8))

    ax_p.axvline(0.20, color=GREEN, ls=":", lw=1.5, label="~20% RelL2 threshold")
    ax_p.set_xlabel("Model RelL2 (‖x_pred - x_true‖ / ‖x_true‖)", fontsize=10)
    ax_p.set_ylabel("Initial residual ratio  r₀/‖b‖", fontsize=10)
    ax_p.set_title("Why model quality matters for warm-start", fontsize=10)
    ax_p.set_xlim(0, 1.1); ax_p.set_ylim(0, 5)
    ax_p.legend(fontsize=8, loc="upper left")
    ax_p.grid(True, alpha=0.2)
    ax_p.text(0.10, 0.4, "want to be here", fontsize=8, color=GREEN,
              style="italic")
    ax_p.text(0.50, 3.5, "currently here", fontsize=8, color=RED, style="italic")

    # Right: key insight boxes
    right_x = 0.58
    insights = [
        (ORANGE, "Current situation",
         "RelL2 ≈ 39%  →  r₀/‖b‖ ≈ 4×\n"
         "Warm start increases initial residual.\n"
         "CSL-FGMRES needs more work to undo the bad start."),
        (GREEN, "Target",
         "RelL2 ≲ 20%  →  r₀/‖b‖ < 1×\n"
         "Warm start reduces initial residual.\n"
         "CSL-FGMRES needs fewer iterations."),
        (BLUE, "TransferUNet expectation",
         "T_down currently at 0.43% RelL2 (100× improvement).\n"
         "T_up training next — expect similar quality.\n"
         "Re-run warm-start eval once T_up checkpoint is ready."),
    ]
    ys = [0.76, 0.52, 0.27]
    for (col, title, text), yt in zip(insights, ys):
        ax.add_patch(FancyBboxPatch((right_x, yt - 0.18), 0.38, 0.22,
                                    boxstyle="round,pad=0.01",
                                    facecolor="white", edgecolor=col, lw=2.0))
        ax.text(right_x + 0.02, yt + 0.005, title,
                fontsize=11, fontweight="bold", color=col)
        ax.text(right_x + 0.02, yt - 0.06, text,
                fontsize=9, color=DARK, linespacing=1.45)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 9 — ω=128 is the meaningful regime
# ─────────────────────────────────────────────────────────────────────────────
def slide_omega128(pdf):
    fig, ax = new_slide("The Right Test: ω=128 on a Fixed 512×512 Grid",
                        "Per Kees: 'ω=128 on a fixed grid is the meaningful regime'")
    add_footer(fig)

    # Left column: why ω=128 matters
    ax.text(0.06, 0.80, "Why ω=128 is the target regime", fontsize=13,
            fontweight="bold", color=DARK)

    pts_l = [
        ("ω=32, ω=64:",
         "CSL-preconditioned FGMRES converges in 3 iterations from zero.\n"
         "Any warm-start benefit is invisible — can't improve on near-instant convergence."),
        ("ω=128:",
         "Harder problem, more FGMRES iterations needed.\n"
         "Warm-start has room to reduce iteration count meaningfully."),
        ("'β not lower than 0.3' (Kees):",
         "Keep CSL with β=0.5; direct solve confirmed in benchmark code.\n"
         "40–50 iterations reported at ω≤32 for harder setups."),
    ]
    y = 0.70
    for label, text in pts_l:
        ax.text(0.07, y, label, fontsize=10, fontweight="bold", color=BLUE)
        ax.text(0.07, y - 0.065, text, fontsize=9, color=DARK, linespacing=1.4)
        y -= 0.175

    # Right column: what's blocking + roadmap
    ax.axvline(0.52, ymin=0.08, ymax=0.88, color="#BDC3C7", lw=1)
    ax.text(0.55, 0.80, "Roadmap to a valid ω=128 result", fontsize=13,
            fontweight="bold", color=DARK)

    steps = [
        (GREEN, "DONE",
         "Train T_down 32→16 (wave7b, ep 167, 0.43% val)"),
        (ORANGE, "RUNNING",
         "Train T_up 16→32 on wave7b (next after T_down)"),
        (ORANGE, "RUNNING",
         "Train T_up 32→64, T_up 64→128 (wave7b queue)"),
        (GREY, "NEXT",
         "Re-run benchmark_warmstart.py --omega 128 with\nnew precond_v3 T_up checkpoint"),
        (GREY, "NEXT",
         "Generate qualitative visual:\nsolution panels at iter 0, 1, 2, 3 for ω=128"),
    ]

    ys = np.linspace(0.68, 0.13, len(steps))
    for (col, status, text), yt in zip(steps, ys):
        status_badge(ax, 0.60, yt, status, col, fontsize=8)
        ax.text(0.665, yt, text, fontsize=9, va="center", color=DARK,
                linespacing=1.3)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 10 — Action items and next steps
# ─────────────────────────────────────────────────────────────────────────────
def slide_next_steps(pdf):
    fig, ax = new_slide("Action Items & Next Steps")
    add_footer(fig)

    ax.text(0.06, 0.83, "Immediate (this week)", fontsize=13,
            fontweight="bold", color=DARK)
    ax.axhline(0.80, xmin=0.05, xmax=0.52, color="#BDC3C7", lw=0.8)

    immediate = [
        ("1", "Wait for T_up training on wave7b\n(pair_16_32 → 32_64 → 64_128 sequentially, ~3–7h/pair)"),
        ("2", "Copy results from /tmp to NFS:\n  cp -r /tmp/fkiewiet/precond_v3_N9600 ~/Freq2Transfer/experiments/claude/precond_v3/runs_N9600"),
        ("3", "Re-run warm-start benchmark with new T_up checkpoint at ω=32, 64, 128\n  python experiments/claude/benchmark_warmstart.py --omega 128 --ckpt <new_best.pt>"),
        ("4", "Generate qualitative visual: solution panels at iter 0, 1, 2, 3  (ω=128)"),
    ]

    y = 0.74
    for num, text in immediate:
        ax.add_patch(FancyBboxPatch((0.06, y - 0.045), 0.44, 0.075,
                                    boxstyle="round,pad=0.01",
                                    facecolor="#EAF2FF", edgecolor="none"))
        ax.text(0.075, y + 0.005, num, fontsize=12, fontweight="bold",
                color=BLUE, va="center")
        ax.text(0.105, y + 0.005, text, fontsize=8.5, color=DARK,
                va="center", linespacing=1.3)
        y -= 0.115

    ax.text(0.56, 0.83, "Open questions (Kees)", fontsize=13,
            fontweight="bold", color=DARK)
    ax.axhline(0.80, xmin=0.54, xmax=0.95, color="#BDC3C7", lw=0.8)

    open_q = [
        (ORANGE, "Why emergent structures?",
         "CNN learns structured wave patterns not seen in training.\n"
         "Plan: Fourier analysis of learned filters + feature maps."),
        (ORANGE, "CSL: constant β or adaptive?",
         "TO UITZOEK: what conditions require constant vs adaptive β?\n"
         "Check resonance — only at true null eigenvalue, not near-zero."),
        (BLUE, "Method chapter split",
         "Warm-start and preconditioning currently combined in §5–6.\n"
         "Kees: split for clarity. Write warm-start as standalone section."),
        (BLUE, "Eigenvalue plot (§3.2 TODO)",
         "x-axis: λ₁...λ₁₀₀₀₀,  3 curves (one per freq).\n"
         "Largest eigenvalue scales like √2  (Kees)."),
    ]
    y = 0.73
    for col, title, text in open_q:
        ax.add_patch(FancyBboxPatch((0.55, y - 0.055), 0.41, 0.085,
                                    boxstyle="round,pad=0.01",
                                    facecolor="white", edgecolor=col, lw=1.5))
        ax.text(0.565, y + 0.005, title, fontsize=9, fontweight="bold",
                color=col, va="center")
        ax.text(0.565, y - 0.035, text, fontsize=8, color=DARK,
                va="center", linespacing=1.3)
        y -= 0.130

    # Bottom: gate check
    ax.add_patch(FancyBboxPatch((0.05, 0.02), 0.90, 0.075,
                                boxstyle="round,pad=0.01",
                                facecolor="#EAFAF1", edgecolor=GREEN, lw=1.5))
    ax.text(0.5, 0.057,
            "W1 Gate: k=0 residual improves (r₀/‖b‖ < 1)  →  proceed to Stage II (Learned V-Cycle)",
            ha="center", va="center", fontsize=10, fontweight="bold", color=GREEN)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE 11 — Summary
# ─────────────────────────────────────────────────────────────────────────────
def slide_summary(pdf):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(DARK)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.text(0.5, 0.92, "Summary", ha="center", fontsize=26, fontweight="bold",
             color="white")
    ax.axhline(0.86, xmin=0.1, xmax=0.9, color=BLUE, lw=2)
    fig.text(0.5, 0.015, "FreqTransfer  ·  April 22, 2026", ha="center",
             fontsize=9, color=GREY)

    takeaways = [
        (ORANGE, "Warm-start eval complete",
         "First real result. Current models (39% RelL2) give r₀/‖b‖ ≈ 4× — worse than zero start."),
        (RED, "ω=32/64: regime too easy",
         "CSL-FGMRES converges in 3 iters from zero. No room for warm-start to help."),
        (GOLD, "ω=128 is the test that matters",
         "Harder problem, more iterations. But need a trained T_up 64→128 model first."),
        (GREEN, "TransferUNet shows 100× improvement",
         "T_down 32→16: 0.43% val (vs 39% before). T_up training running on wave7b."),
        (BLUE, "Path to positive result",
         "Wait for T_up checkpoint → re-run benchmark at ω=128 → check W1 gate."),
    ]

    ys = np.linspace(0.78, 0.14, len(takeaways))
    for (col, title, text), y in zip(takeaways, ys):
        ax.add_patch(FancyBboxPatch((0.06, y - 0.065), 0.88, 0.09,
                                    boxstyle="round,pad=0.01",
                                    facecolor="#1a2535", edgecolor=col, lw=1.8))
        ax.add_patch(FancyBboxPatch((0.06, y - 0.065), 0.04, 0.09,
                                    boxstyle="round,pad=0.01",
                                    facecolor=col, edgecolor="none"))
        ax.text(0.14, y - 0.020, title, fontsize=11, fontweight="bold",
                color=col, va="center")
        ax.text(0.14, y - 0.055, text, fontsize=9, color=LIGHT, va="center")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Build PDF
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Generating slides → {OUT}")
    with PdfPages(OUT) as pdf:
        slide_title(pdf)
        slide_commitments(pdf)
        slide_metric_explained(pdf)
        slide_convergence_curves(pdf)
        slide_field_snapshot(pdf)
        slide_diagnosis(pdf)
        slide_new_training(pdf)
        slide_threshold(pdf)
        slide_omega128(pdf)
        slide_next_steps(pdf)
        slide_summary(pdf)

        d = pdf.infodict()
        d["Title"] = "FreqTransfer Weekly Update — April 22, 2026"
        d["Author"] = "Fenna Kiewiet"
        d["Subject"] = "Stage I Warm-Start Results"

    print(f"Done: {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
