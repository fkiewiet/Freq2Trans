# Chapter 5 Colab Figure Generation

Paste the following as one cell in Google Colab. It creates the Chapter 5
figures with exactly the filenames used in
`thesis/chapter5_frequency_transfer_operator.tex`.

```python
# Chapter 5 figure generation: frequency transfer background
# Output folder in Colab: /content/figures/ch5

import os
import zipfile
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

try:
    from scipy.special import hankel1
except ImportError:
    import sys
    !{sys.executable} -m pip install scipy
    from scipy.special import hankel1

OUT = Path("/content/figures/ch5")
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "figure.dpi": 170,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "dejavusans",
})

BLUE = "#2f6f9f"
TEAL = "#1b9e77"
ORANGE = "#d95f02"
PURPLE = "#7b52ab"
GRAY = "#4d4d4d"
LIGHT = "#f7f7f7"
EDGE = "#222222"


def savefig(name):
    path = OUT / name
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved {path}")


def add_box(ax, xy, w, h, text, fc="white", ec=EDGE, fontsize=11, lw=1.4):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        fc=fc, ec=ec, lw=lw
    )
    ax.add_patch(box)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
            fontsize=fontsize)
    return box


def arrow(ax, p0, p1, color=EDGE, lw=1.6, ms=13, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
        shrinkA=4, shrinkB=4
    ))


# ---------------------------------------------------------------------
# 1. Frequency coherence: real Green's function at omega=32,64,128
# ---------------------------------------------------------------------
def make_green_frequency_coherence():
    n = 512
    x = np.linspace(-0.5, 0.5, n)
    X, Y = np.meshgrid(x, x, indexing="xy")
    r = np.sqrt(X**2 + Y**2)
    r = np.maximum(r, 1.0 / n)
    omegas = [32, 64, 128]

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0), constrained_layout=True)
    vmax = 0.08

    for ax, omega in zip(axes, omegas):
        G = 0.25j * hankel1(0, omega * r)
        val = np.real(G)
        val = val / np.max(np.abs(val)) * vmax
        im = ax.imshow(
            val, extent=[0, 512, 0, 512], origin="lower",
            cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear"
        )
        ax.set_title(rf"$\omega={omega}$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_xticks([0, 256, 512])
        ax.set_yticks([0, 256, 512])
        ax.set_aspect("equal")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(r"$\mathrm{Re}(G_\omega)$, normalised")
    savefig("green_frequency_coherence.png")


# ---------------------------------------------------------------------
# 2. Radial cross-section with amplitude ratio and near-field shading
# ---------------------------------------------------------------------
def make_radial_green_cross_section():
    r = np.linspace(0.006, 0.50, 1400)
    omega_L, omega_H = 64, 128
    G_L = 0.25j * hankel1(0, omega_L * r)
    G_H = 0.25j * hankel1(0, omega_H * r)
    near = 0.045

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True,
                             constrained_layout=True)

    axes[0].plot(r, np.abs(G_L), color=BLUE, lw=2.0,
                 label=rf"$|G(r;\omega_L)|,\ \omega_L={omega_L}$")
    axes[0].plot(r, np.abs(G_H), color=ORANGE, lw=2.0,
                 label=rf"$|G(r;\omega_H)|,\ \omega_H={omega_H}$")
    axes[0].plot(r, np.sqrt(2) * np.abs(G_H), color=ORANGE, lw=1.4, ls="--",
                 label=rf"$\sqrt{{2}}\,|G(r;\omega_H)|$")
    axes[0].axvspan(r[0], near, color=PURPLE, alpha=0.12, label="near-field")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$|G(r;\omega)|$")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(loc="upper right", frameon=False)

    axes[1].plot(r, np.real(G_L), color=BLUE, lw=2.0,
                 label=rf"$\mathrm{{Re}}\,G(r;\omega_L)$")
    axes[1].plot(r, np.real(G_H), color=ORANGE, lw=2.0,
                 label=rf"$\mathrm{{Re}}\,G(r;\omega_H)$")
    axes[1].axhline(0, color="black", lw=0.8, alpha=0.35)
    axes[1].axvspan(r[0], near, color=PURPLE, alpha=0.12)
    axes[1].set_xlabel(r"radial distance $r$")
    axes[1].set_ylabel(r"$\mathrm{Re}\,G(r;\omega)$")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", frameon=False)

    savefig("radial_green_cross_section.png")


# ---------------------------------------------------------------------
# 3. Exact solution-transfer schematic
# ---------------------------------------------------------------------
def make_solution_transfer_schematic():
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)

    add_box(ax, (0.35, 1.25), 1.0, 0.55, r"$f$", fc="#ffffff")
    add_box(ax, (2.0, 2.0), 1.7, 0.55, r"$\mathcal{A}_L^{-1}$", fc="#e8f2fb")
    add_box(ax, (4.15, 2.0), 1.3, 0.55, r"$u_L$", fc="#e8f2fb")
    add_box(ax, (6.0, 2.0), 1.8, 0.55, r"$\mathcal{T}_{\uparrow}$", fc="#e9f6ef")
    add_box(ax, (8.3, 1.25), 1.3, 0.55, r"$u_H$", fc="#fff3e6")
    add_box(ax, (2.0, 0.5), 1.7, 0.55, r"$\mathcal{A}_H^{-1}$", fc="#fff3e6")

    arrow(ax, (1.35, 1.52), (2.0, 2.28))
    arrow(ax, (3.7, 2.28), (4.15, 2.28))
    arrow(ax, (5.45, 2.28), (6.0, 2.28))
    arrow(ax, (7.8, 2.28), (8.3, 1.52))
    arrow(ax, (1.35, 1.52), (2.0, 0.78))
    arrow(ax, (3.7, 0.78), (8.3, 1.42))

    ax.text(5.0, 0.18,
            r"$\mathcal{T}_{\uparrow}=\mathcal{A}_H^{-1}\mathcal{A}_L$"
            "\n"
            r"$\mathcal{A}_H\mathcal{T}_{\uparrow}u_L"
            r"=\mathcal{A}_L u_L=f$",
            ha="center", va="bottom", fontsize=12)
    ax.text(5.0, 2.82, "solution transfer", ha="center",
            va="center", fontsize=15, weight="bold")

    savefig("solution_transfer_operator_schematic.png")


# ---------------------------------------------------------------------
# 4. Learned V-cycle schematic inside FGMRES
# ---------------------------------------------------------------------
def make_learned_vcycle_schematic():
    fig, ax = plt.subplots(figsize=(6.4, 7.0))
    ax.set_axis_off()
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 8)

    ax.text(3, 7.65, "learned V-cycle preconditioner",
            ha="center", va="center", fontsize=15, weight="bold")
    ax.text(3, 7.20, r"inside each FGMRES iteration: $z_H=M_\theta^{-1}r_H$",
            ha="center", va="center", fontsize=10.5, color=GRAY)

    add_box(ax, (1.9, 6.30), 2.2, 0.55, r"high-frequency residual $r_H^{(j)}$", fc="#eef5fb")
    add_box(ax, (1.9, 5.25), 2.2, 0.55, r"learned restriction $\mathcal{R}_\theta$", fc="#eaf7ef")
    add_box(ax, (1.9, 4.20), 2.2, 0.55, r"low-frequency problem $r_L$", fc="#f7f7f7")
    add_box(ax, (1.9, 3.15), 2.2, 0.55, r"coarse solve $\mathbf{A}_L^{-1}$", fc="#fff3e6")
    add_box(ax, (1.9, 2.10), 2.2, 0.55, r"low-frequency correction $e_L$", fc="#f7f7f7")
    add_box(ax, (1.9, 1.05), 2.2, 0.55, r"learned prolongation $\mathcal{P}_\theta$", fc="#eaf7ef")
    add_box(ax, (1.9, 0.10), 2.2, 0.55, r"high-frequency correction $z_H^{(j)}$", fc="#fff3e6")

    ys = [6.30, 5.25, 4.20, 3.15, 2.10, 1.05]
    for y0 in ys:
        arrow(ax, (3.0, y0), (3.0, y0 - 0.45))

    ax.text(4.55, 5.45, "restriction", color=TEAL, fontsize=11, va="center")
    ax.text(4.55, 3.42, "cheap low-frequency solve", color=ORANGE, fontsize=11, va="center")
    ax.text(4.55, 1.25, "prolongation", color=TEAL, fontsize=11, va="center")

    savefig("learned_vcycle_schematic.png")


# ---------------------------------------------------------------------
# 5. Three transfer objects diagram
# ---------------------------------------------------------------------
def make_three_transfer_objects():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    rows = [
        (3.7, "solution transfer", r"$u_L$", r"$T_{\uparrow}^{\mathrm{sol}}$", r"$u_H$",
         "complete wavefield to complete wavefield", "#e8f2fb"),
        (2.3, "residual restriction", r"$r_H$", r"$\mathcal{R}_{\theta}$", r"$r_L$ or $e_L$",
         "current high-frequency residual to low-frequency representation", "#eaf7ef"),
        (0.9, "correction prolongation", r"$e_L$", r"$\mathcal{P}_{\theta}$", r"$e_H$ or $z_H$",
         "low-frequency correction to high-frequency correction", "#fff3e6"),
    ]

    ax.text(5, 4.65, "three different transfer objects",
            ha="center", fontsize=16, weight="bold")

    for y, title, left, mid, right, desc, color in rows:
        ax.text(0.25, y + 0.18, title, ha="left", va="center",
                fontsize=12.5, weight="bold", color=GRAY)
        ax.text(0.25, y - 0.18, desc, ha="left", va="center",
                fontsize=9.5, color=GRAY)
        add_box(ax, (3.15, y - 0.25), 1.0, 0.5, left, fc="white")
        add_box(ax, (4.95, y - 0.25), 1.35, 0.5, mid, fc=color)
        add_box(ax, (7.15, y - 0.25), 1.15, 0.5, right, fc="white")
        arrow(ax, (4.15, y), (4.95, y))
        arrow(ax, (6.30, y), (7.15, y))

    ax.text(5, 0.18,
            "Same architecture family can be used, but the mathematical object and training target differ.",
            ha="center", fontsize=10.5, color=GRAY)
    savefig("three_transfer_objects.png")


# ---------------------------------------------------------------------
# 6. Modal amplification cartoon
# ---------------------------------------------------------------------
def make_modal_amplification_cartoon():
    k = np.arange(1, 129)
    lam = (k / k.max())**2
    error_coeff = 0.075 * np.exp(-k / 24) + 0.004 * np.exp(-((k - 105) / 10)**2)
    residual_coeff = lam * error_coeff
    residual_coeff = residual_coeff / residual_coeff.max() * error_coeff.max()

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6), constrained_layout=True)

    axes[0].plot(k, error_coeff, color=PURPLE, lw=2.4)
    axes[0].fill_between(k, error_coeff, color=PURPLE, alpha=0.18)
    axes[0].set_title("field-error coefficients")
    axes[0].set_xlabel("mode index")
    axes[0].set_ylabel(r"$|c_k(e_0)|$")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(k, residual_coeff, color=ORANGE, lw=2.4)
    axes[1].fill_between(k, residual_coeff, color=ORANGE, alpha=0.20)
    axes[1].set_title("residual coefficients")
    axes[1].set_xlabel("mode index")
    axes[1].set_ylabel(r"$|c_k(r_0)|=|\lambda_k c_k(e_0)|$")
    axes[1].grid(True, alpha=0.25)

    for ax in axes:
        ax.axvspan(90, 128, color=GRAY, alpha=0.08)
        ax.text(109, ax.get_ylim()[1]*0.86, "large\n$|\\lambda_k|$",
                ha="center", va="top", fontsize=9, color=GRAY)

    fig.suptitle("small high-mode field errors can become large residual errors",
                 fontsize=14, weight="bold")
    savefig("modal_amplification_cartoon.png")


# ---------------------------------------------------------------------
# 7. U-Net frequency-transfer schematic
# ---------------------------------------------------------------------
def make_unet_frequency_transfer_schematic():
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    ax.text(5, 4.65, "U-Net frequency-transfer model",
            ha="center", fontsize=16, weight="bold")

    add_box(ax, (0.35, 2.15), 1.25, 0.70, r"input" + "\n" + r"$u_L$ or $r_H$",
            fc="#ffffff")
    arrow(ax, (1.60, 2.50), (2.10, 2.50))

    enc_x = [2.1, 3.0, 3.9]
    enc_h = [0.95, 0.78, 0.62]
    enc_lab = ["encoder\n32 ch", "encoder\n64 ch", "encoder\n128 ch"]
    for x, h, lab in zip(enc_x, enc_h, enc_lab):
        add_box(ax, (x, 2.5 - h/2), 0.75, h, lab, fc="#e8f2fb", fontsize=9.5)
    arrow(ax, (2.85, 2.50), (3.0, 2.50))
    arrow(ax, (3.75, 2.50), (3.9, 2.50))

    add_box(ax, (4.95, 1.95), 1.05, 1.10, "bottleneck\nlarge-scale\ncontext",
            fc="#f7f7f7", fontsize=9.5)
    arrow(ax, (4.65, 2.50), (4.95, 2.50))

    dec_x = [6.35, 7.25, 8.15]
    dec_h = [0.62, 0.78, 0.95]
    dec_lab = ["decoder\n128 ch", "decoder\n64 ch", "decoder\n32 ch"]
    for x, h, lab in zip(dec_x, dec_h, dec_lab):
        add_box(ax, (x, 2.5 - h/2), 0.75, h, lab, fc="#fff3e6", fontsize=9.5)
    arrow(ax, (6.0, 2.50), (6.35, 2.50))
    arrow(ax, (7.10, 2.50), (7.25, 2.50))
    arrow(ax, (8.0, 2.50), (8.15, 2.50))

    add_box(ax, (9.05, 2.15), 1.25, 0.70, r"output" + "\n" + r"$u_H$ or $z_H$",
            fc="#ffffff")
    arrow(ax, (8.90, 2.50), (9.05, 2.50))

    # skip connections
    for x0, x1, y in [(2.48, 8.52, 3.55), (3.38, 7.62, 3.95), (4.28, 6.72, 4.25)]:
        ax.plot([x0, x0, x1, x1], [2.95, y, y, 2.95],
                color=TEAL, lw=1.5, ls="--")
        ax.text((x0 + x1)/2, y + 0.08, "skip", ha="center", fontsize=9, color=TEAL)

    ax.text(5, 0.65,
            "The architecture can be reused, but the learned object is set by the target and loss.",
            ha="center", fontsize=10.5, color=GRAY)
    savefig("unet_frequency_transfer_schematic.png")


make_green_frequency_coherence()
make_radial_green_cross_section()
make_solution_transfer_schematic()
make_learned_vcycle_schematic()
make_three_transfer_objects()
make_modal_amplification_cartoon()
make_unet_frequency_transfer_schematic()

zip_path = Path("/content/ch5_figures.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(OUT.glob("*.png")):
        zf.write(path, arcname=f"figures/ch5/{path.name}")
print(f"\nAll figures written to {OUT}")
print(f"Zip archive: {zip_path}")
```
