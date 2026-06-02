from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DIRICHLET = ROOT / "iteration_curves" / "pair_16_32_dirichlet_beta0p3" / "mean_iteration_metrics.csv"
PML = ROOT / "iteration_curves" / "pair_16_32_pml_beta0p3" / "mean_iteration_metrics.csv"
OUT = ROOT / "combined_1d_dirichlet_pml_true_residual.png"


def plot_panel(ax, csv_path, methods, title):
    df = pd.read_csv(csv_path)
    styles = {
        "cold": dict(color="0.15", linestyle="-", linewidth=2.2, label="cold"),
        "raw_unet": dict(color="#1f77b4", linestyle="-", linewidth=2.2, label="raw U-Net"),
        "residual_gate": dict(color="#2ca02c", linestyle="-", linewidth=2.2, label="residual gate"),
        "green_zero": dict(color="#d62728", linestyle="-", linewidth=2.2, label="green zero-PML"),
        "flux_full": dict(color="#9467bd", linestyle="-", linewidth=2.2, label="flux full-PML"),
    }
    for method in methods:
        sub = df[df["method"] == method].sort_values("iteration")
        if sub.empty:
            raise ValueError(f"Missing method {method!r} in {csv_path}")
        ax.semilogy(sub["iteration"], sub["true_residual"], **styles[method])
    ax.set_title(title)
    ax.set_xlabel("FGMRES iteration")
    ax.set_ylabel(r"true residual $\|b-Ax_k\|_2 / \|b\|_2$")
    ax.grid(True, which="both", color="0.88", linewidth=0.8)
    ax.legend(frameon=False)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=False)
    plot_panel(
        axes[0],
        DIRICHLET,
        ["cold", "raw_unet", "residual_gate"],
        "1D Dirichlet, 16 -> 32, beta=0.3",
    )
    plot_panel(
        axes[1],
        PML,
        ["cold", "green_zero", "flux_full"],
        "1D PML, 16 -> 32, beta=0.3",
    )
    fig.suptitle("True FGMRES residuals: spectral gating vs PML-compatible transfer", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=250, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
