"""
helmholtz_conditioning.py
─────────────────────────
Two publication figures for the numerical dispersion / conditioning chapter.

  Figure 1  Eigenvalue distribution of A(ω) — interior only, no PML
            Analytical formula, runs in seconds at any N.
            Shows the indefinite structure: most eigenvalues << 0,
            a small cluster near k², near-zero eigenvalues cause κ → ∞.

  Figure 2  Condition number κ(ω) — three curves
            No-PML (analytical, enormous), constant σ₀, adaptive σ₀.
            Shows what the PML buys and why σ₀ choice matters.

Physical setup:  unit domain [0,1]²,  h = 1/N,  k = ω.

Usage
─────
  # Full run — N=512 PML curves take several hours, use tmux:
  tmux new-session -s cond
  cd ~/Freq2Transfer && source .venv/bin/activate
  python experiments/claude/helmholtz_conditioning.py

  # Re-draw figures from saved results (seconds):
  python experiments/claude/helmholtz_conditioning.py --replot

  # Quick test with a smaller grid:
  python experiments/claude/helmholtz_conditioning.py --N 64
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── parameters ────────────────────────────────────────────────────────────────
OMEGAS      = [16, 32, 64, 128]
SIGMA_CONST = 85.0
sigma_adapt = lambda w: 6.203 * float(w) ** 0.694

C = dict(
    nopml = "#888888",
    const = "#2E6DA4",
    adapt = "#E07B39",
    neg   = "#2E6DA4",
    pos   = "#E07B39",
    zero  = "#C0392B",
)

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--N",      type=int, default=512, help="grid size (default 512)")
ap.add_argument("--replot", action="store_true",   help="skip compute, replot from JSON")
args = ap.parse_args()

N     = args.N
n_pml = max(4, round(112 / 512 * N))
dx    = 1.0 / N
OUT   = Path("experiments/claude/conditioning")
OUT.mkdir(parents=True, exist_ok=True)
JSON  = OUT / f"results_N{N}.json"

print(f"N={N}  n_pml={n_pml}  h=1/{N}={dx:.5f}")
print(f"Interior diagonal: ω² − 4/h² = ω² − {4/dx**2:.2e}  (negative for all ω ✓)")
print(f"Output → {OUT}/")
print()


# ── Helmholtz FD matrix with PML ──────────────────────────────────────────────

def build_helmholtz(N, n_pml, omega, sigma0, dx):
    """
    5-point FD stencil for (Δ + k²)u = f on [0,1]² with complex PML.
      s(x) = 1 + i·σ(x)/ω,   σ quadratic ramp from 0 to σ₀ over n_pml cells.
    Returns complex CSC sparse matrix of size N²×N².
    """
    k     = float(omega)
    sigma = np.zeros(N)
    i_pml = np.arange(n_pml)
    ramp  = sigma0 * ((n_pml - i_pml) / n_pml) ** 2
    sigma[:n_pml]     = ramp
    sigma[N - n_pml:] = ramp[::-1]
    s  = 1.0 + 1j * sigma / omega          # PML stretching function [N]

    ii, jj = np.mgrid[0:N, 0:N]
    p  = (ii * N + jj).ravel()
    ax = (1.0 / (s[jj] * dx**2)).ravel()  # x-direction coefficient
    ay = (1.0 / (s[ii] * dx**2)).ravel()  # y-direction coefficient
    vd = -2*ax - 2*ay + k**2              # diagonal

    mr = (jj < N-1).ravel();  ml = (jj > 0).ravel()
    mb = (ii < N-1).ravel();  mu = (ii > 0).ravel()

    rows = np.concatenate([p,      p[mr],   p[ml],   p[mb],   p[mu]])
    cols = np.concatenate([p,      p[mr]+1, p[ml]-1, p[mb]+N, p[mu]-N])
    data = np.concatenate([vd, ax[mr], ax[ml], ay[mb], ay[mu]])
    return sp.coo_matrix((data, (rows, cols)), shape=(N*N, N*N)).tocsc()


def kappa_pml(A):
    """κ = σ_max / σ_min via ARPACK + LU inverse iteration."""
    n     = A.shape[0]
    s_max = float(spla.svds(A, k=1, which="LM",
                             return_singular_vectors=False, tol=1e-5)[0])
    lu    = spla.splu(A)
    Ainv  = spla.LinearOperator(
                (n, n),
                matvec =lambda x: lu.solve(np.asarray(x, dtype=complex)),
                rmatvec=lambda x: lu.solve(np.asarray(x, dtype=complex), trans="H"),
                dtype=complex)
    s_min = 1.0 / float(spla.svds(Ainv, k=1, which="LM",
                                    return_singular_vectors=False, tol=1e-5)[0])
    return s_max, s_min, s_max / s_min


# ── computation (analytical no-PML + numerical PML, saved incrementally) ──────

def compute():
    res = {"N": N, "n_pml": n_pml, "h": dx, "data": {}}

    # Resume from previous partial run if available
    if JSON.exists():
        try:
            res = json.loads(JSON.read_text())
            done = list(res["data"].keys())
            print(f"Resuming from {JSON}  (completed ω: {done})\n")
        except Exception:
            pass

    # Precompute base Laplacian eigenvalues (analytical, same for all ω)
    m = np.arange(1, N)
    mm, nn = np.meshgrid(m, m)
    laplacian_eigs = (4 / dx**2) * (np.sin(mm * np.pi * dx / 2)**2 +
                                     np.sin(nn * np.pi * dx / 2)**2)

    for om in OMEGAS:
        key = str(om)
        if key in res["data"]:
            print(f"ω={om:3d}  already done, skipping")
            continue

        k2   = float(om) ** 2
        eigs = (k2 - laplacian_eigs).ravel()   # all real, shape (N-1)²

        # ── no-PML stats (analytical) ──────────────────────────────────────
        n_neg    = int((eigs < 0).sum())
        n_tot    = len(eigs)
        sv_min   = float(np.abs(eigs).min())
        sv_max   = float(np.abs(eigs).max())
        kappa_np = sv_max / sv_min

        # Build symlog-spaced histogram bins centred around the interesting region
        linthresh  = max(k2, 1.0)
        e_min      = float(eigs.min())
        e_max      = float(eigs.max())
        neg_bins   = -np.geomspace(linthresh, -e_min * 1.001, 80)[::-1]
        lin_bins   = np.linspace(-linthresh, linthresh, 30)
        pos_bins   = (np.geomspace(linthresh, e_max * 1.001, 20)
                      if e_max > linthresh else np.array([linthresh, linthresh * 1.01]))
        bins       = np.unique(np.concatenate([neg_bins, lin_bins[1:-1], pos_bins]))
        counts, edges = np.histogram(eigs, bins=bins)

        print(f"ω={om:3d}  k²={k2:.0f}  "
              f"neg={n_neg}/{n_tot} ({100*n_neg/n_tot:.2f}%)  "
              f"σ_min={sv_min:.3e}  κ_nopml={kappa_np:.3e}")

        # ── PML condition numbers (numerical, slow) ────────────────────────
        t0 = time.time()
        _, _, kc = kappa_pml(build_helmholtz(N, n_pml, om, SIGMA_CONST, dx))
        print(f"        const σ₀={SIGMA_CONST:.0f}:         κ={kc:.3e}  "
              f"({time.time()-t0:.0f}s)")

        t0 = time.time()
        sa = sigma_adapt(om)
        _, _, ka = kappa_pml(build_helmholtz(N, n_pml, om, sa, dx))
        print(f"        adapt σ₀={sa:.1f} (σ₀/ω={sa/om:.2f}):  κ={ka:.3e}  "
              f"({time.time()-t0:.0f}s)")
        print()

        res["data"][key] = dict(
            omega   = om,
            k2      = k2,
            nopml   = dict(kappa=kappa_np, sv_min=sv_min, sv_max=sv_max,
                           n_neg=n_neg, n_tot=n_tot,
                           hist_counts=counts.tolist(),
                           hist_edges=edges.tolist()),
            const   = dict(sigma0=SIGMA_CONST, kappa=kc),
            adapt   = dict(sigma0=sa, kappa=ka),
        )
        JSON.write_text(json.dumps(res, indent=2))
        print(f"        saved → {JSON}")

    return res


# ── Figure 1: eigenvalue distribution (no PML) ────────────────────────────────

def fig1(res):
    """
    1×4 histogram panels, one per ω.
    x-axis: symlog scale (linear near 0, log for large |λ|).
    Blue = negative eigenvalues, orange = positive.
    Dashed red line at λ=0, dotted line at λ=k².
    """
    N_r = res["N"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), constrained_layout=True)

    for col, om in enumerate(OMEGAS):
        d      = res["data"][str(om)]
        ax     = axes[col]
        k2     = d["k2"]
        np_    = d["nopml"]
        counts = np.array(np_["hist_counts"])
        edges  = np.array(np_["hist_edges"])
        mids   = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)
        neg    = mids < 0

        ax.bar(mids[neg],  counts[neg],  width=widths[neg],
               color=C["neg"], alpha=0.70, label=r"$\lambda < 0$")
        ax.bar(mids[~neg], counts[~neg], width=widths[~neg],
               color=C["pos"], alpha=0.85, label=r"$\lambda \geq 0$")

        ax.axvline(0,  color=C["zero"], lw=1.5, ls="--", zorder=4)
        ax.axvline(k2, color=C["pos"],  lw=1.0, ls=":",  zorder=4)

        linthresh = max(k2, 1.0)
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.set_xlabel(r"$\lambda$", fontsize=10)
        if col == 0:
            ax.set_ylabel("eigenvalue count", fontsize=8)
        ax.set_title(f"$\\omega = {om}$", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=7)

        # Mark λ=0 and λ=k²
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else counts.max()
        ax.text(0,  ymax * 0.92, r"$\lambda=0$",  ha="center",
                va="top", fontsize=7, color=C["zero"])
        ax.text(k2, ymax * 0.92, f"$k^2={k2:.0f}$", ha="center",
                va="top", fontsize=7, color=C["pos"])

        pct = 100 * np_["n_neg"] / np_["n_tot"]
        ax.text(0.03, 0.97,
                f"neg: {np_['n_neg']}/{np_['n_tot']} ({pct:.1f}\\%)\n"
                f"$\\sigma_{{\\min}} = {np_['sv_min']:.2e}$\n"
                f"$\\kappa = {np_['kappa']:.2e}$",
                transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

        if col == 0:
            ax.legend(fontsize=7.5, loc="center left")

    fig.suptitle(
        r"Eigenvalue distribution: $A(\omega) = \Delta_h + k^2$"
        " — interior Helmholtz, Dirichlet BC, no PML\n"
        r"$\mu_{mn} = k^2 - \frac{4}{h^2}"
        r"\![\sin^2\!\frac{m\pi h}{2} + \sin^2\!\frac{n\pi h}{2}]$"
        f",  $N={N_r}$,  $h=1/{N_r}$",
        fontsize=9.5, fontweight="bold")

    for ext in ("png", "pdf"):
        out = OUT / f"fig1_eigenvalues.{ext}"
        fig.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT}/fig1_eigenvalues.{{png,pdf}}")


# ── Figure 2: κ(ω) scaling ────────────────────────────────────────────────────

def fig2(res):
    """
    Single log-log panel: three κ(ω) curves + reference slopes.
    """
    oms = np.array(OMEGAS, dtype=float)
    kn  = np.array([res["data"][str(w)]["nopml"]["kappa"] for w in OMEGAS])
    kc  = np.array([res["data"][str(w)]["const"]["kappa"] for w in OMEGAS])
    ka  = np.array([res["data"][str(w)]["adapt"]["kappa"] for w in OMEGAS])

    fig, ax = plt.subplots(figsize=(5.8, 4.5))

    ax.loglog(oms, kn, color=C["nopml"], lw=1.5, ls=":", marker="o", ms=5,
              label="No PML  (Dirichlet BC)")
    ax.loglog(oms, kc, color=C["const"], lw=2.0, ls="-", marker="o", ms=6,
              label=f"PML,  constant $\\sigma_0 = {SIGMA_CONST:.0f}$")
    ax.loglog(oms, ka, color=C["adapt"], lw=2.0, ls="-", marker="s", ms=6,
              label=r"PML,  adaptive $\sigma_0(\omega) = 6.203\,\omega^{0.694}$")

    # Reference slope lines between ω=32 and ω=128
    x_ref = np.array([32.0, 128.0])
    ax.plot(x_ref, kc[1] * (x_ref / 32)**2,   color=C["const"], lw=0.9,
            ls="--", alpha=0.5)
    ax.plot(x_ref, ka[1] * (x_ref / 32)**1.4, color=C["adapt"], lw=0.9,
            ls="--", alpha=0.5)
    ax.text(75, kc[1] * (75/32)**2   * 1.7,  r"$\sim\!\omega^2$",
            fontsize=9, color=C["const"])
    ax.text(75, ka[1] * (75/32)**1.4 * 0.55, r"$\sim\!\omega^{1.4}$",
            fontsize=9, color=C["adapt"])

    ax.set_xticks(OMEGAS)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel(r"$\omega$", fontsize=12)
    ax.set_ylabel(r"$\kappa\!\left(A(\omega)\right)$", fontsize=12)
    ax.set_title(
        "Condition number of the Helmholtz FD operator\n"
        f"$N={res['N']}$,  $h=1/{res['N']}$",
        fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5, frameon=True, loc="upper left")
    ax.grid(True, which="both", alpha=0.2)
    ax.tick_params(labelsize=9)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        out = OUT / f"fig2_kappa_scaling.{ext}"
        fig.savefig(out, dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT}/fig2_kappa_scaling.{{png,pdf}}")


# ── main ──────────────────────────────────────────────────────────────────────

if args.replot:
    if not JSON.exists():
        raise FileNotFoundError(
            f"No results at {JSON}. Run without --replot first.")
    res = json.loads(JSON.read_text())
    print(f"Loaded {JSON}  (ω completed: {list(res['data'].keys())})\n")
else:
    res = compute()

fig1(res)
fig2(res)
print("\nDone.")
