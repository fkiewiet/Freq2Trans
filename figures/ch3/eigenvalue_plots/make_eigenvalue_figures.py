"""
make_eigenvalue_figures.py
--------------------------
Generates Chapter 3 eigenvalue figures for the Freq2Transfer thesis.

Physical setup (matching Chapter 3 text):
  - Domain [0,1] x [0,1], spacing h = 1/(n+1) for n interior points
  - k = omega  (c = 1, physical angular frequency in rad/s)
  - n=100 proxy for sorted curves (N=10,000 DOFs)
  - n=50  proxy for complex scatter (N=2,500 DOFs, feasible full diagonalisation)

Figures produced:
  1. sorted_eigenvalues.png   -- "3 curves": sorted Helmholtz eigenvalues (no PML)
  2. pml_vs_nopml.png         -- before/after PML: real vs complex spectrum
  3. spectrum_scatter.png     -- omega-progression scatter with PML (fig:ch3-spectrum)
  4. condition_number.png     -- kappa(A(omega)) vs omega
"""

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Global style  (matches existing thesis figures: Arial, 300 dpi)
# ------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OMEGA_COLORS = {16: "#1f77b4", 32: "#ff7f0e", 64: "#2ca02c", 128: "#d62728"}
SIGMA0_MAP   = {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}

# ------------------------------------------------------------------
# 1. Analytical eigenvalues: 2D Helmholtz, no PML, Dirichlet BCs
# ------------------------------------------------------------------
def helmholtz_eigs_analytical(n, omega):
    """
    Exact eigenvalues of (-Delta_h + k^2 I) on [0,1]^2, Dirichlet BCs.
    Physical spacing h = 1/(n+1), k = omega.
    Returns (N,) real array sorted DESCENDING (largest first).
    """
    h   = 1.0 / (n + 1)
    k2  = float(omega) ** 2
    j   = np.arange(1, n + 1)
    mu  = (2.0 * np.cos(j * np.pi / (n + 1)) - 2.0) / h**2   # 1D Laplacian eigs
    lam = (mu[:, None] + mu[None, :]).ravel() + k2             # 2D: all pairs
    return np.sort(lam)[::-1]                                  # descending


# ------------------------------------------------------------------
# 2. Numerical sparse matrix: 2D Helmholtz + optional PML
# ------------------------------------------------------------------
def build_helmholtz_matrix(n, omega, n_pml=0, sigma0=0.0):
    """
    Build DENSE 2D Helmholtz matrix on [0,1]^2 with optional PML.
    Physical: h = 1/(n+1), k = omega.
    Only feasible for small n (n <= 60).
    Returns complex (n^2, n^2) array.
    """
    h  = 1.0 / (n + 1)
    k2 = float(omega) ** 2
    N  = n * n

    sigma = np.zeros(n)
    if n_pml > 0 and sigma0 > 0:
        for i in range(n_pml):
            val = sigma0 * ((n_pml - i) / n_pml) ** 2
            sigma[i]     = val
            sigma[n-1-i] = val

    s = 1.0 + 1j * sigma / omega  # complex PML stretching [n]

    rows, cols, vals = [], [], []

    def idx(i, j):
        return i * n + j

    for i in range(n):
        for j in range(n):
            p  = idx(i, j)
            ax = 1.0 / (s[j] * h**2)
            ay = 1.0 / (s[i] * h**2)
            rows.append(p); cols.append(p); vals.append(-2*ax - 2*ay + k2)
            if j+1 < n: rows.append(p); cols.append(idx(i, j+1)); vals.append(ax)
            if j-1 >= 0: rows.append(p); cols.append(idx(i, j-1)); vals.append(ax)
            if i+1 < n: rows.append(p); cols.append(idx(i+1, j)); vals.append(ay)
            if i-1 >= 0: rows.append(p); cols.append(idx(i-1, j)); vals.append(ay)

    A = sp.coo_matrix(
        (np.array(vals, dtype=complex), (rows, cols)),
        shape=(N, N)
    ).toarray()
    return A


# ==================================================================
# FIGURE 1 — Sorted eigenvalue curves ("3 curves")
# ==================================================================
print("Figure 1: sorted eigenvalues ...")

n_sorted = 100  # N = 10,000 DOFs
omegas_sorted = [32, 64, 128]
N_total = n_sorted ** 2

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- left panel: full view (symlog) ---
ax = axes[0]
ax.axhline(0, color="k", lw=0.8, ls="--", zorder=0)
ax.axhspan(-1e9, 0, color="#e8e8f0", zorder=0)

for omega in omegas_sorted:
    eigs = helmholtz_eigs_analytical(n_sorted, omega)
    ranks = np.arange(1, N_total + 1)
    ax.plot(ranks, eigs, color=OMEGA_COLORS[omega],
            lw=1.2, label=rf"$\omega = {omega}$")

ax.set_yscale("symlog", linthresh=1e3)
ax.set_xlabel("Eigenvalue rank $j$ (sorted descending)")
ax.set_ylabel(r"$\lambda_j(\mathbf{A})$")
ax.set_title("Sorted eigenvalue spectrum\n(no PML, Dirichlet BCs)")
ax.legend()
ax.set_xlim(1, N_total)
ax.text(0.02, 0.05, r"Indefinite region  $(\lambda < 0)$",
        transform=ax.transAxes, fontsize=9, color="#555599", va="bottom")

# --- right panel: zoom near zero, focus on transition ---
ax2 = axes[1]
ax2.axhline(0, color="k", lw=0.8, ls="--", zorder=0)
ax2.axhspan(-1e9, 0, color="#e8e8f0", zorder=0)

for omega in omegas_sorted:
    eigs = helmholtz_eigs_analytical(n_sorted, omega)
    ranks = np.arange(1, N_total + 1)

    # find zero-crossing rank
    cross_mask = np.where(eigs < 0)[0]
    cross_rank = cross_mask[0] + 1 if len(cross_mask) else N_total

    ax2.plot(ranks, eigs, color=OMEGA_COLORS[omega], lw=1.5,
             label=rf"$\omega = {omega}$,  $j^* \approx {cross_rank:,}$")
    ax2.axvline(cross_rank, color=OMEGA_COLORS[omega], lw=0.8, ls=":", alpha=0.6)

# zoom: show only ranks where eigenvalues are between -5k and +max
max_pos = helmholtz_eigs_analytical(n_sorted, 128)[0] * 1.05
ax2.set_ylim(-5000, max_pos)
ax2.set_xlim(1, 3000)
ax2.set_xlabel("Eigenvalue rank $j$ (sorted descending)")
ax2.set_ylabel(r"$\lambda_j(\mathbf{A})$")
ax2.set_title("Zoom: transition through zero\n" + r"$j^*$: zero-crossing rank")
ax2.legend(fontsize=9)
ax2.text(0.02, 0.05, r"Indefinite region  $(\lambda < 0)$",
         transform=ax2.transAxes, fontsize=9, color="#555599", va="bottom")

fig.suptitle(r"Eigenvalue distribution of $-\Delta_h + \omega^2 I$ (no PML, $n = 100$)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "sorted_eigenvalues.png")
plt.close(fig)
print("  saved sorted_eigenvalues.png")


# ==================================================================
# FIGURE 2 — PML vs no-PML (complex plane)
# Im(λ) is POSITIVE for PML modes (s = 1 + i*sigma/omega gives
# positive diagonal imaginary part; interior modes stay near Im=0).
# ==================================================================
print("Figure 2: PML vs no-PML ...")

n_scatter      = 40
n_pml_cells    = 9    # ~22% of domain
omega_demo     = 64
s0_demo        = SIGMA0_MAP[omega_demo]

A_nopml = build_helmholtz_matrix(n_scatter, omega_demo, n_pml=0,           sigma0=0)
A_pml   = build_helmholtz_matrix(n_scatter, omega_demo, n_pml=n_pml_cells, sigma0=s0_demo)

print("  diagonalising no-PML ...")
eigs_nopml = np.linalg.eigvals(A_nopml)
print("  diagonalising PML ...")
eigs_pml   = np.linalg.eigvals(A_pml)

# Separate interior modes (small Im) from PML modes (large Im)
im_pml = eigs_pml.imag
im_gap = np.percentile(im_pml, 75)   # PML modes are in top quartile of Im
interior_mask = im_pml < im_gap
pml_mask      = ~interior_mask

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- LEFT: no PML — eigenvalues are real (Im ≈ 0 up to float noise) ---
re1 = eigs_nopml.real
neg_mask = re1 < 0
ax1.axvspan(re1.min()*1.05, 0, color="#ffe8e8", alpha=0.7, zorder=0)
ax1.axvline(0, color="#cc0000", lw=0.8, ls="--", zorder=1)
ax1.scatter(re1[neg_mask],  np.zeros(neg_mask.sum()),  s=7, alpha=0.6,
            color="#1f77b4", zorder=2, label=r"$\lambda < 0$ (indefinite)")
ax1.scatter(re1[~neg_mask], np.zeros((~neg_mask).sum()), s=7, alpha=0.6,
            color="#d62728", zorder=2, label=r"$\lambda > 0$")
ax1.set_xlim(re1.min()*1.05, re1.max()*1.05)
ax1.set_ylim(-1, 1)   # eigenvalues sit on real axis
ax1.set_xlabel(r"Re$(\lambda)$")
ax1.set_ylabel(r"Im$(\lambda)$")
ax1.set_title(rf"$\omega = {omega_demo}$ — Without PML")
ax1.legend(fontsize=9, loc="upper left")
# annotate near-origin cluster
near_zero = re1[np.abs(re1) < re1.max() * 0.02]
if len(near_zero):
    ax1.annotate("near-resonant modes\n" + r"($\lambda \approx 0$)",
                 xy=(0, 0), xytext=(re1.max()*0.25, 0.6),
                 arrowprops=dict(arrowstyle="->", color="#666"), fontsize=9, color="#444")

# --- RIGHT: with PML — complex eigenvalues ---
re2 = eigs_pml.real
im2 = eigs_pml.imag
im_max = im2.max()

ax2.axvspan(re2.min()*1.05, 0, color="#ffe8e8", alpha=0.5, zorder=0)
ax2.axvline(0, color="#cc0000", lw=0.8, ls="--", zorder=1)
ax2.scatter(re2[interior_mask], im2[interior_mask], s=7, alpha=0.55,
            color=OMEGA_COLORS[omega_demo], zorder=2, label="Interior modes")
ax2.scatter(re2[pml_mask],      im2[pml_mask],      s=10, alpha=0.7,
            color="#9467bd", marker="^", zorder=3, label="PML modes")
ax2.set_xlim(re2.min()*1.05, re2.max()*1.05)
ax2.set_ylim(-im_max * 0.03, im_max * 1.08)
ax2.set_xlabel(r"Re$(\lambda)$")
ax2.set_ylabel(r"Im$(\lambda)$")
ax2.set_title(rf"$\omega = {omega_demo}$ — With PML")
ax2.legend(fontsize=9, loc="upper left")
ax2.annotate("PML modes: absorbed\n(large Im$(\lambda)$)",
             xy=(re2[pml_mask].mean(), im2[pml_mask].mean()),
             xytext=(re2.max()*0.3, im_max * 0.6),
             arrowprops=dict(arrowstyle="->", color="#666"), fontsize=9, color="#444")
ax2.annotate("Interior modes\nnear real axis",
             xy=(re2[interior_mask].mean(), im2[interior_mask].mean()),
             xytext=(re2.max()*0.3, im_max * 0.15),
             arrowprops=dict(arrowstyle="->", color="#666"), fontsize=9, color="#444")

fig.suptitle(f"Effect of PML on eigenvalue distribution\n"
             rf"($n = {n_scatter}$ proxy grid, $\omega = {omega_demo}$)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "pml_vs_nopml.png")
plt.close(fig)
print("  saved pml_vs_nopml.png")


# ==================================================================
# FIGURE 3 — omega-progression scatter  (fig:ch3-spectrum)
# Two panels: full complex plane (left) + zoom on interior (right)
# ==================================================================
print("Figure 3: omega-progression scatter ...")

n_scatter2 = 40
n_pml_c    = 9

# Build and cache eigenvalues (build once, not twice)
cached_eigs3 = {}
for omega in [16, 32, 64, 128]:
    print(f"  omega={omega} ...")
    s0   = SIGMA0_MAP[omega]
    A    = build_helmholtz_matrix(n_scatter2, omega, n_pml=n_pml_c, sigma0=s0)
    cached_eigs3[omega] = np.linalg.eigvals(A)

# Determine interior-mode Im cutoff from ω=64 spectrum
ref_eigs = cached_eigs3[64]
interior_cutoff = np.percentile(ref_eigs.imag, 80)  # top 20% are PML modes

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
handles3 = []

for omega in [16, 32, 64, 128]:
    eigs = cached_eigs3[omega]
    col  = OMEGA_COLORS[omega]

    # Full complex plane (left)
    axL.scatter(eigs.real, eigs.imag, s=4, alpha=0.4, color=col, zorder=2)

    # Interior modes only (right): Im < cutoff
    imask = eigs.imag < interior_cutoff
    axR.scatter(eigs.real[imask], eigs.imag[imask], s=5, alpha=0.5, color=col, zorder=2)
    handles3.append(mpatches.Patch(color=col, label=rf"$\omega = {omega}$"))

for ax in (axL, axR):
    ax.axvspan(-1e8, 0, color="#f5f0f0", alpha=0.55, zorder=0)
    ax.axvline(0, color="#cc0000", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel(r"Re$(\lambda)$")
    ax.set_ylabel(r"Im$(\lambda)$")

# Full plane limits (include PML modes)
all_re = np.concatenate([e.real for e in cached_eigs3.values()])
all_im = np.concatenate([e.imag for e in cached_eigs3.values()])
axL.set_xlim(np.percentile(all_re, 1)*1.1, np.percentile(all_re, 99)*1.1)
axL.set_ylim(-all_im.max()*0.03, all_im.max()*1.05)
axL.set_title("Full complex plane\n(interior + PML modes)")
axL.legend(handles=handles3, fontsize=9)

# Interior zoom limits
int_re = np.concatenate([e.real[e.imag < interior_cutoff]
                          for e in cached_eigs3.values()])
int_im = np.concatenate([e.imag[e.imag < interior_cutoff]
                          for e in cached_eigs3.values()])
axR.set_xlim(np.percentile(int_re, 1)*1.1, np.percentile(int_re, 99)*1.1)
axR.set_ylim(-interior_cutoff*0.05, interior_cutoff*1.05)
axR.set_title(r"Interior modes (zoom)" + "\n" +
              r"Shaded: Re$(\lambda)<0$ (indefinite region)")
axR.legend(handles=handles3, fontsize=9)

fig.suptitle(r"Eigenvalue distribution of $\mathbf{A}(\omega)$ for $\omega \in \{16,32,64,128\}$"
             "\n" + rf"(PML, $n={n_scatter2}$ proxy grid)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "spectrum_scatter.png")
plt.close(fig)
print("  saved spectrum_scatter.png")


# ==================================================================
# FIGURE 4 — Condition number vs omega
# No-PML, analytical, n=100 (N=10,000).  Dense omega sweep reveals
# the near-resonance spikes and the growing envelope.
# Requires n >= 46 so that 8n^2 > k^2_max = 128^2 = 16384.
# ==================================================================
print("Figure 4: condition number ...")

from scipy.ndimage import maximum_filter1d

n_kappa      = 100   # max|Lap| ≈ 8×100² = 80,000 >> 128² = 16,384  ✓
omega_dense  = np.arange(10, 131, 1, dtype=float)   # 121 values

print("  computing analytical kappa for omega = 10..130 ...")
kappa_dense = []
for omega in omega_dense:
    eigs = helmholtz_eigs_analytical(n_kappa, omega)
    lam_max = np.max(np.abs(eigs))
    lam_min = np.min(np.abs(eigs))
    kappa_dense.append(lam_max / lam_min if lam_min > 1e-6 else np.nan)

kappa_arr = np.array(kappa_dense, dtype=float)

# Envelope: rolling max over ±5 rad/s window to show growing trend
finite_mask   = np.isfinite(kappa_arr)
kappa_fill    = np.where(finite_mask, kappa_arr, 0.0)
kappa_envelope = maximum_filter1d(kappa_fill, size=11)
kappa_envelope[~finite_mask] = np.nan

# omega^3 reference anchored at the envelope value at omega=16
idx16    = np.where(omega_dense == 16)[0][0]
kappa_ref = float(kappa_envelope[idx16]) if np.isfinite(kappa_envelope[idx16]) else 1e3
ref_line  = kappa_ref * (omega_dense / 16.0) ** 3

fig, ax = plt.subplots(figsize=(8, 5))

# Raw condition numbers — grey dots showing near-resonance oscillations
ax.scatter(omega_dense[finite_mask], kappa_arr[finite_mask],
           s=12, color="#aaaacc", alpha=0.55, zorder=2, label=r"$\kappa(\omega)$ (near-resonance oscillations)")

# Envelope — growing trend
ax.semilogy(omega_dense, kappa_envelope,
            color="#1f77b4", lw=2.2, zorder=3, label="Growing envelope")

# omega^3 reference
ax.semilogy(omega_dense, ref_line,
            "k:", lw=1.4, zorder=1, label=r"$\sim\omega^3$ reference")

# Mark the four thesis frequencies
for omega_mark, ytext_frac in zip([16, 32, 64, 128], [0.12, 0.08, 0.06, 0.04]):
    ax.axvline(omega_mark, color="#cc0000", lw=0.8, ls="--", alpha=0.5, zorder=0)
    ax.text(omega_mark + 0.5, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1,
            str(omega_mark), fontsize=8, color="#cc0000", va="bottom")

ax.set_xlabel(r"$\omega$ (rad/s)")
ax.set_ylabel(r"Condition number $\kappa(\mathbf{A})$")
ax.set_title(r"Condition number $\kappa(\mathbf{A}(\omega))$ vs $\omega$"
             "\n" + rf"(no PML, Dirichlet BCs, $n={n_kappa}$ analytical proxy)")
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(omega_dense[0], omega_dense[-1])
ax.grid(True, which="both", alpha=0.25)

# Annotate the key message
ax.annotate("Near-resonance spikes\nas $k^2 \\approx \\lambda_j(\\Delta_h)$",
            xy=(42, kappa_arr[np.nanargmax(kappa_arr[30:60]) + 30]),
            xytext=(60, kappa_arr[np.nanargmax(kappa_arr[30:60]) + 30] * 0.5),
            arrowprops=dict(arrowstyle="->", color="#555"), fontsize=9, color="#333")

plt.tight_layout()
fig.savefig(OUT / "condition_number.png")
plt.close(fig)
print("  saved condition_number.png")

print("\nAll figures saved to:", OUT)
