"""
warm_start_analysis.py — 1D warm-start eigenvalue analysis (three-approach comparison).

Three warm-start strategies compared side-by-side:
  A  raw        — model trained on free-space Green's fn, PML strip NOT zeroed
  B  zero_pml   — same model, PML strip zeroed at inference (post-processing fix)
  C  pml_trained — model trained on FD/PML data with full-grid loss, no zeroing

Plus cold start (x₀=0) and first-order perturbation theory as references.

4-figure thesis package:
  1  Eigenvalue distribution of A_H  (complex plane + condition number)
  2  Spectral transfer function: |β_k/α_k| vs Re(λ_k)
  3  Error spectrum: error per eigenmode for each strategy
  4  GMRES convergence: all strategies (relative residual vs iter)

Usage
-----
  # Minimum (approaches A + B only):
  python warm_start_analysis.py \\
      --omega_l 16 --omega_h 32 \\
      --ckpt_green  runs/pair_16_32/T_up/best.pt

  # Full three-way comparison (adds approach C):
  python warm_start_analysis.py \\
      --omega_l 16 --omega_h 32 \\
      --ckpt_green  runs/pair_16_32/T_up/best.pt \\
      --ckpt_pml    runs/pair_16_32_pml/T_up/best.pt
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from pyamg.krylov import fgmres

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "claude" / "eigenvalue_1d"))

import scipy.sparse as sp
from solver_1d import HelmholtzSolver1D, N, NPML, INT, SIGMA_G
from models_1d import load_checkpoint

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 9, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
C = dict(
    cold      = "#2E6DA4",   # blue
    raw       = "#888888",   # grey   — approach A
    zero_pml  = "#E07B39",   # orange — approach B
    pml_train = "#2ca02c",   # green  — approach C
    pml_int   = "#17becf",   # cyan   — approach E (FD/PML data, interior loss)
    masked    = "#d62728",   # red    — approach D (interior-only trained)
    pert      = "#9467bd",   # purple — perturbation theory
    ref       = "black",
)

N_TEST        = 20    # problems for spectral analysis
N_GMRES_TEST  = 10    # problems for GMRES figure
GMRES_TOL     = 1e-6
GMRES_MAXITER = 200   # raised: weaker precond needs more iterations
GMRES_RESTART = 100
CSL_BETA      = 0.3   # spectral analysis uses this; GMRES uses --gmres_beta
SEED_TEST     = 999
N_INT         = INT.stop - INT.start   # 288 interior points


# ── warm-start helpers ────────────────────────────────────────────────────────

def apply_warm_start(t_up: torch.nn.Module, u_L: np.ndarray,
                     omega_l: float, device=None,
                     mask_pml_input: bool = False) -> np.ndarray:
    """x_0 = T_up(u_L).  Normalise/denormalise by interior RMS.
    mask_pml_input=True: zero PML strip of u_L before feeding (approach D).
    device: inferred from model if None (avoids cross-device mismatches)."""
    dev  = next(t_up.parameters()).device  # always use model's actual device
    u_in = zero_pml_strip(u_L) if mask_pml_input else u_L
    rms  = max(float(np.sqrt(np.mean(np.abs(u_L[INT]) ** 2))), 1e-10)
    inp  = np.stack([u_in.real / rms, u_in.imag / rms], axis=0).astype(np.float32)
    inp_t = torch.from_numpy(inp).unsqueeze(0)
    om_t  = torch.tensor([omega_l], dtype=torch.float32)
    with torch.no_grad():
        out = t_up(inp_t.to(dev), om_t.to(dev)).cpu().numpy()[0]
    return (out[0] + 1j * out[1]) * rms


def zero_pml_strip(x: np.ndarray) -> np.ndarray:
    """Zero PML strip [0:NPML] and [N-NPML:] — in-place copy."""
    x = x.copy()
    x[:NPML]   = 0.0
    x[N-NPML:] = 0.0
    return x


def apply_perturbation_ws(u_L: np.ndarray, A_L_lu, k_L: float,
                          k_H: float) -> np.ndarray:
    """First-order Neumann warm start: x_0 = u_L - (k_H²−k_L²) A_L⁻¹ u_L."""
    dk2 = k_H ** 2 - k_L ** 2
    return u_L - dk2 * A_L_lu.solve(u_L.astype(np.complex128))


# ── spectral projection (interior eigenbasis) ─────────────────────────────────

def project_int(V_int: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Project interior slice of x onto interior eigenbasis.
    V_int is orthonormal (from eigh), so projection = V_int.T @ x[INT]."""
    return V_int.T @ x[INT].astype(np.complex128)


# ── CSL preconditioner ────────────────────────────────────────────────────────

def build_csl_precond(A: sp.spmatrix, omega: float,
                      beta: float = CSL_BETA) -> spla.SuperLU:
    n     = A.shape[0]
    shift = -1j * beta * omega**2
    A_csl = A + shift * sp.eye(n, format="csc", dtype=np.complex128)
    return spla.splu(A_csl)


# ── GMRES helper ──────────────────────────────────────────────────────────────

def run_gmres(A, b, x0, M_lu) -> list[float]:
    """GMRES with optional CSL preconditioner.  M_lu=None → no preconditioning."""
    residuals: list[float] = []
    kw = dict(tol=GMRES_TOL, maxiter=GMRES_MAXITER, restart=GMRES_RESTART,
              residuals=residuals)
    if M_lu is not None:
        kw["M"] = spla.LinearOperator(A.shape, matvec=M_lu.solve, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        fgmres(A, b.astype(np.complex128), x0=x0.astype(np.complex128), **kw)
    return residuals


# ── plotting helpers ──────────────────────────────────────────────────────────

def _pad_and_mean(lst):
    """Pad ragged residual lists with NaN, return (matrix, mean, lo, hi)."""
    L   = max(len(r) for r in lst)
    mat = np.full((len(lst), L), np.nan)
    for i, r in enumerate(lst):
        mat[i, :len(r)] = r
    mean = np.nanmean(mat, axis=0)
    lo   = np.nanpercentile(mat, 25, axis=0)
    hi   = np.nanpercentile(mat, 75, axis=0)
    return mat, mean, lo, hi


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--omega_l",    type=float, default=16.0)
    ap.add_argument("--omega_h",    type=float, default=32.0)
    ap.add_argument("--ckpt_green",  required=True,
                    help="T_up best.pt trained on free-space Green's fn data")
    ap.add_argument("--ckpt_pml",     default=None,
                    help="T_up trained on FD/PML data, full-grid loss, no zeroing (approach C)")
    ap.add_argument("--ckpt_pml_int", default=None,
                    help="T_up trained on FD/PML data, interior-only loss, zeroed at inference (approach E)")
    ap.add_argument("--ckpt_masked",  default=None,
                    help="T_up trained with PML strips zeroed in input+target (approach D)")
    # backward-compat alias
    ap.add_argument("--ckpt",        default=None,
                    help="alias for --ckpt_green (deprecated)")
    ap.add_argument("--device",      default="cpu")
    ap.add_argument("--n_test",      type=int,   default=N_TEST)
    ap.add_argument("--n_gmres",     type=int,   default=N_GMRES_TEST)
    ap.add_argument("--gmres_beta",  type=float, default=0.3,
                    help="CSL β for GMRES preconditioner.  0.3=strong/fast, "
                         "0.05=weak/more-iters (shows warm-start benefit), "
                         "0.0=no preconditioning")
    ap.add_argument("--outdir",
                    default="experiments/claude/eigenvalue_1d/results")
    args = ap.parse_args()

    # backward compat
    if args.ckpt_green is None and args.ckpt is not None:
        args.ckpt_green = args.ckpt

    device = torch.device(args.device)
    tag    = f"pair_{int(args.omega_l)}_{int(args.omega_h)}"
    outdir = ROOT / args.outdir / tag          # results/pair_16_32/
    outdir.mkdir(parents=True, exist_ok=True)
    rng    = np.random.default_rng(SEED_TEST)

    k_L = float(args.omega_l)
    k_H = float(args.omega_h)

    # ── load models ───────────────────────────────────────────────────────────
    print("Loading T_up (green fn) ...")
    t_green, ck_g = load_checkpoint(args.ckpt_green, device=str(device))
    t_green.eval()
    print(f"  epoch {ck_g['epoch']}  val_loss={ck_g['val_loss']:.5f}")

    t_pml = None
    if args.ckpt_pml:
        print("Loading T_up (PML trained) ...")
        t_pml, ck_p = load_checkpoint(args.ckpt_pml, device=str(device))
        t_pml.eval()
        print(f"  epoch {ck_p['epoch']}  val_loss={ck_p['val_loss']:.5f}")

    t_pml_int = None
    if args.ckpt_pml_int:
        print("Loading T_up (FD/PML data, interior-only loss) ...")
        t_pml_int, ck_e = load_checkpoint(args.ckpt_pml_int, device=str(device))
        t_pml_int.eval()
        print(f"  epoch {ck_e['epoch']}  val_loss={ck_e['val_loss']:.5f}")

    t_masked = None
    if args.ckpt_masked:
        print("Loading T_up (PML-masked trained) ...")
        t_masked, ck_m = load_checkpoint(args.ckpt_masked, device=str(device))
        t_masked.eval()
        print(f"  epoch {ck_m['epoch']}  val_loss={ck_m['val_loss']:.5f}")

    active = (["cold", "raw", "zero_pml"]
              + (["pml_int"]     if t_pml_int else [])
              + (["pml_trained"] if t_pml     else [])
              + (["masked"]      if t_masked  else []))
    print(f"\nActive approaches: {active}")

    # ── build FD operators ────────────────────────────────────────────────────
    print(f"\nBuilding A_H (ω={k_H}) and A_L (ω={k_L}), dx=1/(N-1) ...")
    sol_H  = HelmholtzSolver1D(omega=k_H)
    sol_L  = HelmholtzSolver1D(omega=k_L)
    A_H    = sol_H.matrix
    A_L    = sol_L.matrix
    A_L_lu = spla.splu(A_L)

    # ── GMRES preconditioner (separate β from spectral analysis) ─────────────
    if args.gmres_beta > 0.0:
        print(f"Building CSL preconditioner for GMRES (β={args.gmres_beta}) ...")
        M_lu = build_csl_precond(A_H, k_H, beta=args.gmres_beta)
    else:
        print("GMRES: no preconditioning (β=0)")
        M_lu = None

    # ── eigendecompose INTERIOR BLOCK of A_H ─────────────────────────────────
    # The full A_H eigenvectors are ill-conditioned (cond~1e15) because the PML
    # rows introduce near-linear dependence.  The interior block A_H[INT,INT]
    # is a real symmetric tridiagonal matrix → eigh gives orthonormal eigenvectors
    # with cond(V)=1 exactly.  Its eigenvalues are the physically meaningful
    # propagating/evanescent modes of the interior Helmholtz operator.
    print(f"Eigendecomposing interior block A_H[INT,INT] ({N_INT}×{N_INT}, real symmetric) ...")
    t0 = time.time()
    A_H_int = A_H[INT, INT].toarray().real  # real symmetric tridiagonal
    eigs_int, V_int = np.linalg.eigh(A_H_int)   # sorted ascending, orthonormal
    print(f"  done [{time.time()-t0:.1f}s]  "
          f"λ ∈ [{eigs_int.min():.3e}, {eigs_int.max():.3e}]  "
          f"cond(V)=1.0 (orthonormal)")
    near_zero_mask = np.abs(eigs_int) < np.percentile(np.abs(eigs_int), 5)

    # ── generate test problems ────────────────────────────────────────────────
    n_sp   = args.n_test
    n_gm   = args.n_gmres
    n_total = max(n_sp, n_gm)
    print(f"\nGenerating {n_total} test problems (FD/PML exact solutions) ...")

    # spectral data
    all_alpha         = []
    all_beta_raw      = []
    all_beta_zero     = []
    all_beta_pml      = []
    all_beta_pml_int  = []
    all_beta_masked   = []

    # PML energy tracking: ratio ||x_pml||² / ||x_int||² per problem
    def _pml_ratio(x: np.ndarray) -> float:
        pml_nrg = np.sum(np.abs(x[:NPML])**2) + np.sum(np.abs(x[N-NPML:])**2)
        int_nrg = np.sum(np.abs(x[INT])**2) + 1e-30
        return float(pml_nrg / int_nrg)

    pml_ratios: dict[str, list] = {k: [] for k in
        ["target", "raw", "zero_pml", "pml_trained", "pml_int", "masked"]}

    # GMRES histories (one list per approach)
    gmres = {k: [] for k in active}

    for i in range(n_total):
        n_src  = int(rng.integers(3, 7))
        pos    = rng.integers(NPML, N - NPML, size=n_src)
        amps   = rng.uniform(1.0, 2.0, size=n_src)
        phases = rng.uniform(0.0, 2 * np.pi, size=n_src)

        f = sum(sol_H.gaussian_source(p, a, ph)
                for p, a, ph in zip(pos, amps, phases))

        u_L = A_L_lu.solve(f)
        u_H = sol_H.solve(f)

        # warm-start vectors for all approaches
        x_raw     = apply_warm_start(t_green,   u_L, k_L, device)
        x_zero    = zero_pml_strip(x_raw)
        x_pml     = (apply_warm_start(t_pml,     u_L, k_L, device)
                     if t_pml is not None else None)
        # approach E: FD/PML trained, interior-only loss → zero output PML strip
        x_pml_int = (zero_pml_strip(apply_warm_start(t_pml_int, u_L, k_L, device))
                     if t_pml_int is not None else None)
        # approach D: zero input PML before model, then zero output PML strip
        x_masked  = (zero_pml_strip(
                         apply_warm_start(t_masked, u_L, k_L, device,
                                          mask_pml_input=True))
                     if t_masked is not None else None)

        if i < n_sp:
            all_alpha.append(project_int(V_int, u_H))
            all_beta_raw.append(project_int(V_int, x_raw))
            all_beta_zero.append(project_int(V_int, x_zero))
            if x_pml     is not None: all_beta_pml.append(project_int(V_int, x_pml))
            if x_pml_int is not None: all_beta_pml_int.append(project_int(V_int, x_pml_int))
            if x_masked  is not None: all_beta_masked.append(project_int(V_int, x_masked))

        # PML energy for every problem
        pml_ratios["target"].append(_pml_ratio(u_H))
        pml_ratios["raw"].append(_pml_ratio(x_raw))
        pml_ratios["zero_pml"].append(_pml_ratio(x_zero))
        if x_pml     is not None: pml_ratios["pml_trained"].append(_pml_ratio(x_pml))
        if x_pml_int is not None: pml_ratios["pml_int"].append(_pml_ratio(x_pml_int))
        if x_masked  is not None: pml_ratios["masked"].append(_pml_ratio(x_masked))

        if i < n_gm:
            b = f.astype(np.complex128)
            if "cold"        in active: gmres["cold"       ].append(run_gmres(A_H, b, np.zeros(N, dtype=complex), M_lu))
            if "raw"         in active: gmres["raw"        ].append(run_gmres(A_H, b, x_raw,    M_lu))
            if "zero_pml"    in active: gmres["zero_pml"   ].append(run_gmres(A_H, b, x_zero,   M_lu))
            if "pml_int"     in active and x_pml_int is not None: gmres["pml_int"    ].append(run_gmres(A_H, b, x_pml_int, M_lu))
            if "pml_trained" in active and x_pml     is not None: gmres["pml_trained"].append(run_gmres(A_H, b, x_pml,    M_lu))
            if "masked"      in active and x_masked  is not None: gmres["masked"     ].append(run_gmres(A_H, b, x_masked, M_lu))

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{n_total}")

    # ── spectral statistics ───────────────────────────────────────────────────
    alpha       = np.array(all_alpha)
    beta_raw    = np.array(all_beta_raw)
    beta_zero   = np.array(all_beta_zero)
    beta_pml     = np.array(all_beta_pml)     if all_beta_pml     else None
    beta_pml_int = np.array(all_beta_pml_int) if all_beta_pml_int else None
    beta_masked  = np.array(all_beta_masked)  if all_beta_masked  else None

    eps = 1e-30
    tf_raw     = np.abs(beta_raw)     / (np.abs(alpha) + eps)
    tf_zero    = np.abs(beta_zero)    / (np.abs(alpha) + eps)
    tf_pml     = np.abs(beta_pml)     / (np.abs(alpha) + eps) if beta_pml     is not None else None
    tf_pml_int = np.abs(beta_pml_int) / (np.abs(alpha) + eps) if beta_pml_int is not None else None
    tf_masked  = np.abs(beta_masked)  / (np.abs(alpha) + eps) if beta_masked  is not None else None

    err_cold    = np.abs(alpha)
    err_raw     = np.abs(alpha - beta_raw)
    err_zero    = np.abs(alpha - beta_zero)
    err_pml     = np.abs(alpha - beta_pml)     if beta_pml     is not None else None
    err_pml_int = np.abs(alpha - beta_pml_int) if beta_pml_int is not None else None
    err_masked  = np.abs(alpha - beta_masked)  if beta_masked  is not None else None

    def _stats(arr):
        return (np.median(arr, axis=0),
                np.percentile(arr, 25, axis=0),
                np.percentile(arr, 75, axis=0))

    tf_raw_med,  tf_raw_lo,  tf_raw_hi  = _stats(tf_raw)
    tf_zero_med, tf_zero_lo, tf_zero_hi = _stats(tf_zero)
    tf_pml_med     = _stats(tf_pml)[0]     if tf_pml     is not None else None
    tf_pml_int_med = _stats(tf_pml_int)[0] if tf_pml_int is not None else None
    tf_masked_med  = _stats(tf_masked)[0]  if tf_masked  is not None else None

    err_cold_med,  err_cold_lo,  err_cold_hi  = _stats(err_cold)
    err_raw_med,   err_raw_lo,   err_raw_hi   = _stats(err_raw)
    err_zero_med,  err_zero_lo,  err_zero_hi  = _stats(err_zero)
    err_pml_med    = _stats(err_pml)[0]     if err_pml     is not None else None
    err_pml_int_med= _stats(err_pml_int)[0] if err_pml_int is not None else None
    err_masked_med = _stats(err_masked)[0]  if err_masked  is not None else None

    # ── GMRES statistics ──────────────────────────────────────────────────────
    gstats = {}
    for k, lst in gmres.items():
        if lst:
            mat, mean, lo, hi = _pad_and_mean(lst)
            gstats[k] = dict(mat=mat, mean=mean, lo=lo, hi=hi,
                             iters=[len(r) for r in lst])

    # ── FIGURES ───────────────────────────────────────────────────────────────
    tag_title = (f"1D Helmholtz  N={N}, $n_{{\\rm pml}}$={NPML}, "
                 f"$\\omega_L$={int(k_L)}, $\\omega_H$={int(k_H)}")
    modes  = np.arange(N_INT)
    lam_re = eigs_int          # real eigenvalues of interior block, sorted ascending

    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax3, ax4 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # Fig 1: interior eigenvalue distribution + solution energy ───────────────
    # Interior block is real symmetric → eigenvalues are real, physically meaningful.
    alpha_energy = np.mean(np.abs(alpha) ** 2, axis=0)   # mean |alpha_k|² per mode
    ax1b = ax1.twinx()
    ax1.plot(lam_re, color=C["cold"], lw=1.5, label=r"$\lambda_k$ (interior block)")
    ax1.axhline(0, color=C["ref"], lw=0.9, ls="--", alpha=0.6)
    ax1b.semilogy(alpha_energy, color="#E07B39", lw=1.2, alpha=0.8,
                  label=r"$\langle|\alpha_k|^2\rangle$ (solution energy)")
    ax1.set_xlabel("Interior eigenmode index $k$ (sorted by $\\lambda_k$)")
    ax1.set_ylabel(r"$\lambda_k$  [interior FD operator]", color=C["cold"])
    ax1b.set_ylabel(r"Solution energy $\langle|\alpha_k|^2\rangle$", color="#E07B39")
    ax1.set_title(r"Interior eigenvalues + solution energy of $A_H^{\rm int}$")
    # mark near-zero region
    nz_idx = np.where(near_zero_mask)[0]
    if len(nz_idx):
        ax1.axvspan(nz_idx[0], nz_idx[-1], color="purple", alpha=0.12,
                    label=f"near-zero ({len(nz_idx)} modes)")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="lower right")
    neg_int = (eigs_int < 0).sum()
    kappa_int = abs(eigs_int.max() / eigs_int.min()) if eigs_int.min() != 0 else np.inf
    ax1.text(0.02, 0.97,
             f"λ<0: {neg_int}/{N_INT}  κ≈{kappa_int:.2e}\ncond(V)=1 (orthonormal)",
             transform=ax1.transAxes, va="top", ha="left", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    # Fig 2: spectral transfer functions ──────────────────────────────────────
    ax2.fill_between(lam_re, tf_raw_lo,  tf_raw_hi,
                     color=C["raw"], alpha=0.2)
    ax2.plot(lam_re, tf_raw_med,  color=C["raw"],      lw=1.8,
             label="A: raw (green fn, no zeroing)")
    ax2.fill_between(lam_re, tf_zero_lo, tf_zero_hi,
                     color=C["zero_pml"], alpha=0.2)
    ax2.plot(lam_re, tf_zero_med, color=C["zero_pml"], lw=1.8, ls="-.",
             label="B: zero-PML (green fn + zeroed)")
    if tf_pml_int_med is not None:
        ax2.plot(lam_re, tf_pml_int_med, color=C["pml_int"], lw=1.8, ls="-.",
                 label="E: FD/PML trained, int-loss + zeroed")
    if tf_pml_med is not None:
        ax2.plot(lam_re, tf_pml_med, color=C["pml_train"], lw=1.8, ls="--",
                 label="C: PML-trained, full-grid loss (no zeroing)")
    if tf_masked_med is not None:
        ax2.plot(lam_re, tf_masked_med, color=C["masked"], lw=1.8, ls=":",
                 label="D: interior-only trained + zeroed")
    ax2.axhline(1.0, color=C["ref"], lw=0.9, ls=":", alpha=0.7,
                label="ideal (ratio=1)")
    ax2.axvline(0, color=C["ref"], lw=0.6, ls="--", alpha=0.4)
    # auto-scale y to the actual data range with small margin
    _all_tf = [tf_raw_med, tf_zero_med]
    if tf_pml_int_med  is not None: _all_tf.append(tf_pml_int_med)
    if tf_pml_med      is not None: _all_tf.append(tf_pml_med)
    if tf_masked_med   is not None: _all_tf.append(tf_masked_med)
    _tf_stack = np.concatenate([t.ravel() for t in _all_tf])
    _tf_lo = max(0.0, np.percentile(_tf_stack, 1) - 0.05)
    _tf_hi = np.percentile(_tf_stack, 99) + 0.05
    ax2.set_ylim(_tf_lo, _tf_hi)
    ax2.set_xlabel(r"Re($\lambda_k$)")
    ax2.set_ylabel(r"$|\beta_k / \alpha_k|$")
    ax2.set_title(r"Spectral transfer function  $|\beta_k / \alpha_k|$")
    ax2.legend(loc="upper left", fontsize=8)

    # Fig 3: error spectrum ────────────────────────────────────────────────────
    ax3.fill_between(modes, err_cold_lo, err_cold_hi, color=C["cold"], alpha=0.12)
    ax3.fill_between(modes, err_raw_lo,  err_raw_hi,  color=C["raw"],  alpha=0.15)
    ax3.fill_between(modes, err_zero_lo, err_zero_hi, color=C["zero_pml"], alpha=0.15)
    ax3.semilogy(modes, err_cold_med, color=C["cold"],     lw=1.8,
                 label="Cold start  $|\\alpha_k|$")
    ax3.semilogy(modes, err_raw_med,  color=C["raw"],      lw=1.5,
                 label="A: raw  $|\\alpha_k - \\beta_k|$")
    ax3.semilogy(modes, err_zero_med, color=C["zero_pml"],  lw=1.5, ls="-.",
                 label="B: zero-PML")
    if err_pml_int_med is not None:
        ax3.semilogy(modes, err_pml_int_med, color=C["pml_int"], lw=1.5, ls="-.",
                     label="E: FD/PML trained, int-loss + zeroed")
    if err_pml_med is not None:
        ax3.semilogy(modes, err_pml_med, color=C["pml_train"], lw=1.5, ls="--",
                     label="C: PML-trained, full-grid loss")
    if err_masked_med is not None:
        ax3.semilogy(modes, err_masked_med, color=C["masked"], lw=1.5, ls=":",
                     label="D: interior-only trained + zeroed")

    if near_zero_mask.any():
        lo_m, hi_m = modes[near_zero_mask].min(), modes[near_zero_mask].max()
        ax3.axvspan(lo_m, hi_m, color="purple", alpha=0.08,
                    label=r"$|\lambda_k|$ bottom 5% (propagating)")

    ax3.set_xlabel("Eigenmode index (sorted by Re($\\lambda$))")
    ax3.set_ylabel(r"$|c_k|$  (eigenbasis amplitude)")
    ax3.set_title("Error spectrum: cold vs warm start approaches")
    ax3.legend(loc="upper right", fontsize=8)

    # Fig 4: GMRES convergence ─────────────────────────────────────────────────
    label_map = {
        "cold":        ("Cold start  x₀=0",                      C["cold"],      "-",  2.4),
        "raw":         ("A: raw (green fn, no zeroing)",          C["raw"],       "-",  1.8),
        "zero_pml":    ("B: zero-PML  (green fn + zeroed)",       C["zero_pml"],  "-.", 1.8),
        "pml_int":     ("E: FD/PML data, int-loss + zeroed",      C["pml_int"],   "-.", 1.8),
        "pml_trained": ("C: PML-trained, full-grid loss",         C["pml_train"], "--", 1.8),
        "masked":      ("D: interior-only trained + zeroed",      C["masked"],    ":",  1.8),
    }
    for k, info in gstats.items():
        label, col, ls, lw = label_map[k]
        mat = info["mat"]
        for j in range(mat.shape[0]):
            row = mat[j]
            iters = np.arange(len(row[~np.isnan(row)]))
            ax4.semilogy(np.arange(len(row)), row, color=col, alpha=0.2, lw=0.7)
        ax4.semilogy(np.arange(len(info["mean"])), info["mean"],
                     color=col, lw=lw, ls=ls, label=label)

    ax4.axhline(GMRES_TOL, color=C["ref"], lw=0.9, ls=":", alpha=0.6,
                label=f"tol={GMRES_TOL:.0e}")
    ax4.set_xlabel("GMRES iteration")
    ax4.set_ylabel("Relative residual")
    precond_label = (f"CSL β={args.gmres_beta}" if args.gmres_beta > 0
                     else "no preconditioning")
    ax4.set_title(f"GMRES convergence ({precond_label})")
    ax4.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"1D warm-start comparison — $T_{{\\rm up}}$  "
        f"($\\omega_L={int(k_L)} \\to \\omega_H={int(k_H)}$)\n{tag_title}",
        fontsize=12, fontweight="bold")

    for ext in ("png", "pdf"):
        p = outdir / f"warm_start_{tag}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig)

    # ── Fig 5: PML strip energy bar chart ─────────────────────────────────────
    pml_order = [
        ("target",      "True $u_H$",                    C["ref"],       "//"),
        ("raw",         "A: raw",                        C["raw"],       ""),
        ("zero_pml",    "B: zero-PML",                   C["zero_pml"],  ""),
        ("pml_trained", "C: PML-trained",                C["pml_train"], ""),
        ("pml_int",     "E: FD/PML int-loss + zeroed",   C["pml_int"],   ""),
        ("masked",      "D: interior-only + zeroed",     C["masked"],    ""),
    ]
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    bar_labels, bar_means, bar_stds, bar_colors, bar_hatches = [], [], [], [], []
    for key, lbl, col, hatch in pml_order:
        vals = pml_ratios.get(key, [])
        if not vals:
            continue
        bar_labels.append(lbl)
        bar_means.append(float(np.mean(vals)))
        bar_stds.append(float(np.std(vals)))
        bar_colors.append(col)
        bar_hatches.append(hatch)
    xs = np.arange(len(bar_labels))
    bars = ax5.bar(xs, bar_means, yerr=bar_stds, capsize=4,
                   color=bar_colors, edgecolor="black", linewidth=0.7,
                   error_kw=dict(elinewidth=1.2, ecolor="black"))
    for bar, hatch in zip(bars, bar_hatches):
        bar.set_hatch(hatch)
    ax5.set_xticks(xs)
    ax5.set_xticklabels(bar_labels, rotation=20, ha="right", fontsize=9)
    ax5.set_ylabel(r"$\|x_0[\mathrm{PML}]\|^2 \;/\; \|x_0[\mathrm{int}]\|^2$")
    ax5.set_title(
        f"PML strip energy ratio — $\\omega_L={int(k_L)} \\to \\omega_H={int(k_H)}$\n"
        r"(lower = less contamination in absorbing boundary)")
    ax5.axhline(0, color="black", lw=0.6)
    fig5.tight_layout()
    for ext in ("png", "pdf"):
        p = outdir / f"pml_energy_{tag}.{ext}"
        fig5.savefig(p, bbox_inches="tight")
        print(f"Saved → {p}")
    plt.close(fig5)

    # ── console + file summary ────────────────────────────────────────────────
    import io, contextlib
    _summary_buf = io.StringIO()

    def _pr(*a, **kw):
        print(*a, **kw)                          # terminal
        print(*a, **kw, file=_summary_buf)       # buffer → file

    near_mask = near_zero_mask
    _pr(f"\n{'='*65}")
    _pr(f"Warm-start comparison  ω_L={int(k_L)} → ω_H={int(k_H)}")
    _pr(f"{'='*65}")
    _pr(f"  Interior eigenvalue range:  [{eigs_int.min():.3e}, {eigs_int.max():.3e}]")
    _pr(f"  Interior κ ≈ {kappa_int:.3e}  |  near-zero modes: {near_zero_mask.sum()}/{N_INT}")
    _pr(f"  GMRES preconditioner:  β={args.gmres_beta} "
        f"({'strong' if args.gmres_beta >= 0.2 else 'weak' if args.gmres_beta > 0 else 'none'})")
    _pr(f"{'─'*65}")
    _pr(f"  {'Approach':<22}  {'TF (near-zero)':>16}  {'GMRES iters':>14}")
    _pr(f"{'─'*65}")

    rows = [
        ("raw",         beta_raw,     gmres.get("raw",         [])),
        ("zero_pml",    beta_zero,    gmres.get("zero_pml",    [])),
    ]
    if beta_pml_int is not None: rows.append(("pml_int",     beta_pml_int, gmres.get("pml_int",     [])))
    if beta_pml     is not None: rows.append(("pml_trained", beta_pml,     gmres.get("pml_trained", [])))
    if beta_masked  is not None: rows.append(("masked",      beta_masked,  gmres.get("masked",      [])))

    for name, betas, g_list in rows:
        betas_arr = np.array(betas)
        tf = np.abs(betas_arr) / (np.abs(alpha) + eps)
        tf_near = np.median(tf[:, near_mask])
        if g_list:
            iters = [len(r) for r in g_list]
            gstr = f"{np.mean(iters):.1f} ± {np.std(iters):.1f}"
        else:
            gstr = "—"
        _pr(f"  {name:<22}  {tf_near:>16.3f}  {gstr:>14}")

    if "cold" in gstats:
        iters_c = gstats["cold"]["iters"]
        _pr(f"  {'cold (baseline)':<22}  {'—':>16}  "
            f"{np.mean(iters_c):.1f} ± {np.std(iters_c):.1f}")

    _pr(f"{'─'*65}")
    for name, _, g_list in rows:
        if g_list and gstats.get("cold"):
            speedup = np.mean(gstats["cold"]["iters"]) / np.mean([len(r) for r in g_list])
            _pr(f"  speedup vs cold  [{name}]:  {speedup:.2f}×")
    _pr(f"{'='*65}")
    _pr(f"Figures → {outdir}/")

    summary_path = outdir / "summary.txt"
    summary_path.write_text(_summary_buf.getvalue())
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
