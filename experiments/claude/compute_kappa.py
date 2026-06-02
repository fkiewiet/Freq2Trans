"""
compute_kappa.py
────────────────
Condition number κ(A(ω)) = σ_max / σ_min for the Helmholtz FD matrix.

Physical setup
──────────────
  Domain:  [0,1]²   (unit square, L=1)
  Grid:    N × N,   h = 1/N
  PDE:     (Δ + k²) u = f,   k = ω  (wavenumber in rad per unit length)
  PML:     n_pml cells, quadratic σ profile, σ₀ is the peak damping

  With h = 1/N, the diagonal in the interior is:
      d = k² - 4/h² = ω² - 4N²
  For ω ∈ {16,32,64,128} and N ≥ 32, we have 4N² >> ω²:
      → Laplacian dominates, matrix is indefinite (eigenvalues of both signs)
      → σ_max ≈ 4/h² = 4N²  (analytical, from discrete Laplacian)
      → σ_min depends on PML strength σ₀/ω  (smaller at high ω → larger κ)

Why σ_min shrinks with ω (constant σ₀)
──────────────────────────────────────
  PML stretching s = 1 + i·σ/ω.  Relative damping = σ₀/ω.
  Constant σ₀=85:  at ω=16 → σ₀/ω=5.3 (strong),  at ω=128 → σ₀/ω=0.66 (weak).
  Weak PML → near-resonance modes barely shifted into complex plane → small |λ| → small σ_min.
  Adaptive σ₀∝ω^0.694 keeps σ₀/ω larger at high ω → better conditioning.

Usage
─────
  cd ~/Freq2Transfer && source .venv/bin/activate
  python experiments/claude/compute_kappa.py            # N=64, ~30 s
  python experiments/claude/compute_kappa.py --N 128   # N=128, ~5 min
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=64, help="grid size")
args   = parser.parse_args()

N      = args.N
n_pml  = max(4, round(112 / 512 * N))   # proportional to thesis (112/512)
dx     = 1.0 / N                         # unit physical domain [0,1]²
omegas = [16, 32, 64, 128]

SIGMA_CONST = 85.0
sigma_adapt = lambda w: 6.203 * float(w) ** 0.694

print(f"Grid: {N}×{N}   n_pml={n_pml}   h=dx=1/{N}={dx:.5f}")
print(f"4/h² = {4/dx**2:.3e}   (Laplacian dominates over k²=ω²  ✓)")
print(f"Interior diagonal: k²-4/h² = ω²-{4/dx**2:.0f}")
print(f"  ω=16:  {16**2 - 4/dx**2:.0f}  (negative ✓)")
print(f"  ω=128: {128**2 - 4/dx**2:.0f}  (negative ✓)")
print(f"Adaptive σ₀: { {w: round(sigma_adapt(w),2) for w in omegas} }")
print()


# ── matrix assembly ───────────────────────────────────────────────────────────

def build_helmholtz(N, n_pml, omega, sigma0, dx):
    k = float(omega)
    sigma = np.zeros(N)
    i_pml = np.arange(n_pml)
    vals  = sigma0 * ((n_pml - i_pml) / n_pml) ** 2
    sigma[:n_pml]     = vals
    sigma[N - n_pml:] = vals[::-1]

    s  = 1.0 + 1j * sigma / omega   # [N]
    ii, jj = np.mgrid[0:N, 0:N]
    p      = (ii * N + jj).ravel()
    ax     = (1.0 / (s[jj] * dx**2)).ravel()
    ay     = (1.0 / (s[ii] * dx**2)).ravel()
    v_d    = -2.0 * ax - 2.0 * ay + k**2

    m_r = (jj < N-1).ravel(); m_l = (jj > 0).ravel()
    m_b = (ii < N-1).ravel(); m_u = (ii > 0).ravel()

    rows = np.concatenate([p,   p[m_r], p[m_l], p[m_b], p[m_u]])
    cols = np.concatenate([p,   p[m_r]+1, p[m_l]-1, p[m_b]+N, p[m_u]-N])
    data = np.concatenate([v_d, ax[m_r], ax[m_l], ay[m_b], ay[m_u]])
    return sp.coo_matrix((data, (rows, cols)), shape=(N*N, N*N)).tocsc()


# ── condition number ──────────────────────────────────────────────────────────

def estimate_kappa(A):
    n   = A.shape[0]
    svm = spla.svds(A, k=1, which="LM", return_singular_vectors=False, tol=1e-5)[0]
    lu  = spla.splu(A)
    Ai  = spla.LinearOperator(
            (n, n),
            matvec =lambda x: lu.solve(np.asarray(x, dtype=np.complex128)),
            rmatvec=lambda x: lu.solve(np.asarray(x, dtype=np.complex128), trans="H"),
            dtype=np.complex128)
    svm_i = spla.svds(Ai, k=1, which="LM", return_singular_vectors=False, tol=1e-5)[0]
    return float(svm), float(1.0 / svm_i)


# ── main loop ─────────────────────────────────────────────────────────────────

results = {}
hdr = f"{'ω':>5}  {'σ₀_c':>7}  {'σ₀_a':>8}  {'σ₀_c/ω':>7}  {'σ₀_a/ω':>7}  {'κ_const':>13}  {'κ_adapt':>13}  {'t(s)':>6}"
print(hdr); print("─" * len(hdr))

for om in omegas:
    sa = sigma_adapt(om)
    t0 = time.time()
    A_c = build_helmholtz(N, n_pml, om, SIGMA_CONST, dx)
    A_a = build_helmholtz(N, n_pml, om, sa, dx)
    smax_c, smin_c = estimate_kappa(A_c)
    smax_a, smin_a = estimate_kappa(A_a)
    kc = smax_c / smin_c
    ka = smax_a / smin_a
    elapsed = time.time() - t0

    results[om] = dict(sigma_adapt=round(sa, 4),
                       sigma_max_const=smax_c, sigma_min_const=smin_c,
                       sigma_max_adapt=smax_a, sigma_min_adapt=smin_a,
                       kappa_const=kc, kappa_adapt=ka)
    print(f"{om:>5}  {SIGMA_CONST:>7.1f}  {sa:>8.2f}  "
          f"{SIGMA_CONST/om:>7.2f}  {sa/om:>7.2f}  "
          f"{kc:>13.4e}  {ka:>13.4e}  {elapsed:>6.1f}")

# ── save ──────────────────────────────────────────────────────────────────────
out = Path(__file__).parent / "kappa_results" / f"kappa_N{N}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved → {out}")

kc_arr = [results[w]["kappa_const"] for w in omegas]
ka_arr = [results[w]["kappa_adapt"] for w in omegas]
print()
print("# ── paste into kappa.py ──────────────────────────────────────────")
print(f"# N={N}, h=1/{N}, n_pml={n_pml}")
print(f"kappa_const = np.array([{', '.join(f'{v:.4e}' for v in kc_arr)}])")
print(f"kappa_adapt = np.array([{', '.join(f'{v:.4e}' for v in ka_arr)}])")
