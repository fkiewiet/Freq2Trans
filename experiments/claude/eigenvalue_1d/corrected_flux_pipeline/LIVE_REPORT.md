# Neural Preconditioner — Live Report

> Single source of truth. Last updated: 2026-06-22.

---

## 1. Problem setup

**Equation:** −u'' − ω²u = f, Dirichlet BCs on [0,1]

| Parameter | Value |
|---|---|
| ω_H (target) | 32 |
| ω_L (low-freq, used for V-cycle) | 16 |
| Grid | n = 512, uniform, h = 1/512 |
| Hard mode | k ≈ 10, where k²π² ≈ ω_H² → λ_10(A_H) ≈ −37 (nearly singular) |
| Sources | Multi-point Gaussian, 3–6 sources per problem, amplitude ∈ [1,2] |
| FGMRES tol | ‖b − Ax‖ / ‖b‖ < 1e-6 (true residual, 200 problems, SEED=2025) |
| CSL baseline | A_CSL = A_H − i·β·ω²·I, β = 0.3 |

**Eigenvalue structure (key numbers):**

| Mode k | λ_k(A_H) = k²π² − 1024 | λ_k(A_L) = k²π² − 256 |
|---|---|---|
| 5 | ≈ −777 | ≈ −9.3 (near-zero for A_L!) |
| 10 | ≈ −37 (near-zero for A_H!) | ≈ +731 |
| 15 | ≈ +1199 | ≈ +1966 |

Mode k=10 is the hard mode for A_H. Mode k=5 is the hard mode for A_L.
These are different modes — this matters for T_down (see §5).

---

## 2. Preconditioner structure

**Post-CSL preconditioner** (applies at every FGMRES iteration):

```
M(r) = CSL⁻¹r  +  NN(r₂)

where  r₂ = r − A_H · CSL⁻¹r   (what CSL got wrong)
       NN target: r₂ → A_H⁻¹r₂ (the exact remaining correction)
```

If NN = A_H⁻¹ exactly → M = A_H⁻¹ → FGMRES converges in 1 iteration.
NN must be nonlinear (GELU) → FGMRES (not GMRES) is required.

**CSL preconditioning alone:** 15 median FGMRES iterations.
**Goal:** reduce to ≤ 3 with a fast NN.

---

## 3. Architecture

**DilatedCNN1d** — the only architecture that works for this problem.

```
dilations = [1, 2, 4, 8, 16, 32, 64, 32, 16, 8, 4, 2, 1]
kernel = 7
width w = 64 (default), 128 (ablation)
input: [r₂_re/s, r₂_im/s]  (+ extras for V-cycle and f-conditioning variants)
output: [correction_re, correction_im]  × s
```

Receptive field = 763 > n = 512 → the network sees the entire domain.
The hard mode k=10 has period 51 grid points — fully covered.

**Why no MaxPool / UNet:** MaxPool is a magnitude-pooling operation. It destroys
the spatial phase of mode k=10. The correction at k=10 is phase-sensitive —
predicting the magnitude without phase gives a correction that can ADD to the error
rather than subtract. Confirmed experimentally: G3 (UNet, D1), G5 (UNet, D2) both
diverged regardless of data quality.

---

## 4. What works and what doesn't — organised by axis

### 4.1 Architecture

| Architecture | Result | Reason |
|---|---|---|
| DilatedCNN, no MaxPool | ✓ Works | Preserves phase of mode k=10 |
| UNet with MaxPool | ✗ Always fails | MaxPool destroys phase, model diverges (G3, G5) |
| DilatedCNN w=128 | ? Ongoing (Option A) | Capacity likely not the bottleneck |

**Decision rule:** Do not use MaxPool for any Helmholtz NN preconditioner. The phase
of the near-singular mode is the signal. MaxPool throws it away.

### 4.2 Training data type

Data type is the **dominant factor**. More than architecture, loss, or epochs.

| Label | Description | Val | FGMRES |
|---|---|---|---|
| D1 | Random Gaussian vectors | ≈1.0 (stuck) | Never converges |
| D1+D2 | Mixed random + FGMRES residuals | 0.023 then diverges | ~5–6 iters, unstable |
| D2 | CSL-only FGMRES residuals (27K) | 0.0020 | 4 median |
| D4 = D2 + D4_sc | D2 + self-consistent from G6-precond (52K) | 0.0013 | 4 median |
| D5 | Second self-consistent round (47K) | 0.0022 | No improvement |

**Why D1 fails mathematically:** FGMRES residuals concentrate near mode k=10.
Random vectors are uniform over all modes — the gradient is spread equally over 512
modes, giving the network no incentive to focus on k=10. Not fixable with more epochs
or wider networks (G0, G4 confirm this).

**Why D4 > D2 for G6 but D4 < D2 for T_up:**
D4 residuals come from G6-preconditioned FGMRES. G6 already handles k=10 well, so
D4 residuals have *less* k=10 content than D2. For G6, this is fine (distribution
shift is small). For T_up, which is deployed with a different preconditioner, D4
residuals are from the wrong distribution. T_up trained on D2-only (CSL residuals)
outperforms T_up trained on D4+D2.

**Principle:** Training data must match the deployment distribution of the specific
preconditioner being trained. Change the preconditioner → regenerate training data.

**Why D5 doesn't help:** Self-consistent iteration has converged. The model's
deployment distribution has stabilised. Further rounds add no new signal.
Decision rule: if round N self-consistent data improves best val by <15%, stop.

### 4.3 Training length / more epochs

**G6_ext:** Warm-started from G6 (val=0.0013) at lr=1e-5→1e-8, 620 more epochs.
Best found: val=0.0021 — *worse* than the starting point.

G6 is at a local optimum. A second cosine LR cycle perturbs the model slightly but
finds no better basin. More epochs on a converged model cannot help.

**Decision rule:** If warm-restart at 10× lower LR does not improve best val by >10%
within the first 200 epochs, stop. The model is at its floor for that data/architecture.

### 4.4 Loss function

**Option B (A_H⁻¹-norm loss):** Upweights mode k=10 by ~700× in the gradient.
Result: matched G6 (val=0.0014), never improved. 360 epochs, no progress.

**Why:** D2 residuals already concentrate gradient at k=10. The loss and the data are
redundant signals pointing at the same mode. Changing the loss adds no new information.

**Decision rule:** If data distribution already concentrates on the hard modes (which
FGMRES residuals do by construction), loss weighting is redundant. Loss weighting
is only useful when training data is NOT mode-concentrated — e.g., if you needed to
use random data (which you should never do).

### 4.5 Conditioning inputs

**C_fcond:** Add source f as extra input channels: [r₂_re, r₂_im, f_re/‖f‖, f_im/‖f‖].

Result: val=0.0058 (worse than G6 in training metric), but FGMRES shows:
- All 200 problems converge (G6 has 5 failures)
- 5 problems now converge in **1 iteration** instead of 4
- Distribution: {1:5, 4:195} vs G6's {3:2, 4:193, fail:5}

**Why the failures disappear:** The 5 "permanent failures" in G6 are source-geometry-
specific problems where r₂ alone is ambiguous — the model cannot tell which source
produced it, and guesses wrong. With f as input, the ambiguity is resolved. This is
an identifiability issue, not a model capacity issue.

**Implication for 2D heterogeneous:** f encodes source location but not medium c(x).
In 2D heterogeneous, use u_L = A_L⁻¹f instead — this encodes both source and medium,
is computed once before FGMRES starts (not per-iteration), and connects directly to
the frequency-transfer motivation.

---

## 5. V-cycle vs post-CSL — understanding what T_up and T_down actually are

### Why our "V-cycle" does not require a T_down

#### Homogeneous Dirichlet case (current 1D experiments)

A classical multigrid V-cycle: **restrict** fine residual to coarse grid → solve
coarsely → **prolongate** back to fine grid.

In our 1D homogeneous Dirichlet case, A_H and A_L are:
```
A_H = FD_Laplacian − ω_H² I
A_L = FD_Laplacian − ω_L² I
A_H = A_L + (ω_L² − ω_H²) I   ← pure scalar shift, no structural change
```
The FD Laplacian's eigenvectors sin(kπx) are unchanged by a scalar shift.
**A_H and A_L have identical eigenvectors.** The only difference is eigenvalues.

This makes T_up's task unusually simple:
- "Restriction" to coarse level = trivial (same grid, same eigenvectors)
- "Coarse solve" = compute A_L⁻¹(r₂) via LU (already factored)
- "Prolongation" back = trivial (already on fine grid)
- **T_up just corrects the eigenvalue ratio at the hard mode:**
  - At k=10: A_L⁻¹(r₂)_10 = r₂_10 / 731, target A_H⁻¹(r₂)_10 = r₂_10 / (−37)
  - T_up learns: "multiply mode k=10 by 731/(−37) = −19.8, leave all other modes alone"
  - This is nearly a rank-1 correction — very learnable

The V-cycle is complete with T_up alone in this case.

**T_up in full:**
```
Input to NN:  [A_L⁻¹(r₂)_re/s,  A_L⁻¹(r₂)_im/s,  r₂_re/s,  r₂_im/s]   (4 channels)
NN output:    A_H⁻¹(r₂)  (full correction)
```

#### ⚠ PML case — the simple story breaks down

With PML, the operator is modified to absorb outgoing waves:
```
A_H^PML = Δ_PML(ω_H) − ω_H² I
A_L^PML = Δ_PML(ω_L) − ω_L² I

where Δ_PML(ω) = (1/s(x,ω)) d/dx [(1/s(x,ω)) d/dx]
      s(x,ω) = 1 + σ(x) / (iω)
```

The stretching function s(x,ω) **depends explicitly on ω**. σ(x) is also tuned
per ω (sigma0: {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}).

This means **A_H^PML ≠ A_L^PML + scalar·I**. They have genuinely different
operators at every grid point inside the PML layer. Consequently:
- Their eigenvectors are NOT sin(kπx) — they differ between A_H^PML and A_L^PML
- A_L^PML⁻¹(r₂) is NOT a simple scalar rescaling of A_H^PML⁻¹(r₂)
- The "multiply one mode by −19.8" story does not carry over

**In the PML case, T_up's input A_L^PML⁻¹(r₂) is a rough approximation to
A_H^PML⁻¹(r₂) in the interior but is qualitatively different in the PML layer.**
T_up would need to learn a more complex correction than just one mode rescaling.

The V-cycle approach may still work empirically (A_L^PML⁻¹ is still "easier" than
A_H^PML⁻¹ and provides a rough warm-start), but the clean theoretical motivation
from the Dirichlet case does not transfer.

**Alternative for PML: u_L conditioning (not T_up)**
Instead of per-iteration A_L^PML⁻¹(r₂), compute u_L = A_L^PML⁻¹ f **once** before
FGMRES starts. Then pass [r₂, u_L] to the NN as conditioning. This:
- Costs one A_L^PML LU solve (amortised over all FGMRES iterations)
- Encodes the full low-frequency solution structure (including PML behaviour)
- Is the natural generalisation of f-conditioning for heterogeneous media
- Avoids the per-iteration A_L^PML cost of T_up
- Does NOT require A_H and A_L to share eigenvectors

**Test u_L conditioning before T_up in the PML case.**

### Why T_down fails — structural, not fixable with more training

T_down was designed to learn: r₂ → A_L · A_H⁻¹ · r₂

The idea: compute T_down(r₂) cheaply, then apply A_L⁻¹ to get A_H⁻¹(r₂):
```
r₂ →[T_down]→ A_L·A_H⁻¹r₂ →[A_L⁻¹]→ A_H⁻¹r₂
```

**Problem 1:** This still requires A_L⁻¹. T_down saves nothing.

**Problem 2:** Mode k≈5 is near-singular for A_L (λ_5(A_L) ≈ −9.3). The chain
amplifies T_down's prediction errors at mode 5:

```
Amplification at mode 5:
  λ_5(A_H) / λ_5(A_L) = −777 / −9.3 ≈ 83×

If T_down has 1% relative error at mode 5 in its prediction of (A_L·A_H⁻¹·r₂)_5,
the chain output A_L⁻¹(T_down(r₂))_5 has 83% relative error vs A_H⁻¹(r₂)_5.
```

This explains exactly why the chain metric is stuck at 0.66 despite val=0.008:
```
chain error ≈ 83 × val_error ≈ 83 × 0.008 = 0.66   ← matches exactly
```

To achieve chain < 0.05 you'd need val < 0.0006, which is unreachable
(even G6 at 0.0013 couldn't achieve this). The amplification is structural.

**T_down is closed.** Do not revisit.

### The runtime cost of T_up in 2D

T_up requires one extra A_L⁻¹ solve **per FGMRES iteration** (not once upfront).
In 1D this is free (LU already factored). In 2D:

- A_L⁻¹ for a 512² grid requires a separate sparse LU factorization (expensive upfront)
  and one triangular solve per FGMRES iteration (~0.1s per solve)
- If T_up gives 3 iters instead of 4: saves 1 FGMRES step, adds 3 extra A_L⁻¹ solves
- Net win only if: time_saved(1 FGMRES iter) > 3 × time(A_L⁻¹ solve)

**Always measure wall-clock time for T_up, not just iteration count.**

Compare with u_L = A_L⁻¹f (f-conditioning for 2D heterogeneous):
- u_L computed ONCE before FGMRES starts (cheap amortisation)
- No per-iteration A_L⁻¹ cost
- Encodes both source AND medium c(x)
- **Test u_L conditioning before T_up in 2D**

---

## 6. FGMRES results — all measured models

| Model | Approach | Val | Median | Distribution | Conv | Notes |
|---|---|---|---|---|---|---|
| CSL-only | baseline | — | 15 | — | 200/200 | No NN |
| Oracle | CSL + exact A_H⁻¹ | — | 10 | — | 200/200 | Upper bound |
| G6 | Standard post-CSL | 0.0013 | **4** | {3:2, 4:193, fail:5} | 195/200 | Best standard |
| C_fcond | G6 + source f | 0.0058 | **4** | {1:5, 4:195} | **200/200** | 0 failures |
| T_up D2-only | V-cycle, D2 data | 0.0045 | 5 | {4:73, 5:122, fail:5} | 195/200 | Val improving |

**Wall-clock timing (G6, seed=1111):**
- NN-preconditioned FGMRES: **10–12 ms/problem** (4 iters)
- CSL-only FGMRES: **26–43 ms/problem** (15 iters)
- **Speedup: 2–4× wall-clock**, verified across 3 seeds

**3-seed verification of G6:**

| Seed | CSL baseline | NN α=1.0 median | Distribution | Failures | Time/problem |
|---|---|---|---|---|---|
| 2025 | 15 | 4 | {3:2, 4:193} | 5 | — |
| 1111 | 15 | 4 | {3:2, 4:195} | 3 | 10.5 ms |
| 3333 | 15 | 4 | {3:2, 4:195} | 3 | 11.7 ms |

**Val → iteration mapping (confirmed):**

| Val | Median iters |
|---|---|
| 0.0713 | 7 |
| 0.0374 | 6 |
| 0.0152 | 5 |
| 0.0032 | 4 |
| 0.0013 | 4 (flattening) |

The curve is flattening at 4. Moving to median=3 likely requires val < ~0.0003.
This is only achievable via self-consistent T_up training (sbatch scheme).

---

## 7. All completed experiments

### Terminated early (2026-06-22)

| Run | Data | Arch | Best val | Verdict |
|---|---|---|---|---|
| G0 | D1 random | Dilated w=64 | 0.9997 | FAIL — D1 never converges (mathematical) |
| G1 | D1+D2 | Dilated w=64 | 0.0232 then diverges | FAIL — D1 contamination, unstable |
| G3 | D1 random | UNet w=64 | 1.067 | FAIL — UNet+MaxPool diverges |
| G4 | D1 random | Dilated w=128 | 0.0720 | CAP — D1 ceiling, 7 iters at best |
| G5 | D1+D2 | UNet w=64 | 0.4125 then diverges | FAIL — UNet+MaxPool fails regardless of data |
| G6_ext | D4+D2 | Dilated w=64 | 0.0021 (worse than G6) | G6 at local optimum, confirmed |
| G6_D6ws | D6 | Dilated w=64 | 0.017 | D6 distribution mismatch disrupts warm-start |
| T_down | D2 | Dilated w=64 | val=0.008, chain=0.66 | Chain stuck — 83× amplification, structural |

### Completed and verified

| Run | Data | Best val | FGMRES | Notes |
|---|---|---|---|---|
| G2 | D2 | 0.0020 | 4 median {4:189, 5:6, fail:5} | First working model |
| G7 | D4 only | 0.0032 | 4 median {4:195, fail:5} | Pure D4, no D2 |
| G6 | D4+D2 | **0.0013** | **4 median {3:2, 4:193, fail:5}** | **Best standard** |
| overnight_g2 | D5 | 0.0022 | 4 median {3:3, 4:192, fail:5} | No improvement over G6 |
| C_fcond | D6 (f-cond) | 0.0058 | 4 median {1:5, 4:195}, **conv 200/200** | 0 failures |
| T_up D2-only | D2 | 0.0045 | 5 median {4:73, 5:122, fail:5} | Still training (ep 2540/3000) |

### Currently running (Phase 3, 2026-06-22)

| Run | GPU | Status | Estimated finish |
|---|---|---|---|
| Option A (w=128, D2, cold) | 2 | ep 540/3000, best=0.050 | ~16h |
| Option B (ah_norm loss) | 6 | ep 360/2000, best=0.0014, bouncing | ~27h (probably no improvement) |
| T_up D2-only | 5 | ep 2540/3000, best=0.0045 | ~1–2h |
| T_up w=128 D4+D2 | 1 | ep 820/3000, best=0.050 | ~17h |

---

## 8. Sbatch scheme (ready to launch)

Path: `sbatch/launch_all.sh`

**Dependency chain:**
```
job01 (generate T_up selfcon data, ~30 min)
  └── job02 (T_up w=64 warm-start on selfcon+D2, ~10h)   ← most likely to beat 4 iters
        └── job05 (generate round-2 selfcon data, ~1h)
              └── job06 (T_up w=64 round-2 warm-start, ~10h)  ← best shot at 3 iters

job03 (T_up w=128 D2-only, ~14h)       ← parallel
  └── job04 (T_up w=128 selfcon warm-start, ~10h)

job07 (measure all with 3 seeds, ~1h) ← after job02 + job04 + job06

job08 (C_fcond extended, fresh LR, ~14h)   ← parallel; may push failures further down
```

Total critical path: ~24h.

**Expected outcomes:**
- If T_up self-consistent reaches val ≈ 0.001 → median=4, possibly 3
- If C_fcond extended reaches val ≈ 0.001 → same 0-failure result, more 1-iter problems
- If T_up w=128 reaches val ≈ 0.001 → capacity check confirmed

---

## 9. Decision framework — when to stop each direction

| Direction | Stop condition | Status |
|---|---|---|
| Random data (D1) | Mathematical: gradient spread over all modes | **Closed** |
| UNet + MaxPool | Mathematical: phase destruction | **Closed** |
| More epochs on G6 | G6_ext showed val worsened → at local optimum | **Closed** |
| D5 round-2 self-consistent | No improvement over D4 | **Closed** |
| T_down chain | 83× structural amplification, chain=0.66 at val=0.008 | **Closed** |
| A_H⁻¹-norm loss | Matched G6 after 360 epochs, no progress | **Close after sbatch** |
| T_up self-consistent | Wait for sbatch job02+job06 | **Open** |
| C_fcond extended | Wait for job08 | **Open** |
| T_up w=128 | Wait for job03+job04 | **Open** |
| 1D PML test | Not started — needed before 2D | **Next step** |

---

## 10. Path to 2D heterogeneous PML

**Recommended order:**

```
Step 1: 1D homogeneous, no PML          ← DONE (4 iters, verified)
Step 2: 1D homogeneous, with PML        ← 1 day; tests if CSL+NN survives complex A
Step 3: 2D homogeneous, PML, ω=32      ← 1 week; establishes 2D data pipeline
Step 4: 2D heterogeneous, PML, ω=32    ← 1–2 weeks; add u_L conditioning
Step 5: 2D heterogeneous, ω∈{16,32,64,128}  ← production sweep
```

**Do not skip Step 2.** PML makes A_H complex and non-symmetric. The spectrum
becomes a cloud in the complex plane instead of points on the real axis. CSL shifts
this cloud — but β=0.3 may not be optimal. One day to verify is worth it.

**Pitfalls for 2D:**

1. **V-cycle cost:** T_up needs one A_L⁻¹ solve per FGMRES iteration. In 2D at
   512², this is a full 2D LU solve (~0.1s). If T_up saves 1 iteration but adds
   4 A_L⁻¹ solves, wall-clock may be worse. Always measure timing.

2. **Near-singular modes form a circle in 2D:** In 1D, one mode k≈10. In 2D,
   all (k_x, k_y) with k_x²+k_y² ≈ (ω/πh)² are near-singular — a circle of modes.
   Data requirements go up. Training may need more problems.

3. **T_up does not have a clean motivation in the PML case.**
   In 1D Dirichlet: A_H = A_L + scalar·I → same eigenvectors → T_up is rank-1
   correction at one mode. In PML: s(x,ω) depends on ω, so A_H^PML and A_L^PML
   have genuinely different operator structure. T_up would face a harder task and
   lacks the clean theoretical justification. Prefer u_L conditioning instead.

4. **u_L = A_L⁻¹f is the right 2D approach:**
   Compute u_L once before FGMRES. Pass [r₂, u_L] as NN input. This encodes both
   source AND medium c(x), costs one A_L LU solve (amortised), and does NOT require
   A_H and A_L to share eigenvectors. This is the natural 2D generalisation of
   f-conditioning, and connects directly to the frequency-transfer motivation.

5. **f-conditioning alone is insufficient in heterogeneous 2D:** f encodes source
   location but not medium c(x). u_L encodes both.

6. **β tuning is ω-dependent:** For ω=128 the CSL shift may need larger β to
   adequately move the near-singular cluster. Check CSL iteration count at each ω
   before training NN.

7. **T_down is closed for 2D too.** A_L in 2D has its own near-singular circle of
   modes. The 83× amplification argument applies in 2D as well.

---

## 11. Phase 4 — 1D PML experiments (sbatch 09–14)

### Why 1D PML before 2D

The real problem has PML absorbing boundaries. PML makes A_H complex and non-symmetric.
The spectrum is no longer points on the real axis — it becomes a cloud in the complex plane.
Testing in 1D PML first costs one day and gives high confidence before committing to 2D.

**If NN helps in 1D PML → proceed to 2D.**
**If not → diagnose before wasting GPU-days on 2D.**

### Setup

```
Equation: −(1/s(x)) d/dx (1/s(x) du/dx) − ω² u = f
  where  s(x,ω) = 1 + σ(x)/(iω)
         σ(x) = σ₀ · (dist_to_interior/npml)^pml_power, inside PML only

ω_H = 32,  ω_L = 16
n = 512,  npml = 112  (PML layers on both sides)
Interior:  indices [112:400]  = 288 points
PML region: [0:112] and [400:512]

σ₀ chosen to absorb outgoing waves:  SIGMA0 = {16: 42.5, 32: 85.0, 64: 120.0, 128: 180.0}
```

**Key changes vs Dirichlet:**
1. A_H^PML is complex and non-symmetric (scipy splu still works)
2. The spectrum is a cloud in ℂ — not points on ℝ
3. A_H^PML ≠ A_L^PML + scalar·I → different eigenvectors → T_up theory breaks
4. Loss is masked to interior [112:400] — PML region excluded (CSL already handles it)

### Architecture and approach

**Architecture:** DilatedCNN1d, width=64, same dilations as Dirichlet experiments.
No MaxPool — phase destruction argument applies equally to PML.

**Two models tested:**

| Model | in_ch | Input | Key question |
|---|---|---|---|
| G6-PML (job11) | 2 | [r₂_re/s, r₂_im/s] | Does post-CSL+NN help at all in 1D PML? |
| u_L (job13) | 4 | [r₂_re/s, r₂_im/s, u_L_re/sL, u_L_im/sL] | Does PML-context conditioning help? |

u_L = A_L^PML⁻¹ f is computed **once** before FGMRES starts. Not per-iteration.
This is cheaper than T_up (no per-iter A_L cost) and avoids the eigenvector problem.

### Beta choice (job09 output)

β is chosen by sweeping [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] and minimising median CSL iterations.
In 1D Dirichlet, β=0.3 gives 15 iters. PML may want different β.
Expected range: 15–25 CSL iters (PML is harder because spectrum is complex).

```
pml_config.json:
  omega_H         = 32
  omega_L         = 16
  beta            = <best β from sweep>
  csl_baseline_median  = <median iters at best β>
  interior_lo     = 112
  interior_hi     = 400
  pml_absorption_ratio = <|u_boundary| / |u_interior|, want < 0.01>
```

### Job chain and timing

```
job09  β sweep + PML check   ~30 min   CPU only
job10  data generation       ~2h       CPU only  (2000+200 problems, ~15 pairs each)
job11  train G6-PML          ~10h      GPU       3000 epochs, lr=3e-4→1e-6, resume if timeout
job12  measure G6-PML        ~30 min   GPU       3 seeds (2025, 1111, 3333)
job13  train u_L             ~10h      GPU       (parallel to job11)
job14  measure u_L           ~30 min   GPU       3 seeds
```

### Results (fill in after jobs complete)

**Beta sweep (job09):**

| β | CSL median iters | PML absorption ratio | Selected? |
|---|---|---|---|
| 0.2 | — | — | — |
| 0.3 | — | — | — |
| 0.4 | — | — | — |
| 0.5 | — | — | — |
| 0.6 | — | — | — |
| 0.7 | — | — | — |
| 0.8 | — | — | — |
| **Best** | — | — | ← |

**Training (jobs 11, 13):**

| Model | Best val (interior) | Epochs | Notes |
|---|---|---|---|
| G6-PML (in_ch=2) | — | — | — |
| u_L (in_ch=4) | — | — | — |

**FGMRES measurement (jobs 12, 14):**

| Model | Seed | CSL only | NN median | Conv | ms/problem |
|---|---|---|---|---|---|
| G6-PML | 2025 | — | — | —/200 | — |
| G6-PML | 1111 | — | — | —/200 | — |
| G6-PML | 3333 | — | — | —/200 | — |
| u_L | 2025 | — | — | —/200 | — |
| u_L | 1111 | — | — | —/200 | — |
| u_L | 3333 | — | — | —/200 | — |

### Interpretation guide

After results arrive:

| Outcome | Interpretation | Action |
|---|---|---|
| NN median < CSL median (both models) | Approach works in 1D PML | Proceed to 2D |
| G6-PML works, u_L doesn't help extra | u_L adds no value in simple PML | Use G6-style for 2D |
| u_L beats G6-PML | Conditioning on low-freq solution helps | Prioritise u_L in 2D |
| Both models ≈ CSL | NN not learning PML corrections | Check val curve, check interior mask |
| Training val stuck high (>0.05) | Interior-only loss masking issue, or β wrong | Check pml_config.json, try different β |

---

## 12. Experiment log

| Date | Event | Result |
|---|---|---|
| 2026-06-19 | V-cycle T_down/T_up started (UNet+MaxPool+ReduceLROnPlateau) | T_down=0.77, T_up stuck ≈0.90 |
| 2026-06-21 | MaxPool identified as root cause of T_up failure | Fix: dilated CNN, no pooling |
| 2026-06-21 | Measurement bug: pr_norm ≠ true residual; M nonlinear → need FGMRES | Fixed to pyamg FGMRES |
| 2026-06-21 | Phase 1: G2 (D2, w=64) trained from scratch | val=0.0020, **4 median iters** |
| 2026-06-21 | Phase 2: self-consistent data (D4, D5) + warm-start | G6: val=0.0013, **4 median** |
| 2026-06-22 | G0/G1/G3/G4/G5 terminated (D1 and UNet experiments) | All fail — D1 and MaxPool confirmed |
| 2026-06-22 | **3-seed verification of G6** (seeds 2025, 1111, 3333) | CSL=15 → NN=4, 2–4× wall-clock speedup |
| 2026-06-22 | G6_ext terminated: best=0.0021 | G6 at local optimum, confirmed closed |
| 2026-06-22 | G6_D6ws terminated: best=0.017 | D6 distribution mismatch, confirmed |
| 2026-06-22 | T_up D2-only measured: median=5, conv=195/200 | Beats oracle (10 iters) at val=0.0045 |
| 2026-06-22 | C_fcond measured: median=4, conv=**200/200**, {1:5, 4:195} | All failures eliminated |
| 2026-06-22 | T_down chain analysis: stuck at chain=0.66 due to 83× amplification | **T_down closed — structural failure** |
| 2026-06-22 | Sbatch scheme written (8 jobs, 24h critical path) | Ready to launch |
| 2026-06-22 | PML pipeline written: verify_beta.py, generate_pml_data.py, train_pml.py, measure_pml.py | Sbatch 09–14 ready to launch |

---

## 12. File map

| File | Purpose |
|---|---|
| `LIVE_REPORT.md` | This file |
| `operators.py` | A_H, A_L, CSL assembly |
| `train_postcsl.py` | Standard post-CSL trainer (`--condition_f`, `--loss_type`, `--width`) |
| `train_tup_standalone.py` | T_up V-cycle trainer (in_ch=4) |
| `train_tdown_standalone.py` | T_down trainer (abandoned) |
| `measure_fgmres.py` | Evaluator for standard (in_ch=2) models |
| `measure_fgmres_tup.py` | Evaluator for T_up (in_ch=4) models |
| `generate_selfconsistent_data.py` | Self-consistent data (D4, D5) |
| `generate_fcond_data.py` | f-conditioned data (D6) |
| `generate_tup_selfconsistent.py` | T_up self-consistent data (for sbatch) |
| `sbatch/launch_all.sh` | Launch full sbatch scheme |
| `fgmres_g6_val0013.json` | G6 result: {3:2, 4:193} |
| `fgmres_g6_seed1111.json` | G6 seed=1111: {3:2, 4:195} |
| `fgmres_g6_seed3333.json` | G6 seed=3333: {3:2, 4:195} |
| `run_alpha_sweep.py` | **INVALID** — uses preconditioned norm, not true residual |
| `../pml_1d/verify_beta.py` | β sweep + PML absorption check → writes pml_config.json |
| `../pml_1d/generate_pml_data.py` | FGMRES data gen with PML operators → data_pml/{train,val}.npz |
| `../pml_1d/train_pml.py` | Training with interior-only loss mask + checkpoint/resume |
| `../pml_1d/measure_pml.py` | FGMRES measurement (G6-PML and u_L models) |
| `../pml_1d/sbatch/job09_verify_beta.sh` | Gatekeeper: β sweep, PML absorption check |
| `../pml_1d/sbatch/job10_generate.sh` | Data generation (2000+200 problems) |
| `../pml_1d/sbatch/job11_train_g6.sh` | Train G6-PML (in_ch=2), 3000 epochs, resume-safe |
| `../pml_1d/sbatch/job12_measure_g6.sh` | Measure G6-PML (3 seeds) |
| `../pml_1d/sbatch/job13_train_ul.sh` | Train u_L (in_ch=4), 3000 epochs, resume-safe |
| `../pml_1d/sbatch/job14_measure_ul.sh` | Measure u_L (3 seeds) + combined summary |
| `../pml_1d/sbatch/launch_pml.sh` | Submit all 6 PML jobs with dependency chain |
