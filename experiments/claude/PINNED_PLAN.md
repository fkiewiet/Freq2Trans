# Pinned Experiment Plan — Neural Preconditioners for 1D Helmholtz
Last updated: 2026-06-19

---

## Experiment 1: Neural warm-start (heterogeneous medium)

**Operator:** A = −d²/dx² − c(x)² u = f, n=512, Dirichlet BCs
**Medium:** c_L = {16, 24}, c_H = {32, 48} (piecewise at x=0.5)
**Preconditioner for FGMRES:** Dirichlet-CSL_H, β=0.3:  −d²/dx² − c_H(x)²(1+iβ)²

**Goal:** Predict x₀ = T(u_L, f) ≈ u_H so FGMRES needs fewer than 29 iterations from that start.

**Baselines (measured, n=30 problems, tol=1e-6):**

| Configuration | Iterations |
|---|---|
| Cold start x₀=0, Dirichlet-CSL_H | 29 |
| Oracle x₀=u_L | 28 (saves 1 — barely helpful) |
| Oracle x₀=u_H (perfect) | 0 |

**Model:** 5-level UNet, base_ch=32, kernel=7, GELU, no normalization layers
- in_ch=4: [Re(u_L)/‖u_L‖, Im(u_L)/‖u_L‖, Re(f)/‖f‖, Im(f)/‖f‖]
- out_ch=2: [Re(u_H), Im(u_H)] / ‖u_L‖
- At eval: pred_physical = T(x_norm) × ‖u_L‖

**Data:** 50k train / 5k val
- f: 3-6 Gaussian sources, amp~U[1,2], phase~U[0,2π]
- Solves: A_L u_L = f, A_H u_H = f (exact LU)
- Dataset: `data_mid/` (also contains u_mid; ignored by 4ch model)

**Loss:** Relative L2  ‖T(x) − u_H/‖u_L‖‖² / ‖u_H/‖u_L‖‖²

**Quality gate before FGMRES eval:** val < 0.2

**FGMRES eval:** tol=1e-6, n=200 problems, seed=2025

**Status (2026-06-19): EVALUATED — APPROACH SHELVED**

All runs completed (~500 epochs each). Best model: `runs_6ch_resid/warmstart_best.pt`
- Input: [u_L, u_mid, f] (6ch), target: (u_H − u_mid)/‖u_L‖ (residual correction)
- Trained to val=0.1926 (passed the val < 0.20 gate)
- FGMRES eval (n=200, tol=1e-6, seed=2025): **Neural=31 iters, Cold=27 iters (WORSE)**

**Root cause of failure:** The neural x₀ = u_mid + T(...)·‖u_L‖ has initial preconditioned
residual ‖M⁻¹(f − A_H x₀)‖/‖M⁻¹f‖ = 7.47 (vs 1.0 for cold start). Starting from a
vector that satisfies a nearby-but-different equation (A_mid) creates a large residual in
a Krylov-unfriendly direction, adding iterations instead of saving them. The val < 0.20
proxy metric does NOT predict FGMRES improvement.

**Lesson:** For Helmholtz+CSL, warmstart only helps if x₀ ≈ u_H to very high accuracy.
The gap ω_L→ω_H is too large for a useful transfer. **Do not pursue further.**

Other runs: warmstart_6ch_b32 (val=0.84), warmstart_4ch_f (val=0.91), warmstart_6ch_b16
(val=0.69) — none evaluated; all worse than the resid model. DONE.

---

## Experiments 2 & 3: Neural V-cycle (homogeneous medium)

**Operator:** A = −d²/dx² − ω² u = f, n=512, Dirichlet BCs
**Medium:** Homogeneous, constant ω (c(x) = ω everywhere)
**Pair:** ω_L=16, ω_H=32

**Preconditioner structure (additive):**
```
M(r)  =  CSL_H⁻¹(r)  +  T_up( A_L⁻¹( T_down(r) ) )
```
- CSL_H = −d²/dx² − ω_H²(1+iβ)², β=0.3  (applied via sparse LU, factored once)
- Coarse solver: exact A_L⁻¹ (sparse LU of real operator, cheap for 1D)
- T_down: neural network, maps FGMRES residual r_H → r_L for coarse input
- T_up: neural network, maps coarse correction e_L → fine correction e_H

**Baselines — MEASURED (tol=1e-6, n=200 problems, seed=2025):**

| Configuration | Median | Notes |
|---|---|---|
| CSL_H only | 15 | baseline |
| Additive CSL_H⁻¹ + A_L⁻¹ | 20 | WORSE — A_L⁻¹ corrupts Krylov space |
| Multiplicative CSL_H⁻¹ then A_L⁻¹ | 17 | worse |
| **Additive CSL_H⁻¹ + A_H⁻¹** | **10** | **TARGET — 33% fewer iters** |
| A_L⁻¹ only | 21 | for reference |

**Key insight:** A_L⁻¹ by itself hurts (A_L has its own near-resonant mode at k≈5 and amplifies
those components in FGMRES residuals). The target is CSL + A_H⁻¹ = 10 iters. The neural V-cycle
must approximate A_H⁻¹ well enough (not just A_L⁻¹) to close the gap from 15 toward 10.

**Structure (additive — confirmed as right choice when correction ≈ A_H⁻¹):**
```
M(r)  =  CSL_H⁻¹(r)  +  T_up( A_L⁻¹( T_down(r) ) )
```
where T_up ∘ A_L⁻¹ ∘ T_down  ≈  A_H⁻¹

Justification: if T_down = A_L A_H⁻¹ exactly, then A_L⁻¹(T_down(r)) = A_H⁻¹(r) and T_up = I.
The two networks together absorb the full spectral mismatch between A_L and A_H.

**Architecture:**
- T_down: 5-level UNet, base_ch=32, in_ch=2 [Re(r), Im(r)] → out_ch=2 [Re(r_L), Im(r_L)]
- T_up:   5-level UNet, base_ch=32, in_ch=4 [Re(e_L), Im(e_L), Re(r), Im(r)] → out_ch=2
  T_up sees both the coarse correction e_L AND the original residual r.
  With r available, T_up can implicitly do defect correction: it learns where T_down went wrong.
- Kernel=7, GELU, no normalization layers, per-sample RMS normalization of input

**Joint training (end-to-end through A_L⁻¹):**
```
r  →  T_down  →  r_L  →  A_L⁻¹  →  e_L  →  T_up([e_L, r])  →  e_H
Loss  =  ‖e_H  −  A_H⁻¹(r)‖  /  ‖A_H⁻¹(r)‖
```
Backprop through A_L⁻¹ via implicit differentiation (A_L symmetric → backward = another A_L⁻¹ solve).
T_down and T_up are trained simultaneously; the split of work between them is learned, not prescribed.

Applied every FGMRES iteration as part of the preconditioner:
```
M(r)  =  CSL_H⁻¹(r)  +  T_up( A_L⁻¹( T_down(r) ),  r )
```

**Quality gate:** end-to-end val < 0.2 before wiring into FGMRES eval.
(Separate T_down/T_up val is no longer the metric — only the joint chain matters.)

**FGMRES eval:** tol=1e-6, n=200 problems, seed=2025
**Success target:** M(r) delivers fewer FGMRES iterations than CSL_H alone (< ~15 iters)

---

### Experiment 2: V-cycle with random-vector training

**Training data:**
- Input r: random complex vector (sum of iid complex Gaussians, n=512)
- Target: A_H⁻¹(r) — one solve per sample
- 50k train / 5k val; r normalised by rms(r)
- Rationale: random vectors span all of ℂⁿ; covers the full FGMRES residual space

**Joint training:**
- T_down sees r (2ch), T_up sees [A_L⁻¹(T_down(r)), r] (4ch)
- End-to-end loss: ‖T_up(A_L⁻¹(T_down(r))) − A_H⁻¹(r)‖ / ‖A_H⁻¹(r)‖
- Both networks optimised simultaneously via backprop through A_L⁻¹

---

### Experiment 3: V-cycle with FGMRES-residual training

**Same as Exp 2** except training vectors are actual FGMRES residuals, not random.

**Training data:**
- Run CSL_H-preconditioned FGMRES on 2000 random sources f (homogeneous, ω_H=32)
- Save residuals r_k at iterations k = 0, 1, 2, 4, 8, 16 → ~12000 samples
- Target: A_H⁻¹(r_k) — one A_H solve per residual; r_k normalised by rms(r_k)
- Rationale: exact distribution match — networks train on exactly what they will see at eval time

**Joint training:** identical to Exp 2 — end-to-end through A_L⁻¹.

**Rationale over Exp 2:** Previous FGMRES-residual T_down achieved val=0.019 vs val=0.28 for random-vector training. Distribution match is the dominant factor.

---

## Priority order (iteration count is the primary goal — target: 10 iters)

Homogeneous oracles already measured. Target = 10 iters (CSL + A_H⁻¹).
Exp 1 (warm-start) is DONE and SHELVED — failed at eval despite val < 0.2.

**Current focus: Exp 3 (V-cycle, FGMRES-residual data). All 8 GPUs free.**

**Status (2026-06-19): All V-cycle runs hit 500 epoch limit, none reached val < 0.20.**

| Run | Best val | Root cause of plateau |
|---|---|---|
| tdown_only (GPU 5) | 0.7696 at ep 475 | Improving slowly; needs 500+ more epochs |
| tup_only (GPU 4) | 0.9096 at ep 497 | **LR killed too early: patience=30 → 4 halvings → stuck at 3.1e-05** |
| vcycle_joint (GPU 3) | 0.9123 at ep 494 | Same LR issue + gradient shrinkage through A_L⁻¹ |

**The single most important fix: change `ReduceLROnPlateau(patience=30)` → `patience=80`**
in `train_tup_standalone.py` and `train_vcycle_joint.py`. The scheduler halved LR 4×
before the models had converged. At patience=80, the LR decays 3× slower, allowing
proper convergence before the optimizer loses mobility.

**Restart plan (on any waveserver with free GPUs):**
1. Fix patience=80 in both scripts
2. GPU 0: T_up standalone fresh start — `CUDA_VISIBLE_DEVICES=0 python train_tup_standalone.py --train`
3. GPU 1: Joint E2E fresh start — `CUDA_VISIBLE_DEVICES=1 python train_vcycle_joint.py --train --data_dir ./data_vcycle_joint --out_dir ./runs_vcycle_joint`
4. GPU 2: T_down continue (it IS still learning) — `CUDA_VISIBLE_DEVICES=2 python train_tdown_standalone.py --train`

All scripts are in: `experiments/claude/eigenvalue_1d/corrected_flux_pipeline/`
Training data already exists in: `experiments/claude/eigenvalue_1d/corrected_flux_pipeline/data_vcycle_joint/`

1. ~~Exp 1 warm-start~~ — DONE, SHELVED (failed at eval)
2. **Exp 3 T_up** — restart with patience=80, target val < 0.2
3. **Exp 3 joint** — restart with patience=80, target val < 0.2
4. **Full Exp 3 eval** once any run hits val < 0.2: wire into FGMRES, target < 15 iters
5. Exp 2 (random vectors) — lower priority, try if Exp 3 fails

Stop any run showing no path to val < 0.2 after 300 epochs — diagnose root cause first.

---

## Architecture decisions (fixed)

- 5-level UNet, base_ch=32 (~11M params), no normalization layers
- Kernel size 7, GELU activations, MaxPool1d(2) downsampling, ConvTranspose1d upsampling
- Head: Conv1d(b, 2, 1) → 2 output channels (Re, Im)
- Per-sample RMS normalisation of inputs; no InstanceNorm or BatchNorm inside the UNet
- Optimizer: Adam, lr=5e-4, ReduceLROnPlateau(factor=0.5, **patience=80**, min_lr=1e-6)
  (patience=30 was a bug — caused 4 premature LR halvings, stuck all runs at 0.91)

---

## What NOT to do

- Do NOT train T_down and T_up separately with independent losses — joint end-to-end training is required so they learn to work together
- Do NOT use CSL_L⁻¹ as coarse solver (oracle showed it cancels benefit vs A_L⁻¹)
- Do NOT wire the chain into FGMRES before end-to-end val < 0.2
- Do NOT use PML system as the main system (too easy)
- Do NOT use InstanceNorm inside UNet (destroys Re/Im amplitude ratio = phase)
- Do NOT normalise inputs by the same scale when they have very different amplitudes
- Do NOT use modal gating (uses analytical eigenbasis — not learned)
- Do NOT use tol=1e-8 for FGMRES eval (use tol=1e-6 throughout)
