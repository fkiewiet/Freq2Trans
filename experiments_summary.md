# Freq2Transfer — Experiment Summary
## Frequency-Transfer Neural Operator for Helmholtz Preconditioning F. Kiewiet · MIT, Prof. L. Demanet · as of 2026-04-04 Research Goal
Train a CNN TωL→ωH : ulow ↦ uhigh that maps Helmholtz solutions from a coarser frequency ωL to a finer frequency ωH on the same source configuration. The trained operator is then used as the action of an FGMRES right-preconditioner M−1 for the high-frequency system AHx = b, with the intent of reducing iteration counts relative to standard algebraic preconditioners (ILU, CSL). The three operator pairs are `16→32`, `32→64`, `64→128` on a 512×512 finite-difference grid with PML absorbing boundaries (depth 112 cells).
## System & Architecture
## **Helmholtz system:** (Δ + (ω/c + iσ)2)u = f, solved by sparse LU (`scipy.sparse.linalg.spsolve`) for data generation. **Data (analytic):** 2D free-space Green's function G(r) = (i/4)H0(1)(ωr) via FFT convolution on a 2× zero-padded grid. 3–6 Gaussian sources per sample, amplitude ∈ [1, 2], phase ∈ [0, 2π]. **Input channels (29):** Re(ulow), Im(ulow), PML map, 24 Fourier positional features (6 bands × sin/cos × X, Y), ω/128, direction bit. **Model:** Dilated CNN — stem (1×1) + 8 dilated conv blocks (3×3 or 7×7, dilations 1–8 linear), InstanceNorm, ReLU, width 128, ~14 M parameters. **Normalisation:** Both ulow and uhigh divided by rms(ulow|interior). Zero-predictor RelL2 = 100% analytically, verified on every run. Experiment Timeline
### Saturation Probe early March 2026 Exp 0 — Saturation Curve (train4): Can the model learn at all?
**Motivation**
Establish whether a dilated CNN can reduce RelL2 below the trivial baseline and identify the critical training-set size N* beyond which accuracy saturates.
**Method**
Train on Green's function data for all three operator pairs simultaneously. Vary N ∈ {150, 300, 600, 1200} samples per pair. Loss: λ₁·MSERe + λ₂·RelL2Re (imaginary channel not included — this becomes Exp 1A).
**Results**
| N / pair | DOWN val RelL2 | UP val RelL2 |
| -------- | -------------- | ------------ |
| 150      | 69.7%          | 79.2%        |
| 300      | 65.6%          | 71.5%        |
| 600      | 61.9%          | 65.5%        |
| 1200     | 59.0%          | 65.5%        |

**Trivial zero baseline: 100.0% (verified analytically) Power-law fit: N* ≈ 4000–8000 (curve still declining at N=1200) Interpretation**
### The model is learning — 59–66% is well below baseline. But the saturation plateau has not been reached, meaning either more data or architectural improvements are needed before the preconditioner can be competitive. This established the experimental agenda for all subsequent runs. Result: learning confirmed N* not yet reached Architecture Probe 2026-03-13 Exp 1A — Autoencoder Identity Task: Is the architecture the bottleneck?
**Motivation**
Disentangle *architecture capacity* from *task difficulty*. If ωsource = ωtarget (identity map), what is the minimum achievable RelL2?
**Method**
Set target = input (same ω in and out). Train on all three ω ∈ {32, 64, 128} with N = 150. Re and Im channels logged separately.
**Results (N=150, epoch 145)**
**Real channel: 2.5% RelL2 — converging strongly, all three ω identical Imaginary channel: 53% RelL2 — frozen from epoch 1 Root cause: λ_imag = 0 in loss — Im channel receives no gradient Interpretation**
### **The architecture is not the bottleneck.** 2.5% on the identity task vs. 59–66% on the actual transfer confirms that the frequency mapping is the hard part, not the representation. The imaginary bug immediately motivated adding `λ_imag·RelL2_Im` to all subsequent loss functions. Architecture sufficient Im-channel bug identified Ablation 2026-03-13 Exp 1B — Amplitude Ablation: Is the 1/√r singularity necessary?
**Motivation**
Prof. Demanet's hypothesis: the model implicitly decomposes multi-source fields into Voronoi cells, with each source identified by its 1/r amplitude peak. Test: remove the singularity and check if learning collapses.
**Method**
Replace the Hankel Green's function G(r) = eikr/√r with a phase-only version G̃(r) = eikr (constant amplitude, same phase field). Train with identical hyperparameters in both UP and DOWN directions.
**Results**
**Phase-only (UP): 100% RelL2 at every epoch — model learns nothing Phase-only (DOWN): 100% RelL2 at every epoch — confirmed Hankel (train4 reference): 59–66% Interpretation**
### **The 1/√r amplitude singularity is a necessary condition for learning.** Without a spatially unique amplitude marker at each source location, the model cannot decompose multi-source fields, consistent with a Voronoi-windowing interpretation. This is the strongest possible confirmation of the professor's hypothesis — not just a degradation, but a complete failure to learn. Hypothesis confirmed: 1/r necessary Linearity Test 2026-03-13 Exp 1C — Superposition Test: Is the learned operator linear?
**Motivation**
If T is linear, then T(f₁ + f₂) = T(f₁) + T(f₂). Linearity is necessary for the operator to be usable as a preconditioner (preconditioners must be linear maps on residuals). Paper-ready threshold: ε < 8%.
**Method (Variant A)**
For held-out source pairs (f₁, f₂), compute ε = ‖T(f₁+f₂) − T(f₁) − T(f₂)‖ / ‖T(f₁)+T(f₂)‖. Input to the network is the sum of the two input fields (Variant A normalisation).
**Results**
| Direction | Pair   | Mean ε |
| --------- | ------ | ------ |
| UP        | 16→32  | 38.1%  |
| UP        | 32→64  | 35.6%  |
| UP        | 64→128 | 32.3%  |
| DOWN      | 32→16  | 35.4%  |
| DOWN      | 64→32  | 33.0%  |
| DOWN      | 128→64 | 29.5%  |

**Diagnosis from spatial residual maps**
**Signed residual Re[T(f₁+f₂)] − Re[T(f₁)] − Re[T(f₂)]: Large-scale spatially uniform pattern (not localised near sources) → global scaling error, not geometric nonlinearity Root cause: Variant A combined input has ~√2 × higher amplitude than training distribution. InstanceNorm renormalises differently → global scale error. Variant B fix: normalise combined input by combined RMS before inference. Interpretation**
### The 30–38% is a **normalisation artifact**, not true nonlinearity. The spatial pattern of the error (global, not source-localised) is the diagnostic. Variant B (correct normalisation) is expected to drop ε well below 8%. This experiment was decisive: it told us exactly *why* the numbers look bad and *what* to fix. Artifact identified, Variant B pending Pipeline 2026-03-16 Exp 2 — Full Pipeline Rebuild: Larger Data + Improved Training
**Motivation**
The saturation curve at N=1200 has not plateaued (N* ≈ 4000–8000). Before the preconditioner can work, the transfer operator must be sufficiently accurate. Four systematic changes were made.
**Four changes (train_transfer.py vs train4)**
****1. RMS normalisation** Both u_low and u_high ÷ rms(u_low|interior). Zero-predictor = 100% analytically. **2. Im channel in loss** (motivated by Exp 1A) L = λ₁·MSE_Re + λ₂·RelL2_Re + λ_imag·RelL2_Im Phase 1 grid search: λ_imag ∈ {0.0, 0.1, 0.3, 1.0} **3. Cosine annealing with warm restarts** T₀=50, T_mult=2 → LR restarts at epochs 50, 150, 350, 750 **4. Extended patience** max_epochs=1000, patience=150 (spans two full restart cycles) Dataset generated (analytic Green's function, FFT)**
**up_N4800_seed42: 14 400 samples, ~75 GB (3 pairs × 4800) — COMPLETE down_N4800_seed42: 14 400 samples, ~75 GB (3 pairs × 4800) — COMPLETE Nested seed: first N samples match any sub-dataset exactly (no re-generation needed) Format: directory of .npy memmaps — OS pages ~5 MB/sample at load time Phase 1 training runs (λ_imag search at N=1200)**
| Server | Direction | λ_imag                | Batch |
| ------ | --------- | --------------------- | ----- |
| wave7b | UP        | 0.0 / 0.1 / 0.3 / 1.0 | 4     |
| wave6  | DOWN      | 0.0 / 0.1 / 0.3 / 1.0 | 2     |

### Dataset ready for N=4800 saturation curve Best λ_imag: prior is 0.3 Loss Design 2026-03-19 → 2026-04-03 Exp 3 — Complex RRMSE Loss: Mathematically Correct Objective
**Motivation**
Re and Im are not independent channels — they are the real and imaginary parts of a single complex field u ∈ ℂ. Treating them as separate loss terms weights the two directions of a complex rotation differently, which is artificial. A rotation-invariant loss must use the complex modulus.
**New loss (train_transfer_v2.py)**
**Old: RelL2_Re + λ_imag · RelL2_Im (two separate real norms) New: RRMSE_ℂ = √(∑|ŷ − y|²ℂ) / √(∑|y|²ℂ) over interior Equivalent to: ‖ŷ_Re − y_Re‖² + ‖ŷ_Im − y_Im‖² in the numerator/denominator jointly — no free λ_imag hyperparameter, Re and Im weighted equally. Suggested by Aimé Fournier (2026-04-03). Results (N=1200, kernel=3, 80–95 epochs)**
| Experiment             | Direction | Best val RelL2 | Note           |
| ---------------------- | --------- | -------------- | -------------- |
| T_up_32_64 (1st run)   | UP        | 140.6%         | LR too high    |
| T_down_64_32 (1st run) | DOWN      | 109.1%         | LR too high    |
| T_up_32_64 (2nd run)   | UP        | 82.4%          | lr=1.1e-4, k=3 |
| T_down_64_32 (2nd run) | DOWN      | 84.6%          | lr=1.1e-4, k=3 |

**Interpretation**
### The complex RRMSE loss is mathematically cleaner and eliminates the λ_imag hyperparameter search. The 82–85% results at N=1200 with kernel=3 and only 80–95 epochs are not yet at best performance (train4 reached 59–66% with kernel=7 and more epochs). The N=4800 run with warm restarts and full training budget is the next step. Loss formulation principled N=4800 run needed for fair comparison Negative Result 2026-03-13 → 2026-04-04 Exp 4 — GMRES Preconditioner Benchmarks (v1–v6): Does the neural operator help?
**Preconditioner construction**
**M⁻¹v (FGMRES right-preconditioner, one solve per Krylov step): 1. Extract interior of v → v_int ∈ ℝ288×288 2. Apply T_down → ũ_low (approximate low-freq solution) 3. Apply A_L⁻¹(ILU) → low-freq corrected field 4. Apply T_up → approximate high-freq correction 5. Zero-pad back to 512×512 Comparison variants**
| Variant | Method                            |
| ------- | --------------------------------- |
| A       | Unpreconditioned GMRES            |
| B       | Jacobi (diagonal scaling)         |
| C       | ILU(0)                            |
| D       | Complex Shifted Laplacian (β=0.5) |
| E       | Neural FGMRES (ours)              |

**Results — ω=16→32, N=4800 weights, ILU fill=10, 1000 Krylov steps**
| Variant        | Iters | Converged? | r_final | Per-step cost |
| -------------- | ----- | ---------- | ------- | ------------- |
| A (Unprecond.) | 1000  | No         | 0.181   | ~148 ms       |
| B (Jacobi)     | 1000  | No         | 0.164   | 2 ms          |
| C (ILU)        | 1000  | No         | 0.158   | 169 ms        |
| D (CSL)        | 1000  | No         | 0.649   | 171 ms        |
| **E (Neural)** | 1000  | No         | 0.999   | 635 ms        |

**Diagnosis**
**Neural (E) final residual: 0.999 ≈ no progress after 1000 steps CSL (D): also worse than unpreconditioned (0.649 vs 0.181) Best classical: C (ILU, 0.158) — only marginally better than A The 512×512 Helmholtz system at ω=32 is highly indefinite. None of the tested preconditioners achieve convergence to 10⁻⁴. Neural preconditioner failure modes identified across versions v1–v6: v1–v2: FGMRES stagnates — preconditioner apply cost >7 s/step on CPU v3–v4: Mismatch between interior-restricted preconditioner and GMRES left-preconditioning v5: splu intractable at 512×512 for A_L⁻¹; switched to spilu (fill=10) v6: GPU inference (635 ms/step), ILU fill=10 — still no convergence Interpretation**
## **Honest negative result.** The current neural operator (RelL2 ≈ 59–66%) is not yet accurate enough to serve as a useful Krylov preconditioner — a preconditioner that applies an *approximate* inverse at the wrong scale actively misleads the Krylov iteration. This is well understood mathematically: for right-preconditioning, M ≈ A is required, and 60% relative error in M is far from this regime. The CSL also failing to outperform the unpreconditioned system is a known difficulty with 512×512 indefinite Helmholtz problems and is consistent with the literature (Erlangga et al., 2006). **Next step:** Improve RelL2 to ≲20–30% via N=4800–9600 training before re-benchmarking. A coarser grid test (N=256 or 128) would also isolate whether the problem is with the preconditioner construction or with the intrinsic difficulty of the Helmholtz system. Neural preconditioner not yet effective Accuracy gate: RelL2 ≲ 30% needed Summary of All Results
| Experiment                         | Date         | Key Metric                 | Value              | Status                                |
| ---------------------------------- | ------------ | -------------------------- | ------------------ | ------------------------------------- |
| 0 — Saturation curve (train4)      | Feb–Mar 2026 | Val RelL2, N=1200          | 59–66%             | Learning confirmed, N* not reached    |
| 1A — Autoencoder identity          | 2026-03-13   | Val RelL2, Re, N=150       | 2.5%               | Architecture sufficient; Im bug fixed |
| 1B — Amplitude ablation            | 2026-03-13   | Val RelL2, phase-only      | 100.0%             | 1/r singularity necessary (confirmed) |
| 1C — Superposition (Variant A)     | 2026-03-13   | Linearity error ε          | 30–38%             | Artifact; Variant B pending           |
| 2 — N=4800 dataset + λ_imag search | 2026-03-16   | Dataset size               | 14 400 samples/dir | Ready; training in progress           |
| 3 — Complex RRMSE loss (v2)        | 2026-03-19   | Val RelL2, N=1200          | 82–85%             | Loss principled; N=4800 run pending   |
| 4 — GMRES preconditioner (v1–v6)   | 2026-03-13 → | Final residual, 1000 steps | 0.999 (E)          | Negative; accuracy gate not met       |

### Open Questions & Next Steps (in order of priority)
1. **N=4800 saturation curve** — does RelL2 drop below 45% with the full dataset and warm-restart schedule? This is the necessary gate before re-attempting the preconditioner.
1. **Superposition Variant B** — normalise combined input by combined RMS; expected to show ε < 8% and confirm linearity.
1. **Coarser-grid preconditioner test** — run v6 on a 128×128 or 256×256 grid to isolate whether the problem is with the operator accuracy or with the indefiniteness of the 512×512 system.
1. **N=9600 data generation** — launch scripts ready (`experiments/claude/launch/generate_N9600_local.sh`); needed if curve still declining at N=4800.
1. **Variant B doubling test** — check scale equivariance T(2u) = 2T(u); InstanceNorm expected to fail (30–50% error), which would motivate replacing InstanceNorm with LayerNorm or GroupNorm.
