# Fair CNN vs UNet Comparison — Context & Strategy

Paste the block below into a future Claude session to get a fully-informed comparison plan.

---

## Prompt to paste

```
I want to do a fair (apples-to-apples) comparison between two architectures trained for
Helmholtz frequency transfer (Freq2Transfer project). Here is everything you need to know.

─── TASK ────────────────────────────────────────────────────────────────────────────────
Both models predict Re(u_high) + Im(u_high) from Re(u_low) + Im(u_low) (and spatial
features). Three UP operators (16→32, 32→64, 64→128) and three DOWN operators
(32→16, 64→32, 128→64). Grid: 512×512, PML depth NPML=112, interior region [112:400,112:400].
Metric: interior RelL2 (%) on Re channel.  Trivial baseline: use u_low as prediction for u_high.

─── MODEL A — Flat Dilated CNN ──────────────────────────────────────────────────────────
Class:   FrequencyTransferCNN  (train_transfer.py)
Input:   29 channels: [u_low_re/rms, u_low_im/rms, 24×Fourier(sin/cos, 6 bands), PML_map,
         omega_norm=(omega-16)/112, eta_norm=(eta-42.5)/137.5]
Arch:    stem(Conv1×1→128, IN, ReLU) → 8×DilatedConvBlock(Conv7×7, IN, ReLU,
         dilations=[1,2,3,4,5,6,7,8]) → head(Conv1×1→2)
Loss:    λ_mse·MSE_re + λ_re·RelL2_re + λ_im·RelL2_im  (all λ=1.0)

GOLDEN WEIGHTS (Green's function data — NOT directly comparable to UNet):
  UP:   experiments/claude/results_train4/run_up_20260310_142852/checkpoints/model_N600.pt
  DOWN: experiments/claude/results_train4/run_down_20260310_110520/checkpoints/model_N600.pt
  Trained on: analytic Green's function solver, N=600 per pair
  ⚠ DIFFERENT DATA SOURCE from UNet — do not use these for a fair comparison.

UMFPACK CNN WEIGHTS (same data as UNet — USE THESE for fair comparison):
  Checkpoints live in: experiments/claude/results_transfer/
  Training script: experiments/claude/train_transfer.py
  Dataset: experiments/claude/datasets/up_N4800_seed42  (UP)
           experiments/claude/datasets/down_N4800_seed42 (DOWN)
  Look for the run(s) with n_per_pair=2400 and best val_rel_l2_re.
  These must be identified from results_transfer/ JSON files / checkpoint metadata.

─── MODEL B — ResU-Net (Trial H, HPO winner) ────────────────────────────────────────────
Class:   FrequencyTransferUNet  (train_unet_hparam.py)
Input:   same 29 channels as CNN above
Arch:    stem(Conv1×1→32, IN, ReLU) → 4-level ResU-Net:
           chs=[32,64,128,256,512], InstanceNorm for levels 0-1, GroupNorm(8) for 2-3
           Encoder: ResBlock + stride-2 Conv3×3 downsampling
           Bottleneck: ResBlock
           Decoder: bilinear upsample + Conv1×1 merge + ResBlock
         head(Conv1×1→2)
Loss:    SpatialWeightedLoss: interior_w=1.0, pml_w=0.05
         λ_mse·MSE_re + λ_re·RelL2_re + λ_im·RelL2_im  (all λ=1.0)

CHECKPOINTS:
  UP:   experiments/claude/unet_hparam/runs/H_3000ep/best.pt
        epoch=25, val_rel_l2_re=0.5506  (55.1%)
        args: dataset=up_N4800_seed42, n_per_pair=2400, bs=8, lr=1e-4, max_epochs=3000
  DOWN: experiments/claude/unet_hparam/runs/H_down_3000ep/best.pt
        epoch=23, val_rel_l2_re=0.5458  (54.6%)
        args: dataset=down_N4800_seed42, n_per_pair=2400, bs=8, lr=1e-4, max_epochs=3000

─── DATASET (shared, UMFPACK solver) ────────────────────────────────────────────────────
UP dataset:   experiments/claude/datasets/up_N4800_seed42/
DOWN dataset: experiments/claude/datasets/down_N4800_seed42/
Files per dataset: u_low_re.npy, u_low_im.npy, u_high_re.npy, u_high_im.npy,
                   source_re.npy, rms.npy, omega_low.npy
Layout: 3 frequency pairs × 4800 samples each = 14400 total.
  Pair 0: array rows [0..4799],      UP: 16→32, DOWN: 32→16
  Pair 1: array rows [4800..9599],   UP: 32→64, DOWN: 64→32
  Pair 2: array rows [9600..14399],  UP: 64→128, DOWN: 128→64
Normalization: ALL fields pre-normalized by RMS of interior u_low (already in .npy files).
  rms.npy stores the per-sample scale factor for reference.

Training used n_per_pair=2400 → rows 0..2399, 4800..7199, 9600..12799 are SEEN.
Unseen (fair test set): rows 2400..4799, 7200..9599, 12800..14399 per pair.
Train/val/test split in training code: 70/15/15 random permutation (seed=42).

─── WHAT TO BUILD ───────────────────────────────────────────────────────────────────────
A side-by-side prediction plot (like fig4 in make_professor_plots.py) comparing CNN vs UNet:
  6 rows × 7 columns:
    Cols: Re(u_src) | GT Re(u_tgt) | CNN pred | CNN error | UNet pred | UNet error | metrics
    Rows: 16→32 up, 32→64 up, 64→128 up | 32→16 down, 64→32 down, 128→64 down
  Evaluate BOTH models on the SAME unseen UMFPACK samples.
  Metric column: CNN %, UNet %, Trivial %, winner + margin in pp.

Reference scripts:
  experiments/claude/make_professor_plots.py   ← fig4 layout template (CNN only)
  experiments/claude/unet/plot_unet_comparison.py  ← CNN vs UNet layout (but uses Green's fn data)
  experiments/claude/make_unet_plots.py        ← UNet-only fig4 analog on UMFPACK data

─── APPLES-TO-APPLES CHECKLIST ──────────────────────────────────────────────────────────
✓ Same dataset (UMFPACK, up/down_N4800_seed42)
✓ Same n_per_pair=2400 for both models
✓ Same 29-channel input assembly
✓ Same evaluation set (unseen rows beyond index 2400 per pair)
✓ Same metric: interior RelL2 on [112:400, 112:400]
✓ Same trivial baseline: ||u_low_re - u_high_re|| / ||u_high_re||
✗ Loss function differs slightly (CNN: no spatial weighting; UNet: pml_w=0.05)
✗ Training epochs differ (CNN: ~1000 with early stop; UNet: 3000 with early stop, best at ep~25)
  → These are acceptable differences; they reflect the actual trained models.
```
