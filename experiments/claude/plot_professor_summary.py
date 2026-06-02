"""
plot_professor_summary.py
--------------------------
Professor-facing summary figure showing the full experiment pipeline:
  Data → Phase 1 (Transfer UNet) → Warm-Start → Phase 2 (Precond UNet) → FGMRES

Layout (3 rows):
  Row 0: Pipeline schematic
  Row 1: [precond_v3 training curve] [warmstart residuals ω=32] [warmstart ω=64] [warmstart quality]
  Row 2: [UNet predictions (image)] [precond_v2 FGMRES ω=32] [precond_v2 FGMRES ω=64] [status]

Usage:
    cd ~/Freq2Transfer && source .venv/bin/activate
    python experiments/claude/plot_professor_summary.py

Output: /tmp/fkiewiet/precond_v3_plots/professor_summary.png
        figures/precond_v3/professor_summary.png  (if quota allows)
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[2]
RES     = ROOT / 'experiments' / 'claude' / 'results_transfer'
V3DIR   = Path('/tmp/fkiewiet/precond_v3_N9600/pair_16_32/T_up')
OUTDIR  = Path('/tmp/fkiewiet/precond_v3_plots')
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── colours ───────────────────────────────────────────────────────────────────
C = dict(train='#1f77b4', val='#ff7f0e', best='#2ca02c',
         zero='#555555', warm='#d62728',
         A='#888888', C='#2196F3', F='#E91E63',
         pipe_data='#CFD8DC', pipe_ph1='#BBDEFB', pipe_ws='#FFE082',
         pipe_ph2='#C8E6C9', pipe_fgmres='#E1BEE7')

# ── loaders ───────────────────────────────────────────────────────────────────

def load_v3_log():
    data = np.genfromtxt(V3DIR / 'log.csv', delimiter=',', names=True)
    s    = json.load(open(V3DIR / 'summary.json'))
    return data, s


def load_warmstart(omega: int):
    p = RES / f'warmstart_omega{omega}' / 'results.json'
    if not p.exists():
        return None
    return json.load(open(p))


def load_benchmark(omega: int):
    p = RES / f'benchmark_unet_omega{omega}' / 'results.json'
    if not p.exists():
        return None
    return json.load(open(p))


# ── Row 0: pipeline diagram ────────────────────────────────────────────────────

def draw_pipeline(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')

    stages = [
        (0.3, 'Data\nGeneration\nN=9600', C['pipe_data'], ''),
        (2.3, 'Phase 1\nTransfer UNet\nω_low→ω_high', C['pipe_ph1'], '✓ trained'),
        (4.3, 'Warm-Start\nFGMRES\nu₀ = UNet(u_low)', C['pipe_ws'], '✗ no speedup'),
        (6.3, 'Phase 2\nPrecond UNet\n≈ A⁻¹_CSL', C['pipe_ph2'], '>> in progress'),
        (8.3, 'Helmholtz\nFGMRES\n+ preconditioner', C['pipe_fgmres'], '>> pending'),
    ]
    BOX_W, BOX_H = 1.7, 1.3
    CX, CY = 1.0, 0.9  # text centre relative to box bottom-left

    for x0, label, color, status in stages:
        fancy = FancyBboxPatch((x0, 0.35), BOX_W, BOX_H,
                               boxstyle='round,pad=0.05',
                               facecolor=color, edgecolor='#455A64', linewidth=1.2)
        ax.add_patch(fancy)
        ax.text(x0 + BOX_W/2, 0.35 + BOX_H/2, label,
                ha='center', va='center', fontsize=7.5, fontweight='bold',
                multialignment='center')
        if status:
            col = '#1B5E20' if '✓' in status else ('#B71C1C' if '✗' in status else '#1565C0')
            ax.text(x0 + BOX_W/2, 0.28, status,
                    ha='center', va='top', fontsize=7, color=col, fontweight='bold')

    # Arrows between boxes
    for xi in range(len(stages)-1):
        x_start = stages[xi][0]   + BOX_W
        x_end   = stages[xi+1][0] - 0.02
        y_mid   = 0.35 + BOX_H/2
        ax.annotate('', xy=(x_end, y_mid), xytext=(x_start, y_mid),
                    arrowprops=dict(arrowstyle='->', color='#455A64', lw=1.5))

    ax.set_title('Experiment Pipeline  —  Neural Solvers for Helmholtz (512×512)',
                 fontsize=11, fontweight='bold', pad=6)


# ── Row 1 col 0: precond_v3 training ─────────────────────────────────────────

def plot_v3_training(ax):
    data, s = load_v3_log()
    ep = data['epoch'].astype(int)
    tr = data['train_loss']
    vl = data['val_loss']
    be = s['best_epoch']
    bv = s['best_val_loss']
    tv = s['test_loss_at_best']

    ax.semilogy(ep, tr, color=C['train'], lw=1.5, label='Train')
    ax.semilogy(ep, vl, color=C['val'],   lw=1.5, label='Val')
    ax.axvline(be, color=C['best'], ls='--', lw=1.0, alpha=0.85)
    ax.plot(be, bv, 'o', color=C['best'], ms=6, zorder=5,
            label=f'Best ep {be}\nval={bv:.4f}')
    ax.axhline(tv, color='purple', ls=':', lw=1.0, alpha=0.85,
               label=f'Test={tv:.4f}')
    ax.set_xlabel('Epoch', fontsize=8)
    ax.set_ylabel('RelL2 loss', fontsize=8)
    ax.set_title('Precond v3  Training\npair 16→32  N=9600', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, which='both', alpha=0.2)
    ax.tick_params(labelsize=7)


# ── Row 1 col 1-2: warm-start FGMRES residuals ────────────────────────────────

def plot_warmstart_residuals(ax, omega: int):
    d = load_warmstart(omega)
    if d is None:
        ax.text(0.5, 0.5, f'ω={omega}: no data', ha='center', va='center',
                transform=ax.transAxes, color='gray')
        ax.set_title(f'Warm-Start ω={omega}', fontsize=9)
        return
    for pidx, prob in enumerate(d['problems'][:2]):
        alpha = 1.0 if pidx == 0 else 0.55
        z = prob['Z']['residuals']
        w = prob['W']['residuals']
        iters_z = np.arange(len(z))
        iters_w = np.arange(len(w))
        lbl_z = f'Zero-start ({prob["Z"]["conv_iter"]} it)' if pidx == 0 else ''
        lbl_w = f'Warm-start ({prob["W"]["conv_iter"]} it)' if pidx == 0 else ''
        ax.semilogy(iters_z, z, color=C['zero'], lw=1.6, alpha=alpha,
                    label=lbl_z, ls='-')
        ax.semilogy(iters_w, w, color=C['warm'], lw=1.6, alpha=alpha,
                    label=lbl_w, ls='--')
    ax.axhline(1e-4, color='k', ls=':', lw=0.8, alpha=0.5, label='tol=1e-4')
    q_mean = np.mean([p['warm_prediction_quality'] for p in d['problems']])
    ax.set_xlabel('FGMRES iteration', fontsize=8)
    ax.set_ylabel('Relative residual', fontsize=8)
    ax.set_title(f'Warm-Start ω={omega}\n(CSL-precond FGMRES,  r₀ ratio={q_mean:.1f}×)',
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, which='both', alpha=0.2)
    ax.tick_params(labelsize=7)


# ── Row 1 col 3: warm-start quality bar chart ─────────────────────────────────

def plot_warmstart_quality(ax):
    omegas  = [32, 64, 128]
    labels  = ['ω=32\n(16→32)', 'ω=64\n(32→64)', 'ω=128\n(64→128)']
    quality = []
    for w in omegas:
        d = load_warmstart(w)
        if d:
            q = np.mean([p['warm_prediction_quality'] for p in d['problems']])
        else:
            q = float('nan')
        quality.append(q)

    colors = ['#ef5350' if q > 1 else '#66bb6a' for q in quality]
    bars = ax.bar(labels, [min(q, 200) for q in quality], color=colors,
                  edgecolor='#455A64', linewidth=0.8)
    ax.axhline(1.0, color='#1B5E20', ls='--', lw=1.2, label='Ideal (ratio=1×)')
    for bar, q in zip(bars, quality):
        txt = f'{q:.1f}×' if q < 500 else f'{q/1000:.0f}k×'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                txt, ha='center', va='bottom', fontsize=8, fontweight='bold', color='#B71C1C')
    ax.set_ylabel('Initial residual ratio\n(warm / zero, lower is better)', fontsize=8)
    ax.set_title('Warm-Start Quality\nacross frequencies', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 230)
    note = 'Values >1 mean warm-start\nstarts worse than zero.\nω=128: ~36 000× (clipped)'
    ax.text(0.97, 0.97, note, transform=ax.transAxes,
            fontsize=7, va='top', ha='right', color='#B71C1C',
            bbox=dict(boxstyle='round', facecolor='#FFEBEE', alpha=0.85))
    ax.tick_params(labelsize=7)


# ── Row 2 col 0: UNet prediction image ────────────────────────────────────────

def plot_predictions_image(ax):
    img_path = V3DIR / 'predictions_interior.png'
    if not img_path.exists():
        img_path = V3DIR / 'predictions.png'
    if img_path.exists():
        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title('Precond v3  Field Predictions\npair 16→32  (interior region)',
                     fontsize=9, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'predictions.png\nnot found', ha='center', va='center',
                transform=ax.transAxes, color='gray')
        ax.axis('off')


# ── Row 2 col 1-2: precond_v2 FGMRES benchmark ────────────────────────────────

def plot_precond_benchmark(ax, omega: int):
    d = load_benchmark(omega)
    if d is None:
        ax.text(0.5, 0.5, f'ω={omega}: no benchmark', ha='center', va='center',
                transform=ax.transAxes, color='gray')
        ax.set_title(f'Precond v2 FGMRES ω={omega}', fontsize=9)
        return

    prob = d['problems'][0]
    for key, label, color, lw, ls in [
        ('A', f'Unpreconditioned ({prob["A"]["iters"]}it {"✓" if prob["A"]["converged"] else "✗"})', C['A'], 1.4, '-'),
        ('C', f'ILU(10) ({prob["C"]["iters"]}it {"✓" if prob["C"]["converged"] else "✗"})',          C['C'], 1.4, '-'),
        ('F', f'Neural UNet ({prob["F"]["iters"]}it {"✓" if prob["F"]["converged"] else "✗"})',      C['F'], 1.8, '--'),
    ]:
        res = prob[key].get('residuals', [])
        show = res[:60] if key == 'F' else res  # clip diverging curve
        ax.semilogy(range(len(show)), show, color=color, lw=lw, ls=ls, label=label)

    if len(prob['F'].get('residuals', [])) > 60:
        ax.text(62, prob['F']['residuals'][59], '…diverges',
                fontsize=7, color=C['F'], va='center')

    ax.axhline(1e-4, color='k', ls=':', lw=0.8, alpha=0.5, label='tol=1e-4')
    ax.set_xlabel('FGMRES iteration', fontsize=8)
    ax.set_ylabel('Relative residual', fontsize=8)
    ax.set_title(f'Precond v2  FGMRES ω={omega}\n(neural precond diverges)',
                 fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, which='both', alpha=0.2)
    ax.set_xlim(0, 65)
    ax.tick_params(labelsize=7)


# ── Row 2 col 3: status / narrative panel ─────────────────────────────────────

def plot_status(ax):
    ax.axis('off')
    _, s = load_v3_log()

    text = (
        'Experiment Status\n'
        + '━' * 34 + '\n\n'
        '✓ Phase 1 — Transfer UNet\n'
        '  Trained: 16→32, 32→64, 64→128\n'
        '  RelL2 (32→64): 0.39  (trivial: 1.83)\n\n'
        '✗ Warm-Start FGMRES\n'
        '  CSL-precond already converges\n'
        '  in 2–3 iters from zero — no room\n'
        '  for improvement. Warm-start hurts\n'
        '  at ω=128 (ratio: ~36 000×).\n\n'
        '✗ Precond UNet v2\n'
        '  Trains to near-zero loss, but\n'
        '  diverges as FGMRES preconditioner.\n'
        '  (architecture redesign needed)\n\n'
        '>> Precond UNet v3  (current)\n'
       f'  pair 16→32, N=9600\n'
       f'  Best val:  {s["best_val_loss"]:.4f} @ ep {s["best_epoch"]}\n'
       f'  Test loss: {s["test_loss_at_best"]:.4f}\n'
        '  Checkpoint: best.pt  (ready)\n'
        '  → FGMRES benchmark pending\n'
    )
    ax.text(0.04, 0.97, text,
            transform=ax.transAxes,
            fontsize=7.8, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.95,
                      edgecolor='#90A4AE', linewidth=1.0))
    ax.set_title('Progress Summary', fontsize=9, fontweight='bold')


# ── main ──────────────────────────────────────────────────────────────────────

def make_figure():
    fig = plt.figure(figsize=(16, 13))

    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        height_ratios=[0.9, 3.2, 3.2],
        hspace=0.55,
        wspace=0.38,
    )

    # ── row 0: pipeline ──────────────────────────────────────────────────────
    ax_pipe = fig.add_subplot(gs[0, :])
    draw_pipeline(ax_pipe)

    # ── row 1 ────────────────────────────────────────────────────────────────
    ax_v3   = fig.add_subplot(gs[1, 0])
    ax_ws32 = fig.add_subplot(gs[1, 1])
    ax_ws64 = fig.add_subplot(gs[1, 2])
    ax_wsq  = fig.add_subplot(gs[1, 3])

    plot_v3_training(ax_v3)
    plot_warmstart_residuals(ax_ws32, 32)
    plot_warmstart_residuals(ax_ws64, 64)
    plot_warmstart_quality(ax_wsq)

    # ── row 2 ────────────────────────────────────────────────────────────────
    ax_pred  = fig.add_subplot(gs[2, 0])
    ax_bm32  = fig.add_subplot(gs[2, 1])
    ax_bm64  = fig.add_subplot(gs[2, 2])
    ax_stat  = fig.add_subplot(gs[2, 3])

    plot_predictions_image(ax_pred)
    plot_precond_benchmark(ax_bm32, 32)
    plot_precond_benchmark(ax_bm64, 64)
    plot_status(ax_stat)

    # ── row labels ───────────────────────────────────────────────────────────
    for row_y, label in [(0.97, ''), (0.655, 'Phase 1  &  Warm-Start'), (0.315, 'Phase 2  —  Preconditioner')]:
        if label:
            fig.text(0.005, row_y, label, fontsize=9, fontweight='bold',
                     color='#37474F', va='top', rotation=90)

    fig.suptitle(
        'Freq2Transfer  —  Neural Solvers for Helmholtz  (512×512,  N=9600)',
        fontsize=13, fontweight='bold', y=0.995,
    )

    out = OUTDIR / 'professor_summary.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out}')

    # also try to save to figures/precond_v3/
    fig_dir = ROOT / 'figures' / 'precond_v3'
    fig_dir.mkdir(parents=True, exist_ok=True)
    out2 = fig_dir / 'professor_summary.png'
    try:
        import shutil
        shutil.copy(out, out2)
        print(f'Copied → {out2}')
    except Exception as e:
        print(f'  (could not copy to figures/: {e})')

    return str(out)


if __name__ == '__main__':
    make_figure()
