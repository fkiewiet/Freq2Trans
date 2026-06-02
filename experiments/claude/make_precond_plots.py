"""
make_precond_plots.py
----------------------
Professor-ready plots from neural preconditioner training + FGMRES benchmark.

Usage (morning after training):
    cd ~/Freq2Transfer && source .venv/bin/activate
    python experiments/claude/make_precond_plots.py

Outputs -> experiments/claude/results_transfer/professor_plots/
    fig1_training_curves.png
    fig2_fgmres_omega32.png
    fig3_iteration_table.png
    fig4_combined.png          <-- main figure for professor
"""

from __future__ import annotations
import json, re, sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT    = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'experiments' / 'claude' / 'results_transfer'
OUTDIR  = RESULTS / 'professor_plots'
OUTDIR.mkdir(parents=True, exist_ok=True)

OMEGAS = [16, 32, 64, 128]
COLORS = {16: '#1f77b4', 32: '#ff7f0e', 64: '#2ca02c', 128: '#d62728'}
MCOL   = {'A': '#888888', 'C': '#2196F3', 'F': '#E91E63'}
MLAB   = {'A': 'Unpreconditioned', 'C': 'ILU(10)', 'F': 'Neural UNet'}


# ── loaders ───────────────────────────────────────────────────────────────────

def load_training_log(omega: int) -> dict:
    """
    Parse log.txt produced by train_precond.py.

    Log format (column-based, NOT key=value):
        epoch  rl2_train  rl2_val   cos_val   lr         time_s
        ─────────────────────────────────────────────────────────
             1  0.0588     0.0030    0.0014    3.00e-04  46s
            ✓ best.pt  (rl2=0.0030  epoch=1)
             2  0.0016     0.0008  ...
    """
    log = RESULTS / f'precond_unet_v2_omega{omega}' / 'log.txt'
    if not log.exists():
        print(f"  [warn] missing: {log}")
        return {}
    # Match lines starting with optional whitespace then an integer (epoch number)
    # followed by 4 floats (rl2_train, rl2_val, cos_val, lr)
    pat = re.compile(
        r'^\s+(\d+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)'
    )
    # Also parse "best.pt" lines: "✓ best.pt  (rl2=X  epoch=Y)"
    best_pat = re.compile(r'best\.pt\s+\(rl2=([\d.eE+\-]+)\s+epoch=(\d+)\)')
    epochs, train_l, val_l = [], [], []
    best_val, best_ep = float('inf'), 0
    with open(log) as f:
        for line in f:
            m = pat.match(line)
            if m:
                ep = int(m.group(1))
                tr = float(m.group(2))
                vr = float(m.group(3))
                epochs.append(ep); train_l.append(tr); val_l.append(vr)
                if vr < best_val:
                    best_val, best_ep = vr, ep
            bm = best_pat.search(line)
            if bm:
                bv = float(bm.group(1))
                be = int(bm.group(2))
                if bv < best_val:
                    best_val, best_ep = bv, be
    if not epochs:
        print(f"  [warn] no epoch lines parsed from {log}")
        return {}
    return dict(epochs=epochs, train_loss=train_l, val_rl2=val_l,
                best_val=best_val, best_epoch=best_ep)


def load_benchmark(omega: int) -> dict:
    p = RESULTS / f'benchmark_unet_omega{omega}' / 'results.json'
    if not p.exists():
        print(f"  [warn] missing: {p}")
        return {}
    with open(p) as f:
        return json.load(f)


# ── fig 1: training curves ─────────────────────────────────────────────────────

def plot_training_curves(ax):
    has_data = False
    for omega in OMEGAS:
        d = load_training_log(omega)
        if not d:
            continue
        has_data = True
        c = COLORS[omega]
        ax.semilogy(d['epochs'], d['val_rl2'], color=c, lw=1.6,
                    label=f'ω={omega}  (best={d["best_val"]:.1e} @ep{d["best_epoch"]})')
        ax.axvline(d['best_epoch'], color=c, ls='--', lw=0.8, alpha=0.5)
        ax.plot(d['best_epoch'], d['best_val'], 'o', color=c, ms=5)
    if not has_data:
        ax.text(0.5, 0.5, '(training logs not yet available)',
                ha='center', va='center', transform=ax.transAxes, color='gray', fontsize=10)
    ax.axhline(1e-3, color='k', ls=':', lw=0.7, alpha=0.4)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel('Val Rel-L2', fontsize=9)
    ax.set_title('Precond Training — Validation Loss', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, which='both', alpha=0.25)


# ── fig 2: fgmres residuals ────────────────────────────────────────────────────

def plot_residuals_omega(omega: int, axes: list):
    bm = load_benchmark(omega)
    problems = bm.get('problems', []) if bm else []
    n = min(len(problems), len(axes))
    for pidx in range(len(axes)):
        ax = axes[pidx]
        ax.set_title(f'Problem {pidx+1}', fontsize=9)
        ax.set_xlabel('Iteration', fontsize=8)
        if pidx == 0:
            ax.set_ylabel('Relative Residual', fontsize=8)
        if pidx >= n:
            ax.text(0.5, 0.5, '(not run)', ha='center', va='center',
                    transform=ax.transAxes, color='gray', fontsize=9)
            continue
        prob = problems[pidx]
        n_src = prob.get('n_src', prob.get('n_sources', '?'))
        ax.set_title(f'Prob {pidx+1} ({n_src} src)', fontsize=9)
        for key in ['A', 'C', 'F']:
            res   = prob.get(key, {})
            resids = res.get('residuals', [])
            conv  = res.get('converged', False)
            iters = res.get('iters', 0)
            label = f'{MLAB[key]} ({iters}it{"✓" if conv else "✗"})'
            lw    = 1.8 if key == 'F' else 1.2
            ls    = '-' if conv else '--'
            if resids:
                ax.semilogy(resids, color=MCOL[key], lw=lw, ls=ls, label=label)
            else:
                fr = res.get('final_res', float('nan'))
                ax.axhline(fr, color=MCOL[key], lw=1, ls=':', label=label)
        ax.axhline(1e-4, color='k', ls=':', lw=0.8, alpha=0.5)
        ax.legend(fontsize=6.5)
        ax.grid(True, which='both', alpha=0.25)
        ax.set_ylim(1e-6, 1e2)


# ── fig 3: iteration table ─────────────────────────────────────────────────────

def plot_iteration_table(ax):
    ax.axis('off')
    rows = []
    col_labels = ['ω', 'Prob', '#Src', 'A (none)', 'C (ILU)', 'F (Neural)']
    for omega in OMEGAS:
        bm = load_benchmark(omega)
        if not bm:
            rows.append([f'ω={omega}', '—', '—', '—', '—', '(no data)'])
            continue
        for pidx, prob in enumerate(bm.get('problems', [])):
            n_src = prob.get('n_src', prob.get('n_sources', '?'))
            cells = [f'ω={omega}', str(pidx+1), str(n_src)]
            for key in ['A', 'C', 'F']:
                r = prob.get(key, {})
                it = r.get('iters', '?')
                ok = r.get('converged', False)
                cells.append(f'{it} {"✓" if ok else "✗"}')
            rows.append(cells)
    if not rows:
        ax.text(0.5, 0.5, '(run benchmark_precond_unet.py first)',
                ha='center', va='center', transform=ax.transAxes, color='gray', fontsize=11)
        ax.set_title('FGMRES Iteration Counts', fontsize=10)
        return
    t = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.7)
    for j in range(len(col_labels)):
        t[(0, j)].set_facecolor('#37474F')
        t[(0, j)].set_text_props(color='white', fontweight='bold')
    for i, row in enumerate(rows):
        for j, cell in enumerate(row[3:], start=3):
            if '✓' in cell:
                t[(i+1, j)].set_facecolor('#C8E6C9')
            elif '✗' in cell:
                t[(i+1, j)].set_facecolor('#FFCDD2')
    ax.set_title('FGMRES Iteration Counts — Neural UNet Preconditioner', fontsize=10, pad=14)


# ── status summary ─────────────────────────────────────────────────────────────

def plot_status(ax):
    ax.axis('off')
    lines = ['Training Status\n' + '─'*28]
    for omega in OMEGAS:
        d = load_training_log(omega)
        if d:
            ep  = d['epochs'][-1] if d['epochs'] else 0
            lines.append(f'ω={omega:3d}: epoch={ep:3d}  best={d["best_val"]:.2e} @ep{d["best_epoch"]}')
        else:
            lines.append(f'ω={omega:3d}: (log not found)')
    lines.append('')
    lines.append('Benchmark Status\n' + '─'*28)
    for omega in OMEGAS:
        bm = load_benchmark(omega)
        if bm:
            probs = bm.get('problems', [])
            n_conv = sum(
                1 for p in probs
                for key in ['A', 'C', 'F']
                for r in [p.get(key, {})]
                if r.get('converged')
            )
            lines.append(f'ω={omega:3d}: {len(probs)} problems  {n_conv} conv')
        else:
            lines.append(f'ω={omega:3d}: (not run)')
    ax.text(0.05, 0.97, '\n'.join(lines), transform=ax.transAxes,
            fontsize=7.5, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9))
    ax.set_title('Run Status', fontsize=9)


# ── combined make_fig4 ─────────────────────────────────────────────────────────

def make_fig4():
    """
    Main combined figure — call this from morning_eval.sh or interactively.
    Returns the path to the saved PNG.
    """
    print('\n── make_fig4(): combined professor figure ──')

    fig = plt.figure(figsize=(20, 10))
    gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.50, wspace=0.38,
                            height_ratios=[1.3, 1.1])

    # Training curves
    ax0 = fig.add_subplot(gs[0, 0])
    plot_training_curves(ax0)

    # FGMRES residuals ω=32 (problems 1-3)
    res_axes = [fig.add_subplot(gs[0, 1+i]) for i in range(3)]
    plot_residuals_omega(32, res_axes)
    res_axes[1].set_title(f'FGMRES ω=32  — ' + res_axes[1].get_title(), fontsize=9)

    # Status panel
    ax_stat = fig.add_subplot(gs[0, 4])
    plot_status(ax_stat)

    # Iteration table (full bottom row)
    ax_tab = fig.add_subplot(gs[1, :])
    plot_iteration_table(ax_tab)

    fig.suptitle(
        'Neural UNet Preconditioner for Helmholtz FGMRES  ·  N=512×512',
        fontsize=14, fontweight='bold', y=1.02,
    )

    out = OUTDIR / 'fig4_combined.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out}')
    return str(out)


# ── standalone figures ─────────────────────────────────────────────────────────

def make_fig1():
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_training_curves(ax)
    plt.tight_layout()
    out = OUTDIR / 'fig1_training_curves.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved → {out}')


def make_fig2(omega: int = 32):
    bm = load_benchmark(omega)
    n = len(bm.get('problems', [])) if bm else 0
    n_show = max(n, 3)
    fig, axes = plt.subplots(1, n_show, figsize=(6*n_show, 4), sharey=True)
    if n_show == 1: axes = [axes]
    plot_residuals_omega(omega, list(axes))
    plt.suptitle(f'FGMRES Convergence — ω={omega}', fontsize=12)
    plt.tight_layout()
    out = OUTDIR / f'fig2_fgmres_omega{omega}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved → {out}')


def make_fig3():
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_iteration_table(ax)
    plt.tight_layout()
    out = OUTDIR / 'fig3_iteration_table.png'
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved → {out}')


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--fig', choices=['1','2','3','4','all'], default='all')
    p.add_argument('--omega', type=int, default=32, help='ω for fig2')
    args = p.parse_args()

    print(f'Output → {OUTDIR}')
    if args.fig in ('1', 'all'): make_fig1()
    if args.fig in ('2', 'all'): make_fig2(args.omega)
    if args.fig in ('3', 'all'): make_fig3()
    if args.fig in ('4', 'all'): make_fig4()
    print(f'\nDone.  Open: {OUTDIR}')
