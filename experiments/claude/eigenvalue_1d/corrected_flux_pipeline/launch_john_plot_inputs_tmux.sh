#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$HOME/Freq2Transfer}"
PIPE="$ROOT/experiments/claude/eigenvalue_1d/corrected_flux_pipeline"
SCRIPT="$PIPE/john_plot_inputs.py"
LOG_DIR="${LOG_DIR:-/tmp/$USER/john_plot_inputs_logs}"
mkdir -p "$LOG_DIR"

OMEGA_L="${OMEGA_L:-16}"
OMEGA_H="${OMEGA_H:-32}"
N_GRID="${N_GRID:-512}"
DEVICE="${DEVICE:-cpu}"
N_SAMPLES="${N_SAMPLES:-20}"
OUT_ROOT="${OUT_ROOT:-$PIPE/outputs_john_plots}"

CKPT_FLUX_FULL="${CKPT_FLUX_FULL:-$PIPE/outputs/runs/pair_16_32_flux_full/T_up/best.pt}"
CKPT_GREEN="${CKPT_GREEN:-$PIPE/outputs_dirichlet_latest/runs_train1d/pair_16_32_dirichlet_n512/T_up/best.pt}"

launch_plot() {
    local plot="$1"
    local slug
    case "$plot" in
        1) slug="residual_energy_spectrum" ;;
        2) slug="true_pml_eigenvalues" ;;
        3) slug="csl_pa_spectrum_beta03" ;;
        4) slug="gmres_modal_heatmap" ;;
        5) slug="residual_gate_decisions" ;;
        *) echo "unknown plot: $plot" >&2; exit 2 ;;
    esac
    local session="john_p${plot}_${slug}_w${OMEGA_L}_${OMEGA_H}_n${N_GRID}"
    local log="$LOG_DIR/${session}_$(date +%Y%m%d_%H%M%S).log"
    local cmd
    cmd="cd '$ROOT'"
    if [ -f "$ROOT/.venv/bin/activate" ]; then
        cmd="$cmd && source '$ROOT/.venv/bin/activate'"
    fi
    cmd="$cmd && python -u '$SCRIPT' --plot '$plot' --omega_l '$OMEGA_L' --omega_h '$OMEGA_H' --n_grid '$N_GRID' --device '$DEVICE' --n_samples '$N_SAMPLES' --out_root '$OUT_ROOT' --ckpt_flux_full '$CKPT_FLUX_FULL' --ckpt_green '$CKPT_GREEN' 2>&1 | tee '$log'"
    tmux new-session -d -s "$session" "$cmd"
    echo "launched $session"
    echo "  log: $log"
}

for plot in 1 2 3 4 5; do
    launch_plot "$plot"
done

echo
echo "watch with: tmux ls"
echo "attach with: tmux attach -t john_p1_residual_energy_spectrum_w${OMEGA_L}_${OMEGA_H}_n${N_GRID}"
echo "outputs: $OUT_ROOT/pair_${OMEGA_L}_${OMEGA_H}_n${N_GRID}"
