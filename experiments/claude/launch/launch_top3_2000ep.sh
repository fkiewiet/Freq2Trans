#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# launch_top3_2000ep.sh
#
# Launches long training runs for the top-3 HPO configs to understand
# convergence behaviour and find the error floor.
#
# Scientific rigour takes priority over the 4-day window; runs go to 2000ep
# even if they don't finish within 4 days.
#
#   Trial H — 32ch, n=2400, bs=8, lr=1e-4   (~207s/ep → ~4.8 days for 2000ep)
#   Trial C — 64ch, n=1200, bs=4, lr=1e-4   (~243s/ep → ~5.6 days for 2000ep)
#   Trial N — 64ch, n=1200, bs=4, lr=3e-4   (~243s/ep → ~5.6 days for 2000ep)
#
# Output dirs:
#   experiments/claude/unet_hparam/runs/H_2000ep/
#   experiments/claude/unet_hparam/runs/C_2000ep/
#   experiments/claude/unet_hparam/runs/N_2000ep/
#
# Each run writes:
#   metrics.csv      — epoch, tr_total, tr_re, tr_im, val_re, val_im (every epoch)
#   best.pt          — best model by val_re
#   log.txt          — full training log
#
# After the run, use:
#   python experiments/claude/eval_long_runs.py
# to get comparison plots and a summary table.
#
# Usage:
#   bash experiments/claude/launch/launch_top3_2000ep.sh
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/experiments/claude/unet_hparam/train_unet_hparam.py"
DATASET="${REPO_ROOT}/experiments/claude/datasets/up_N4800_seed42"
OUTBASE="${REPO_ROOT}/experiments/claude/unet_hparam/runs"
LOGDIR="${REPO_ROOT}/experiments/claude/launch/logs"
SESSION="top3_long"

mkdir -p "${LOGDIR}"

# ── helper to launch one trial ─────────────────────────────────────────────
launch_trial() {
    local TRIAL="$1"
    local OUTDIR="$2"
    local DEVICE="$3"
    local EPOCHS="$4"
    shift 4
    local EXTRA_FLAGS="$@"

    local LOG="${LOGDIR}/${TRIAL}_$(date +%Y%m%d_%H%M%S).log"
    local WRAP="${LOGDIR}/${TRIAL}_wrapper.sh"

    mkdir -p "${OUTDIR}/plots" "${OUTDIR}/checkpoints"

    cat > "${WRAP}" <<WEOF
#!/usr/bin/env bash
cd '${REPO_ROOT}'
source .venv/bin/activate
echo "=== ${TRIAL} started: \$(date) ==="
echo "Epochs: ${EPOCHS} | Device: ${DEVICE}"
echo "Output: ${OUTDIR}"
echo ""
PYTHONUNBUFFERED=1 ${PYTHON} ${SCRIPT} \\
    --dataset '${DATASET}' \\
    --outdir  '${OUTDIR}' \\
    --device  '${DEVICE}' \\
    --max_epochs ${EPOCHS} \\
    --yes \\
    ${EXTRA_FLAGS} 2>&1 | tee '${LOG}'
echo ""
echo "=== ${TRIAL} done: \$(date) ==="
WEOF
    chmod +x "${WRAP}"

    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        tmux kill-window -t "${SESSION}:${TRIAL}" 2>/dev/null || true
    else
        tmux new-session -d -s "${SESSION}"
    fi
    tmux new-window -t "${SESSION}" -n "${TRIAL}"
    tmux send-keys -t "${SESSION}:${TRIAL}" "bash '${WRAP}'" Enter
    echo "  Launched ${TRIAL} on ${DEVICE} → tmux ${SESSION}:${TRIAL}"
}

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         Long-run training: top-3 HPO configs                    ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  H:  32ch, n=2400, bs=8, lr=1e-4   → 2000ep on cuda:1          ║"
echo "║  C:  64ch, n=1200, bs=4, lr=1e-4   → 2000ep on cuda:2          ║"
echo "║  N:  64ch, n=1200, bs=4, lr=3e-4   → 2000ep on cuda:3          ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  tmux session: ${SESSION}                                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ── Trial H — winner: more data, smaller model ────────────────────────────
launch_trial "H_2000ep" \
    "${OUTBASE}/H_2000ep" \
    "cuda:1" \
    2000 \
    "--n_per_pair 2400 --batch_size 8 --base_ch 32 --levels 4 --lr 1e-4"

# ── Trial C — larger model, standard data ─────────────────────────────────
launch_trial "C_2000ep" \
    "${OUTBASE}/C_2000ep" \
    "cuda:2" \
    2000 \
    "--n_per_pair 1200 --batch_size 4 --base_ch 64 --levels 4 --lr 1e-4"

# ── Trial N — larger model, higher lr ─────────────────────────────────────
launch_trial "N_2000ep" \
    "${OUTBASE}/N_2000ep" \
    "cuda:3" \
    2000 \
    "--n_per_pair 1200 --batch_size 4 --base_ch 64 --levels 4 --lr 3e-4"

echo ""
echo "All 3 runs launched. To monitor:"
echo "  tmux attach -t ${SESSION}"
echo "  Ctrl-b w  → switch windows"
echo ""
echo "Tail logs from outside tmux:"
echo "  tail -f ${LOGDIR}/H_2000ep_*.log"
echo "  tail -f ${LOGDIR}/C_1200ep_*.log"
echo "  tail -f ${LOGDIR}/N_1200ep_*.log"
echo ""
echo "After runs complete, evaluate with:"
echo "  python experiments/claude/eval_long_runs.py"
