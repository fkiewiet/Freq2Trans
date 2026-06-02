#!/bin/bash
# Canonical precond_v3 launcher:
#   - stages the required N9600 dataset to node-local storage
#   - writes logs to ORCD scratch
#   - keeps checkpoints/results on ORCD scratch
#   - composes a temporary runtime override so train.py reads the local dataset
#
# Usage examples:
#   sbatch --job-name=pcv3_up_16_32 \
#     --export=ALL,CONFIG=experiments/claude/precond_v3/configs/pair_16_32.yaml,DIRECTION=up \
#     experiments/claude/precond_v3/launch/sbatch_pair_up_staged.sh
#
#   sbatch --job-name=pcv3_down_32_64 \
#     --export=ALL,CONFIG=experiments/claude/precond_v3/configs/pair_32_64.yaml,DIRECTION=down \
#     experiments/claude/precond_v3/launch/sbatch_pair_up_staged.sh

#SBATCH --job-name=pcv3_pair_staged
#SBATCH --output=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.log
#SBATCH --error=/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$HOME/Freq2Transfer}"
LOG_DIR="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/slurm_logs"
RUN_ROOT="/orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_v3_runs"
POOL_ROOT="/orcd/pool/006/fkiewiet/freq2transfer/datasets_N9600"
LOCAL_DATA_ROOT="${SLURM_TMPDIR:-/tmp/$USER}/datasets_N9600"

CONFIG="${CONFIG:-experiments/claude/precond_v3/configs/pair_16_32.yaml}"
DIRECTION="${DIRECTION:-up}"
MAX_RUNTIME_H="${MAX_RUNTIME_H:-10.5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXPERIMENT_OUTDIR="${EXPERIMENT_OUTDIR:-$RUN_ROOT}"

cd "$ROOT"
mkdir -p "$LOG_DIR" "$RUN_ROOT" "$LOCAL_DATA_ROOT"

if [ -f "$ROOT/.venv/bin/activate" ]; then
    source "$ROOT/.venv/bin/activate"
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

if [ "$DIRECTION" = "up" ]; then
    DATASET_NAME="up_N9600_seed42_repaired"
elif [ "$DIRECTION" = "down" ]; then
    DATASET_NAME="down_N9600_seed42"
else
    echo "ERROR: unsupported direction '$DIRECTION' (expected up or down)" >&2
    exit 1
fi

PAIR_TAG="$(basename "$CONFIG" .yaml)"
PAIR_TAG="${PAIR_TAG#pair_}"
LIVE_OVERRIDE="$ROOT/experiments/claude/precond_v3/configs/live/pair_${PAIR_TAG}_override.yaml"
TMP_OVERRIDE="/tmp/precond_v3_${DIRECTION}_${PAIR_TAG}_${SLURM_JOB_ID:-manual}_override.yaml"

DS_SRC="$POOL_ROOT/$DATASET_NAME"
DS_LOCAL="$LOCAL_DATA_ROOT/$DATASET_NAME"
DS_LOCK="$LOCAL_DATA_ROOT/.${DATASET_NAME}.stage.lock"

dataset_ready() {
    local d="$1"
    for f in COMPLETE metadata.json u_low_re.npy u_low_im.npy u_high_re.npy u_high_im.npy rms.npy omega_low.npy source_re.npy; do
        if [ ! -f "$d/$f" ]; then
            return 1
        fi
    done
    return 0
}

echo "========================================================"
echo "precond_v3 staged launcher"
echo "job id      : ${SLURM_JOB_ID:-unknown}"
echo "job name    : ${SLURM_JOB_NAME:-unknown}"
echo "host        : $(hostname)"
echo "cwd         : $(pwd)"
echo "python      : $(which python3)"
echo "date        : $(date)"
echo "partition   : ${SLURM_JOB_PARTITION:-unknown}"
echo "job gpus    : ${SLURM_JOB_GPUS:-unset}"
echo "node gpus   : ${SLURM_GPUS_ON_NODE:-unset}"
echo "cuda vis    : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "config      : $CONFIG"
echo "direction   : $DIRECTION"
echo "pair tag    : $PAIR_TAG"
echo "pool ds     : $DS_SRC"
echo "local ds    : $DS_LOCAL"
echo "run root    : $EXPERIMENT_OUTDIR"
echo "log dir     : $LOG_DIR"
echo "========================================================"
echo ""

if ! dataset_ready "$DS_SRC"; then
    echo "ERROR: pooled dataset not found: $DS_SRC" >&2
    echo "Expected metadata.json and all required .npy arrays." >&2
    exit 1
fi

if ! dataset_ready "$DS_LOCAL"; then
    echo "[stage] local dataset missing or incomplete; staging to node-local storage"
    while ! mkdir "$DS_LOCK" 2>/dev/null; do
        echo "[stage] another job is staging $DATASET_NAME; waiting for lock: $DS_LOCK"
        sleep 30
        if dataset_ready "$DS_LOCAL"; then
            break
        fi
    done
    if [ -d "$DS_LOCK" ]; then
        trap 'rmdir "$DS_LOCK" 2>/dev/null || true' EXIT
        if ! dataset_ready "$DS_LOCAL"; then
            mkdir -p "$DS_LOCAL"
            rsync -a --delete --info=progress2 "$DS_SRC/" "$DS_LOCAL/"
        fi
        rmdir "$DS_LOCK" 2>/dev/null || true
        trap - EXIT
    fi
else
    echo "[stage] reusing complete local dataset: $DS_LOCAL"
fi

if ! dataset_ready "$DS_LOCAL"; then
    echo "ERROR: local dataset is still incomplete after staging: $DS_LOCAL" >&2
    exit 1
fi
echo ""

{
    if [ -f "$LIVE_OVERRIDE" ]; then
        cat "$LIVE_OVERRIDE"
        echo
    fi
    echo "outdir: $EXPERIMENT_OUTDIR"
    echo
    echo "datasets:"
    if [ "$DIRECTION" = "up" ]; then
        echo "  up_dir: $DS_LOCAL"
    else
        echo "  down_dir: $DS_LOCAL"
    fi
} > "$TMP_OVERRIDE"

echo "[stage] runtime override written to: $TMP_OVERRIDE"
echo "[stage] override contents:"
sed -n '1,120p' "$TMP_OVERRIDE"
echo ""

nvidia-smi -L || true
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true
echo ""

PYTHONUNBUFFERED=1 python3 "$ROOT/experiments/claude/precond_v3/train.py" \
    --config "$CONFIG" \
    --override_config "$TMP_OVERRIDE" \
    --direction "$DIRECTION" \
    --device cuda:0 \
    --num_workers "$NUM_WORKERS" \
    --max_runtime_h "$MAX_RUNTIME_H" \
    ${EXTRA_ARGS}
TRAIN_EXIT=$?

echo ""
echo "Run complete: $(date)  exit_code=$TRAIN_EXIT"

# ── per-run summary + auto-resubmit ──────────────────────────────────────────
RUN_DIR="$EXPERIMENT_OUTDIR/pair_${PAIR_TAG}/T_${DIRECTION}"
SUMMARY_JSON="$RUN_DIR/summary.json"
RESUME_COUNT_FILE="$RUN_DIR/.resume_count"
MAX_RESUMES=10

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  RUN SUMMARY"
echo "  job      : ${SLURM_JOB_NAME:-unknown}  id=${SLURM_JOB_ID:-?}"
echo "  pair     : ${PAIR_TAG}  direction=${DIRECTION}"
echo "  run_dir  : $RUN_DIR"

if [ -f "$SUMMARY_JSON" ]; then
    python3 - <<PYEOF
import json, sys
try:
    d = json.load(open("$SUMMARY_JSON"))
    r = d.get("stopped_reason", "unknown")
    print(f"  stopped  : {r}")
    print(f"  epoch    : {d.get('last_epoch','?')}  best_epoch={d.get('best_epoch','?')}")
    print(f"  best_val : {d.get('best_val_loss','?'):.6f}")
    print(f"  test_loss: {d.get('test_loss_at_best','?'):.6f}")
    print(f"  log_csv  : {d.get('log_csv','?')}")
except Exception as e:
    print(f"  [summary parse error: {e}]")
PYEOF
    STOPPED_REASON=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON')).get('stopped_reason','unknown'))" 2>/dev/null || echo "unknown")
else
    echo "  [no summary.json found — training may have crashed]"
    STOPPED_REASON="crash"
fi

COUNT=$(cat "$RESUME_COUNT_FILE" 2>/dev/null || echo 0)
echo "  resumes  : $COUNT / $MAX_RESUMES"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

if [ "$STOPPED_REASON" = "runtime_cap" ] && [ "$COUNT" -lt "$MAX_RESUMES" ]; then
    NEW=$((COUNT + 1))
    echo "$NEW" > "$RESUME_COUNT_FILE"
    echo "→ Auto-resubmitting (attempt $NEW of $MAX_RESUMES)..."
    sbatch \
        --requeue \
        --job-name="${SLURM_JOB_NAME:-pcv3_auto}" \
        --output="${LOG_DIR}/${SLURM_JOB_NAME:-pcv3_auto}_%j.log" \
        --error="${LOG_DIR}/${SLURM_JOB_NAME:-pcv3_auto}_%j.err" \
        --time=12:00:00 \
        --partition=mit_preemptable \
        --gres=gpu:1 \
        --cpus-per-task=4 \
        --mem=64G \
        --nodes=1 \
        --ntasks=1 \
        --export=ALL \
        "$ROOT/experiments/claude/precond_v3/launch/sbatch_pair_up_staged.sh"
elif [ "$STOPPED_REASON" = "runtime_cap" ]; then
    echo "→ Max resumes ($MAX_RESUMES) reached. Training stopped."
else
    echo "→ Training finished ($STOPPED_REASON). Not resubmitting."
    rm -f "$RESUME_COUNT_FILE"

    # ── post-training: loss curve ─────────────────────────────────────────────
    LOG_CSV="$RUN_DIR/log.csv"
    if [ -f "$LOG_CSV" ]; then
        echo ""
        echo "Generating training curve plot..."
        python3 "$ROOT/experiments/claude/precond_v3/plot_training.py" \
            --log "$LOG_CSV" \
            --outdir "$RUN_DIR" \
        && echo "  → $RUN_DIR/training_curve.png" \
        || echo "  [plot_training.py failed — skipping]"
    fi

    # ── post-training: FGMRES warm-start benchmark ────────────────────────────
    BEST_PT="$RUN_DIR/best.pt"
    if [ -f "$BEST_PT" ]; then
        echo ""
        echo "Running FGMRES warm-start benchmark..."
        BENCH_OUTDIR="$RUN_DIR/benchmark"
        mkdir -p "$BENCH_OUTDIR"
        set +e
        python3 "$ROOT/experiments/claude/benchmark_warmstart_unet.py" \
            --ckpt "$BEST_PT" \
            --device cuda:0 \
            --outdir "$BENCH_OUTDIR" \
            --n_problems 5 \
            --n_iters 60 \
            --seed 77777
        BENCH_RC=$?
        set -e
        if [ "$BENCH_RC" -eq 0 ]; then
            echo "  → $BENCH_OUTDIR/convergence.png"
            echo "  → $BENCH_OUTDIR/snapshots.png"
            echo "  → $BENCH_OUTDIR/results.json"
        else
            echo "  [benchmark failed (exit $BENCH_RC) — training output is still saved]"
        fi
    else
        echo "  [no best.pt found — skipping FGMRES benchmark]"
    fi
fi
