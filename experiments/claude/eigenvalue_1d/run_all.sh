#!/bin/bash
# run_all.sh — 1D warm-start analysis pipeline (3-approach comparison).
#
# Trains four T_up models and runs a full comparison:
#   Pert           — 1st-order perturbation theory (analytical reference)
#   A  raw         — green-fn trained, no PML zeroing (broken baseline)
#   B  zero_pml    — green-fn trained, PML strip zeroed at inference
#   C  pml_trained — FD/PML trained, full-grid loss, no zeroing needed
#   D  masked      — green-fn trained with PML strips zeroed in training + inference
#   E  pml_int     — FD/PML trained, interior-only loss, zeroed at inference
#                    (isolates distribution effect from loss design; compare vs B and C)
#
# Usage (from project root or any directory):
#   bash experiments/claude/eigenvalue_1d/run_all.sh              # 16→32, cpu, 500 epochs
#   bash experiments/claude/eigenvalue_1d/run_all.sh 32 64        # 32→64
#   bash experiments/claude/eigenvalue_1d/run_all.sh 16 32 cuda:0 # GPU
#   bash experiments/claude/eigenvalue_1d/run_all.sh 16 32 cuda:0 500
#
# Recommended via wrapper (all 3 pairs):
#   tmux new -s eig1d
#   bash experiments/claude/eigenvalue_1d/run_all_pairs.sh cuda:0
#   # detach: Ctrl-b d   reattach: tmux attach -t eig1d

set -e
OMEGA_L=${1:-16}
OMEGA_H=${2:-32}
DEVICE=${3:-cpu}
EPOCHS=${4:-500}   # higher-freq pairs need more epochs; 300 was sufficient for 16→32
N_SAMPLES=2400     # 2000 train + 400 val

cd "$(dirname "$0")/../../.."   # go to project root
source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "================================================================"
echo "1D Warm-Start Analysis  ω_L=$OMEGA_L → ω_H=$OMEGA_H"
echo "device=$DEVICE  epochs=$EPOCHS"
echo "Approaches: A=raw  B=zero_pml  C=pml_trained  D=masked  E=pml_int"
echo "================================================================"

CKPT_GREEN="experiments/claude/eigenvalue_1d/runs/pair_${OMEGA_L}_${OMEGA_H}/T_up/best.pt"
CKPT_PML="experiments/claude/eigenvalue_1d/runs/pair_${OMEGA_L}_${OMEGA_H}_pml/T_up/best.pt"
CKPT_PML_INT="experiments/claude/eigenvalue_1d/runs/pair_${OMEGA_L}_${OMEGA_H}_pml_int/T_up/best.pt"
CKPT_MASKED="experiments/claude/eigenvalue_1d/runs/pair_${OMEGA_L}_${OMEGA_H}_masked/T_up/best.pt"

echo ""
echo "--- Step 1a: Generate data (free-space Green's fn) ---"
python experiments/claude/eigenvalue_1d/generate_data_1d.py \
    --omega_l $OMEGA_L --omega_h $OMEGA_H --n $N_SAMPLES --seed 42 \
    --solver green

echo ""
echo "--- Step 1b: Generate data (FD/PML solver) ---"
python experiments/claude/eigenvalue_1d/generate_data_1d.py \
    --omega_l $OMEGA_L --omega_h $OMEGA_H --n $N_SAMPLES --seed 42 \
    --solver pml

echo ""
echo "--- Step 2a: Train T_up on green-fn data (approaches A + B) ---"
# No --fresh: resumes from existing checkpoint if present (avoids retraining).
python experiments/claude/eigenvalue_1d/train_1d.py \
    --omega_l $OMEGA_L --omega_h $OMEGA_H --direction up --device $DEVICE \
    --levels 5 --epochs $EPOCHS

echo ""
echo "--- Step 2b: Train T_up on FD/PML data, full-grid loss (approach C) ---"
python experiments/claude/eigenvalue_1d/train_1d.py \
    --omega_l $OMEGA_L --omega_h $OMEGA_H --direction up --device $DEVICE \
    --levels 5 --fresh --epochs $EPOCHS --tag _pml --full_grid_loss

echo ""
echo "--- Step 2c: Train T_up with PML-masked inputs (approach D) ---"
# Green-fn data, PML strips zeroed in input+target → pure interior-to-interior.
python experiments/claude/eigenvalue_1d/train_1d.py \
    --omega_l $OMEGA_L --omega_h $OMEGA_H --direction up --device $DEVICE \
    --levels 5 --fresh --epochs $EPOCHS --tag _masked --mask_pml --data_tag ""

echo ""
echo "--- Step 2d: Train T_up on FD/PML data, interior-only loss (approach E) ---"
# Reads from the _pml data directory; writes checkpoint to _pml_int run directory.
python experiments/claude/eigenvalue_1d/train_1d.py \
    --omega_l $OMEGA_L --omega_h $OMEGA_H --direction up --device $DEVICE \
    --levels 5 --fresh --epochs $EPOCHS --tag _pml_int --data_tag _pml

echo ""
echo "--- Step 3: Full warm-start comparison (cold + pert + A/B/C/D/E) ---"
python experiments/claude/eigenvalue_1d/warm_start_analysis.py \
    --omega_l      $OMEGA_L \
    --omega_h      $OMEGA_H \
    --ckpt_green   "$CKPT_GREEN" \
    --ckpt_pml     "$CKPT_PML" \
    --ckpt_pml_int "$CKPT_PML_INT" \
    --ckpt_masked  "$CKPT_MASKED" \
    --gmres_beta   0.3 \
    --device       $DEVICE

echo ""
echo "================================================================"
echo "Done.  Results → experiments/claude/eigenvalue_1d/results/"
echo "================================================================"
