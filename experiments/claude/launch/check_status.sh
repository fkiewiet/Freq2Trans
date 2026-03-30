#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# check_status.sh  —  quick overview of all running/completed experiments
#
# Run from anywhere:
#   bash experiments/claude/launch/check_status.sh
# ══════════════════════════════════════════════════════════════════════════════

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNS="${REPO_ROOT}/experiments/claude/unet_hparam/runs"
RES="${REPO_ROOT}/experiments/claude/results_transfer"

RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'; BLU='\033[0;34m'; NC='\033[0m'

echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  EXPERIMENT STATUS  —  $(date)"
echo "══════════════════════════════════════════════════════════════════════"

# ── GPU status ─────────────────────────────────────────────────────────────
echo ""
echo "  GPUs:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '{
        util=$2; mem_used=$3; mem_tot=$4
        pct=int(mem_used*100/mem_tot)
        status="free"
        if (util > 5) status="BUSY"
        printf "    cuda:%-2s  util=%3s%%  mem=%5s/%5s MiB (%3s%%)  [%s]\n",
            $1, util, mem_used, mem_tot, pct, status
    }'

# ── helper: show last line of metrics.csv ─────────────────────────────────
show_run() {
    local LABEL="$1"
    local DIR="$2"
    local TARGET="$3"
    local CSV="${DIR}/metrics.csv"
    local LOG=$(ls "${DIR}"/../../../launch/logs/${LABEL}_*.log 2>/dev/null | sort | tail -1)

    printf "  %-18s  " "${LABEL}"

    if [ ! -f "${CSV}" ]; then
        printf "${RED}NOT STARTED${NC}\n"
        return
    fi

    local N=$(( $(wc -l < "${CSV}") - 1 ))   # subtract header
    if [ "${N}" -lt 1 ]; then
        printf "${YLW}0 epochs written yet${NC}\n"
        return
    fi

    local LAST=$(tail -1 "${CSV}")
    local EP=$(echo "${LAST}" | cut -d',' -f1)
    local VAL_RE=$(echo "${LAST}" | cut -d',' -f5)

    # Best val_re ever
    local BEST=$(tail -n +2 "${CSV}" | cut -d',' -f5 | sort -n | head -1)

    local STATUS=""
    if [ "${N}" -ge "${TARGET}" ]; then
        STATUS="${GRN}COMPLETE${NC}"
    else
        # Check if still running (log modified in last 10 min)
        if [ -n "${LOG}" ] && [ $(( $(date +%s) - $(stat -c %Y "${LOG}" 2>/dev/null || echo 0) )) -lt 600 ]; then
            STATUS="${BLU}running${NC}"
        else
            STATUS="${RED}STALLED?${NC}"
        fi
    fi

    printf "ep %4d/%-4d  val_re(last)=%.4f  best=%.4f  %b\n" \
        "${EP}" "${TARGET}" "${VAL_RE}" "${BEST}" "${STATUS}"
}

# ── UNet training runs ─────────────────────────────────────────────────────
echo ""
echo "  UNet training runs  (tmux: train3000):"
echo "  ──────────────────────────────────────────────────────────────────"
echo "  Run                    Ep/Target     val_re(last)  best      Status"
echo "  ──────────────────────────────────────────────────────────────────"
echo "  — T_up operators (low→high) —"
show_run "H_3000ep"            "${RUNS}/H_3000ep"            3000
show_run "C_3000ep"            "${RUNS}/C_3000ep"            3000
show_run "N_3000ep"            "${RUNS}/N_3000ep"            3000
show_run "H_n4800_3000ep"      "${RUNS}/H_n4800_3000ep"      3000
echo "  — T_down operators (high→low) —"
show_run "H_down_3000ep"       "${RUNS}/H_down_3000ep"       3000
show_run "C_down_3000ep"       "${RUNS}/C_down_3000ep"       3000
show_run "N_down_3000ep"       "${RUNS}/N_down_3000ep"       3000
show_run "H_down_n4800_3000ep" "${RUNS}/H_down_n4800_3000ep" 3000

# ── Baseline UNet (separate, already running) ──────────────────────────────
echo ""
echo "  Baseline UNet (29ch, 500ep, reference):"
echo "  ──────────────────────────────────────────────────────────────────"
show_run "run_29ch"  "${REPO_ROOT}/experiments/claude/unet/run_29ch"  500

# ── GMRES results ──────────────────────────────────────────────────────────
echo ""
echo "  GMRES v5 results  (tmux: gmres):"
echo "  ──────────────────────────────────────────────────────────────────"
echo "  — golden weights (running now on CPU) —"
for PAIR in 16_32 32_64 64_128; do
    JSON="${RES}/precond_gmres_v5_${PAIR}/results_v5.json"
    printf "  %-28s  " "gmres_v5_golden_${PAIR}"
    if [ -f "${JSON}" ]; then
        E_SU=$(python3 -c "
import json, math
d=json.load(open('${JSON}'))
probs=d['problems']
sus=[p.get('speedup_E',1) for p in probs]
print(f'{math.exp(sum(math.log(s+1e-9) for s in sus)/len(sus)):.2f}')
" 2>/dev/null || echo "?")
        printf "${GRN}DONE${NC}  neural speedup E=%.2fx\n" "${E_SU}"
    else
        # Check if in progress (log file active)
        GLOG=$(ls "${LOGDIR}"/gmres_v5_golden_*.log 2>/dev/null | sort | tail -1)
        if [ -n "${GLOG}" ] && [ $(( $(date +%s) - $(stat -c %Y "${GLOG}" 2>/dev/null || echo 0) )) -lt 600 ]; then
            printf "${BLU}running${NC}\n"
        else
            printf "${YLW}not started${NC}\n"
        fi
    fi
done
echo "  — UNet weights (run after training finishes) —"
for PAIR in 16_32 32_64 64_128; do
    JSON="${RES}/precond_gmres_v5_unet_${PAIR}/results_v5.json"
    printf "  %-28s  " "gmres_v5_unet_${PAIR}"
    [ -f "${JSON}" ] && printf "${GRN}DONE${NC}\n" || printf "${YLW}not started${NC}\n"
done

# ── What to do next ────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  NEXT STEPS:"
echo ""

# Check if core 4 training runs done (H + H_down are what GMRES needs)
ALL_DONE=true
for RUN in H_3000ep H_down_3000ep; do
    CSV="${RUNS}/${RUN}/metrics.csv"
    if [ ! -f "${CSV}" ]; then ALL_DONE=false; break; fi
    N=$(( $(wc -l < "${CSV}") - 1 ))
    if [ "${N}" -lt 3000 ]; then ALL_DONE=false; break; fi
done

if ${ALL_DONE}; then
    echo -e "  ${GRN}ALL 4 TRAINING RUNS COMPLETE.${NC}"
    echo "  → Run:  bash experiments/claude/launch/launch_gmres_v5_unet.sh"
    echo "  → Then: python experiments/claude/eval_long_runs.py"
else
    echo "  Waiting for training runs to finish. Check back later."
    echo "  To watch a run live: tmux attach -t train3000"
fi

# Check if GMRES done
ALL_GMRES=true
for PAIR in 16_32 32_64 64_128; do
    [ -f "${RES}/precond_gmres_v5_${PAIR}/results_v5.json" ] || { ALL_GMRES=false; break; }
done
if ${ALL_GMRES}; then
    echo ""
    echo -e "  ${GRN}ALL GMRES v5 RUNS COMPLETE.${NC}"
    echo "  → Run:  python experiments/claude/eval_gmres_v5.py"
fi

echo "══════════════════════════════════════════════════════════════════════"
echo ""
