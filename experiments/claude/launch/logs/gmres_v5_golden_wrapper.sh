#!/usr/bin/env bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
source .venv/bin/activate

run_pair() {
    local OL="$1" OH="$2"
    echo ""
    echo "████████████ GMRES v5 golden: ω=${OL}→${OH}  $(date) ████████████"
    PYTHONUNBUFFERED=1 python experiments/claude/preconditioner_gmres_v5.py \
        --omega_l "${OL}" --omega_h "${OH}"
    echo "████████████ Done: ω=${OL}→${OH}  $(date) ████████████"
}

run_pair 16 32
run_pair 32 64
run_pair 64 128

echo ""
echo "════════════════════════════════════════════════"
echo "  ALL GMRES v5 (golden) DONE: $(date)"
echo "  Evaluate: python experiments/claude/eval_gmres_v5.py"
echo "════════════════════════════════════════════════"
