#!/bin/bash
set -euo pipefail

watch -n 15 '
squeue -u $USER
echo
echo "Recent baseline logs:"
ls -1 ~/Freq2Transfer/experiments/claude/precond_v3/launch/logs 2>/dev/null | tail -n 12
echo
echo "Recent training outputs:"
ls -d /orcd/scratch/orcd/006/fkiewiet/freq2transfer/precond_v3_runs/pair_* 2>/dev/null | sort
echo
echo "Recent warm-start eval outputs:"
ls -d ~/Freq2Transfer/experiments/warmstart_gmres/runs/omega*_csl_precond_v3_unet_seed* 2>/dev/null | sort
'
