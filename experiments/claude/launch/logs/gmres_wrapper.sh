#!/usr/bin/env bash
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate
echo "=== FGMRES comparison started: $(date) ==="
echo "Log: /math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/gmres_20260319_190413.log"
echo ""
PYTHONUNBUFFERED=1 /math/home/fkiewiet/Freq2Transfer/.venv/bin/python /math/home/fkiewiet/Freq2Transfer/experiments/claude/preconditioner_gmres.py 2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/gmres_20260319_190413.log'
echo ""
echo "=== Done: $(date) ==="
echo "(Ctrl-b d to detach)"
