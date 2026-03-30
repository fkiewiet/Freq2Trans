#!/usr/bin/env bash
cd '/math/home/fkiewiet/Freq2Transfer'
source .venv/bin/activate
echo "=== GMRES vs FGMRES (VORONOI-LOOKaLIKE-1703) started: $(date) ==="
echo "Log: /math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/fgmres_voronoi_20260320_102026.log"
echo ""
PYTHONUNBUFFERED=1 /math/home/fkiewiet/Freq2Transfer/.venv/bin/python /math/home/fkiewiet/Freq2Transfer/experiments/claude/fgmres_comparison.py 2>&1 | tee '/math/home/fkiewiet/Freq2Transfer/experiments/claude/launch/logs/fgmres_voronoi_20260320_102026.log'
echo ""
echo "=== Done: $(date) ==="
echo "(Ctrl-b d to detach)"
