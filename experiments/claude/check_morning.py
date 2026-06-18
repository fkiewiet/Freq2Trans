#!/usr/bin/env python3
"""Morning status check — run this to see where all overnight experiments are.

Usage:
    python experiments/claude/check_morning.py
"""
import os, sys, torch, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_ckpt(path):
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def tmux_tail(session, n=5):
    try:
        out = subprocess.check_output(
            ["tmux", "capture-pane", "-pt", session, "-S", f"-{n*3}"],
            stderr=subprocess.DEVNULL, text=True)
        lines = [l for l in out.splitlines() if l.strip()]
        return lines[-n:] if lines else ["(no output)"]
    except Exception:
        return ["(session not found)"]


def log_tail(path, n=6):
    if not os.path.exists(path):
        return ["(no log)"]
    with open(path) as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines[-n:] if l.strip()]


print("=" * 70)
print("  MORNING STATUS CHECK")
print("=" * 70)

# ── Experiment 1: Warm-start ────────────────────────────────────────────────
print("\n── Exp 1: Warm-start  (goal: val < 0.20) ──")
WS = os.path.join(ROOT, "experiments/claude/warmstart_hetero_1d")

configs = [
    ("warmstart_resid",  "runs_6ch_resid/warmstart_best.pt",  "runs_6ch_resid/train_log.txt",
     "T: [u_L,u_mid,f] → (u_H-u_mid)/‖u_L‖   MOST PROMISING"),
    ("warmstart_b32",    "runs_6ch_b32/warmstart_best.pt",    None,
     "T: [u_L,u_mid,f] → u_H/‖u_L‖  6ch b32"),
    ("warmstart_4ch_f",  "runs_4ch_f_b32/warmstart_best.pt",  None,
     "T: [u_L,f] → u_H/‖u_L‖  4ch b32"),
    ("runs_6ch (done)",  "runs_6ch/warmstart_best.pt",         None,
     "T: [u_L] → u_H/‖u_L‖  6ch b16  (finished)"),
]

for name, ckpt_rel, log_rel, desc in configs:
    ck = load_ckpt(os.path.join(WS, ckpt_rel))
    if ck:
        ep  = ck.get("epoch", "?")
        val = ck.get("val_loss", ck.get("val", "?"))
        gate = "  *** GATE ***" if isinstance(val, float) and val < 0.20 else ""
        print(f"  {name:<22}  ep={ep:>4}  val={float(val):.4f}{gate}")
        print(f"    {desc}")
    else:
        print(f"  {name:<22}  (no checkpoint)")

    if log_rel:
        print("  Recent log:")
        for l in log_tail(os.path.join(WS, log_rel)):
            print(f"    {l}")
    print()

# ── Experiment 2&3: Neural V-cycle ──────────────────────────────────────────
print("── Exp 2&3: Neural V-cycle  (goal: val < 0.20) ──")
PIPE = os.path.join(ROOT,
    "experiments/claude/eigenvalue_1d/corrected_flux_pipeline")

vcycle_configs = [
    ("tdown_only",    "runs_tdown_only/tdown_best.pt",  "tdown_only_log.txt",
     "T_down: r → A_L·A_H⁻¹(r)"),
    ("tup_only",      "runs_tup_only/tup_best.pt",      "tup_only_log.txt",
     "T_up: [A_L⁻¹r, r] → A_H⁻¹(r)"),
    ("vcycle_joint",  "runs_vcycle_joint/vcycle_best.pt","vcycle_joint_log.txt",
     "Joint T_down+T_up E2E"),
]

for name, ckpt_rel, log_rel, desc in vcycle_configs:
    ck = load_ckpt(os.path.join(PIPE, ckpt_rel))
    if ck:
        ep  = ck.get("epoch", "?")
        val = ck.get("val_loss", ck.get("val", "?"))
        gate = "  *** GATE ***" if isinstance(val, float) and val < 0.20 else ""
        print(f"  {name:<22}  ep={ep:>4}  val={float(val):.4f}{gate}")
        print(f"    {desc}")
    else:
        print(f"  {name:<22}  (no checkpoint)")

    if log_rel:
        print("  Recent log:")
        for l in log_tail(os.path.join(PIPE, log_rel)):
            print(f"    {l}")
    print()

# ── GPU / tmux status ────────────────────────────────────────────────────────
print("── GPU utilization ──")
try:
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=index,utilization.gpu,memory.used,memory.free",
         "--format=csv,noheader"],
        text=True)
    for line in out.splitlines()[:8]:
        print(f"  GPU {line.strip()}")
except Exception:
    print("  (nvidia-smi not available)")

print()
print("── tmux sessions ──")
try:
    out = subprocess.check_output(["tmux", "ls"], text=True, stderr=subprocess.DEVNULL)
    relevant = [l for l in out.splitlines()
                if any(k in l for k in ["warmstart", "vcycle", "tup", "tdown"])]
    for l in relevant:
        print(f"  {l}")
except Exception:
    print("  (tmux not available)")

print()
print("Run `tmux attach -t <session>` to inspect any session live.")
print("Run `tail -f experiments/claude/eigenvalue_1d/corrected_flux_pipeline/tdown_only_log.txt` etc. for live logs.")
