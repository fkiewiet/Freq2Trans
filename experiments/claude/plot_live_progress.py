"""
plot_live_progress.py
---------------------
Merges hardcoded historical data (epochs 1-40, scraped before logging started)
with live log files captured by tmux pipe-pane.

Outputs:
  results_visuals/live_progress_up.png
  results_visuals/live_progress_down.png

Run:
  python experiments/claude/plot_live_progress.py
Or continuously:
  watch -n 600 python experiments/claude/plot_live_progress.py
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("/tmp/fkiewiet/logs")
OUT_DIR = Path(__file__).parent / "results_visuals"
OUT_DIR.mkdir(exist_ok=True)

# ── hardcoded history (scraped from tmux before pipe-pane was started) ─────────

HISTORY = {
    "up_rll2": {
        "epoch":    [1,2,3,4,5,10,20,30,40,50,60,70,80],
        "train_re": [0.9531,0.7542,0.6666,0.6369,0.6199,0.5764,0.5153,0.4653,0.4247,0.4084,0.4454,0.4035,0.3621],
        "val_re":   [0.8310,0.6915,0.6473,0.6277,0.6164,0.5845,0.5493,0.5417,0.5451,0.5492,0.5454,0.5585,0.5650],
        "val_im":   [0.8310,0.6900,0.6455,0.6282,0.6158,0.5839,0.5473,0.5398,0.5433,0.5472,0.5444,0.5569,0.5633],
        "pairs": {
            "16→32":  [72.8,65.9,62.3,60.2,59.3,55.6,53.3,53.2,54.5,55.3,55.0,56.7,57.1],
            "32→64":  [77.8,68.4,64.6,63.0,61.8,58.7,55.5,54.9,55.0,55.4,55.2,56.6,58.0],
            "64→128": [100.0,73.5,67.5,65.3,64.0,61.2,56.0,54.5,53.9,54.0,53.3,54.1,54.2],
        },
    },
    "up_mse": {
        "epoch":    [1,2,3,4,5,10,20,30,40,50,60,70,80],
        "train_re": [0.9488,0.7608,0.6748,0.6426,0.6221,0.5750,0.5206,0.4709,0.4709,0.4084,0.4454,0.4035,0.3621],
        "val_re":   [0.8318,0.7015,0.6561,0.6329,0.6180,0.5819,0.5531,0.5490,0.5490,0.5492,0.5454,0.5585,0.5650],
        "val_im":   [0.8324,0.7007,0.6542,0.6322,0.6181,0.5812,0.5511,0.5467,0.5467,0.5472,0.5444,0.5569,0.5633],
        "pairs": {
            "16→32":  [73.5,66.2,62.4,60.7,59.2,55.3,52.6,53.1,53.1,55.3,55.0,56.7,57.1],
            "32→64":  [77.5,68.7,65.2,63.3,61.7,58.5,55.5,55.3,55.3,55.4,55.2,56.6,58.0],
            "64→128": [99.9,76.0,69.5,66.0,64.7,61.0,58.0,56.4,56.4,54.0,53.3,54.1,54.2],
        },
    },
    "dn_rll2": {
        "epoch":    [1,2,3,4,5,10,20,30,40],
        "train_re": [0.8882,0.6503,0.6177,0.6017,0.5906,0.5534,0.4941,0.4410,0.3930],
        "val_re":   [0.6775,0.6237,0.6099,0.5931,0.5846,0.5593,0.5215,0.5133,0.5250],
        "val_im":   [0.6850,0.6280,0.6130,0.6001,0.5899,0.5630,0.5235,0.5184,0.5282],
        "pairs": {
            "32→16":  [67.5,61.8,60.8,58.3,57.1,53.5,47.0,44.9,43.9],
            "64→32":  [67.5,63.1,61.6,59.9,59.3,57.0,54.4,55.2,57.6],
            "128→64": [68.3,62.2,60.6,59.8,59.0,57.6,55.1,54.7,55.1],
        },
    },
    "dn_mse": {
        "epoch":    [1,2,3,4,5,10,20,30,40],
        "train_re": [0.8929,0.6512,0.6187,0.6026,0.5912,0.5539,0.4949,0.4450,0.4030],
        "val_re":   [0.6853,0.6273,0.6121,0.5941,0.5870,0.5589,0.5214,0.5158,0.5222],
        "val_im":   [0.6878,0.6254,0.6122,0.5982,0.5873,0.5619,0.5237,0.5163,0.5245],
        "pairs": {
            "32→16":  [68.7,63.4,61.7,58.8,57.5,53.6,47.1,43.5,42.2],
            "64→32":  [66.6,62.4,61.0,59.6,59.2,56.7,54.6,55.1,57.8],
            "128→64": [70.5,62.4,60.9,59.9,59.4,57.2,54.8,55.5,57.6],
        },
    },
}

EPOCH_RE = re.compile(
    r"E\s+(\d+)\s+train_re=([\d.]+)\s+val_re=([\d.]+)\s+val_im=([\d.]+)\s+\[(.+?)\]"
)
PAIR_RE = re.compile(r"([\w→]+)=([\d.]+)%")


def parse_log(path: Path) -> dict:
    out = {"epoch": [], "train_re": [], "val_re": [], "val_im": [], "pairs": {}}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        m = EPOCH_RE.search(line)
        if not m:
            continue
        ep = int(m.group(1))
        if ep in out["epoch"]:
            continue
        out["epoch"].append(ep)
        out["train_re"].append(float(m.group(2)))
        out["val_re"].append(float(m.group(3)))
        out["val_im"].append(float(m.group(4)))
        for pk, val in PAIR_RE.findall(m.group(5)):
            out["pairs"].setdefault(pk, []).append(float(val))
    return out


def merge(hist: dict, live: dict) -> dict:
    """Merge historical and live data, deduplicating by epoch."""
    combined = {"epoch": [], "train_re": [], "val_re": [], "val_im": [], "pairs": {}}
    seen = set(hist["epoch"])
    epochs = list(hist["epoch"])
    tr     = list(hist["train_re"])
    vr     = list(hist["val_re"])
    vi     = list(hist["val_im"])
    pairs  = {pk: list(v) for pk, v in hist["pairs"].items()}

    for i, ep in enumerate(live["epoch"]):
        if ep not in seen:
            epochs.append(ep)
            tr.append(live["train_re"][i])
            vr.append(live["val_re"][i])
            vi.append(live["val_im"][i])
            seen.add(ep)
            for pk in live["pairs"]:
                if i < len(live["pairs"][pk]):
                    pairs.setdefault(pk, []).append(live["pairs"][pk][i])

    order = sorted(range(len(epochs)), key=lambda i: epochs[i])
    combined["epoch"]    = [epochs[i] for i in order]
    combined["train_re"] = [tr[i]     for i in order]
    combined["val_re"]   = [vr[i]     for i in order]
    combined["val_im"]   = [vi[i]     for i in order]
    for pk, vals in pairs.items():
        combined["pairs"][pk] = vals
    return combined


PAIR_COLORS = {
    "16→32": "#2E6DA4", "32→64": "#E07B39", "64→128": "#2CA02C",
    "32→16": "#2E6DA4", "64→32": "#E07B39", "128→64": "#2CA02C",
}


def plot_direction(direction: str, key_a: str, key_b: str,
                   log_a: Path, log_b: Path, out_path: Path):
    da = merge(HISTORY[key_a], parse_log(log_a))
    db = merge(HISTORY[key_b], parse_log(log_b))

    last_ep = max(
        da["epoch"][-1] if da["epoch"] else 0,
        db["epoch"][-1] if db["epoch"] else 0,
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.suptitle(
        f"Training progress — {direction.upper()} direction  "
        f"(N=2400/pair, latest epoch={last_ep})   [{ts}]",
        fontsize=11, fontweight="bold",
    )

    pct = lambda lst: [v * 100 for v in lst]

    # Panel 1: loss mode comparison
    ax = axes[0]
    if da["epoch"]:
        ax.plot(da["epoch"], pct(da["val_re"]),   color="#2563EB", lw=2,
                label="rll2  val re")
        ax.plot(da["epoch"], pct(da["val_im"]),   color="#2563EB", lw=1.5,
                ls=":", label="rll2  val im")
        ax.plot(da["epoch"], pct(da["train_re"]), color="#2563EB", lw=1,
                ls="--", alpha=0.4, label="rll2  train re")
    if db["epoch"]:
        ax.plot(db["epoch"], pct(db["val_re"]),   color="#DC2626", lw=2,
                label="mse_rll2  val re")
        ax.plot(db["epoch"], pct(db["val_im"]),   color="#DC2626", lw=1.5,
                ls=":", label="mse_rll2  val im")
        ax.plot(db["epoch"], pct(db["train_re"]), color="#DC2626", lw=1,
                ls="--", alpha=0.4, label="mse_rll2  train re")
    ax.axhline(100, color="#9CA3AF", ls="--", lw=1, label="trivial (100%)")
    ax.axhline(10,  color="#16A34A", ls="--", lw=1.2, label="target (10%)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("RelL2 (%)")
    ax.set_title("Loss mode comparison (val)")
    ax.legend(fontsize=7); ax.grid(alpha=0.25); ax.set_ylim(0, 105)

    # Panel 2: per-pair (rll2)
    ax = axes[1]
    for pk, vals in da["pairs"].items():
        eps = da["epoch"][:len(vals)]
        ax.plot(eps, vals, color=PAIR_COLORS.get(pk, "grey"), lw=2, label=pk)
    ax.axhline(100, color="#9CA3AF", ls="--", lw=1)
    ax.axhline(10,  color="#16A34A", ls="--", lw=1.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val RelL2 re (%)")
    ax.set_title("Per frequency pair (rll2)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25); ax.set_ylim(0, 105)

    # Panel 3: re vs im parity (rll2)
    ax = axes[2]
    if da["epoch"]:
        ax.plot(da["epoch"], pct(da["val_re"]), color="#2563EB", lw=2,
                label="val Re(u_high)")
        ax.plot(da["epoch"], pct(da["val_im"]), color="#7C3AED", lw=2,
                label="val Im(u_high)")
        ax.fill_between(da["epoch"], pct(da["val_re"]), pct(da["val_im"]),
                        alpha=0.15, color="#7C3AED", label="Re/Im gap")
    ax.axhline(10, color="#16A34A", ls="--", lw=1.2, label="target (10%)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val RelL2 (%)")
    ax.set_title("Real vs Imaginary channel (rll2)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25); ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}  (epoch {last_ep})")


if __name__ == "__main__":
    print(f"Plotting — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    plot_direction("up",   "up_rll2", "up_mse",
                   LOG_DIR/"up_rll2.log", LOG_DIR/"up_mse.log",
                   OUT_DIR/"live_progress_up.png")
    plot_direction("down", "dn_rll2", "dn_mse",
                   LOG_DIR/"dn_rll2.log", LOG_DIR/"dn_mse.log",
                   OUT_DIR/"live_progress_down.png")
