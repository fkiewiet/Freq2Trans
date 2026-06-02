from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODEX_DIR = ROOT / "experiments" / "codex"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_parallel(commands: list[tuple[str, list[str], dict | None]], logs_dir: Path) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for name, cmd, env_update in commands:
        env = os.environ.copy()
        if env_update:
            env.update(env_update)
        log_path = logs_dir / f"{name}.log"
        handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append((name, proc, handle, log_path))

    failures = []
    while procs:
        pending = []
        for name, proc, handle, log_path in procs:
            rc = proc.poll()
            if rc is None:
                pending.append((name, proc, handle, log_path))
                continue
            handle.close()
            if rc != 0:
                failures.append((name, rc, log_path))
        if pending:
            time.sleep(2.0)
        procs = pending

    if failures:
        msg = "\n".join([f"{name} failed with code {rc}; see {path}" for name, rc, path in failures])
        raise RuntimeError(msg)


def parse_path_map(values: list[str] | None, omegas: list[int]) -> dict[int, Path]:
    if not values:
        return {}
    out = {}
    for value in values:
        omega_str, path_str = value.split(":", 1)
        out[int(omega_str)] = Path(path_str)
    missing = [omega for omega in omegas if omega not in out]
    if missing:
        raise ValueError(f"missing dataset-map entries for omegas: {missing}")
    return out


def stage_generate(
    run_root: Path,
    omegas: list[int],
    dataset_dirs: dict[int, Path],
    n_problems: int,
    gmres_iters: int,
    save_stages: str,
    preview_count: int,
    skip_generate: bool,
) -> tuple[dict, dict[int, Path]]:
    if skip_generate:
        gates = {}
        for omega in omegas:
            dataset_dir = dataset_dirs[omega]
            manifest = dataset_dir / "manifest.jsonl"
            count = sum(1 for _ in manifest.open("r", encoding="utf-8"))
            gates[str(omega)] = {
                "pass": manifest.exists() and count > 0,
                "reason": f"reused existing dataset with {count} problems",
                "dataset_dir": str(dataset_dir),
            }
        return gates, dataset_dirs

    logs_dir = run_root / "logs" / "generate"
    commands = []
    generated_dirs = {}
    for omega in omegas:
        dataset_dir = run_root / f"omega{omega}_dataset"
        generated_dirs[omega] = dataset_dir
        commands.append(
            (
                f"omega{omega}",
                [
                    sys.executable,
                    str(CODEX_DIR / "generate_residual_dataset.py"),
                    "--outdir",
                    str(dataset_dir),
                    "--omegas",
                    str(omega),
                    "--n-problems",
                    str(n_problems),
                    "--gmres-iters",
                    str(gmres_iters),
                    "--save-stages",
                    save_stages,
                    "--preview-count",
                    str(preview_count),
                ],
                None,
            )
        )
    run_parallel(commands, logs_dir=logs_dir)

    gates = {}
    for omega, dataset_dir in generated_dirs.items():
        manifest = dataset_dir / "manifest.jsonl"
        count = sum(1 for _ in manifest.open("r", encoding="utf-8"))
        ok = count == n_problems
        gates[str(omega)] = {
            "pass": ok,
            "reason": f"{count}/{n_problems} problems generated",
            "dataset_dir": str(dataset_dir),
        }
    return gates, generated_dirs


def stage_inspect(run_root: Path, omegas: list[int], dataset_dirs: dict[int, Path]) -> dict:
    logs_dir = run_root / "logs" / "inspect"
    commands = []
    outdirs = {}
    for omega in omegas:
        dataset_dir = dataset_dirs[omega]
        outdir = dataset_dir / "inspection"
        outdirs[omega] = outdir
        commands.append(
            (
                f"omega{omega}",
                [
                    sys.executable,
                    str(CODEX_DIR / "inspect_residuals.py"),
                    "--dataset-dir",
                    str(dataset_dir),
                    "--outdir",
                    str(outdir),
                ],
                None,
            )
        )
    run_parallel(commands, logs_dir=logs_dir)

    gates = {}
    for omega, outdir in outdirs.items():
        summary_path = outdir / "summary.json"
        summary = read_json(summary_path)
        ok = summary["n_problems"] > 0 and len(summary["stages"]) > 0
        gates[str(omega)] = {
            "pass": ok,
            "reason": f"inspection summary present with {summary['n_problems']} problems",
            "inspection_dir": str(outdir),
        }
    return gates


def stage_train(
    run_root: Path,
    omegas: list[int],
    dataset_dirs: dict[int, Path],
    gpu_map: dict[int, str],
    epochs: int,
    batch_size: int,
    max_stage: int,
) -> dict:
    logs_dir = run_root / "logs" / "train"
    commands = []
    run_dirs = {}
    for omega in omegas:
        run_dir = run_root / f"omega{omega}_train"
        run_dirs[omega] = run_dir
        env_update = {"CUDA_VISIBLE_DEVICES": gpu_map[omega]}
        commands.append(
            (
                f"omega{omega}",
                [
                    sys.executable,
                    str(CODEX_DIR / "train_residual_to_correction.py"),
                    "--dataset-dir",
                    str(dataset_dirs[omega]),
                    "--run-dir",
                    str(run_dir),
                    "--epochs",
                    str(epochs),
                    "--batch-size",
                    str(batch_size),
                    "--device",
                    "cuda",
                    "--max-stage",
                    str(max_stage),
                ],
                env_update,
            )
        )
    run_parallel(commands, logs_dir=logs_dir)

    gates = {}
    for omega, run_dir in run_dirs.items():
        metrics_path = run_dir / "metrics.jsonl"
        rows = [json.loads(line) for line in metrics_path.open("r", encoding="utf-8")]
        first = rows[0]["val_loss"]
        best = min(row["val_loss"] for row in rows)
        ok = best < first and (run_dir / "checkpoints" / "best.pt").exists()
        gates[str(omega)] = {
            "pass": ok,
            "reason": f"first val_loss={first:.4e}, best val_loss={best:.4e}",
            "train_dir": str(run_dir),
        }
    return gates


def choose_best_result(results: list[dict], curve_key: str, gate_step: int) -> dict:
    def score(item: dict) -> tuple[float, float]:
        curve = item["summary"][curve_key]
        step_idx = min(gate_step, len(curve) - 1)
        return (curve[step_idx], curve[-1])

    return min(results, key=score)


def build_mode_gate(best: dict, curve_key: str, gate_step: int) -> dict:
    curve = best["summary"][curve_key]
    step_idx = min(gate_step, len(curve) - 1)
    step_val = curve[step_idx]
    final_val = curve[-1]
    improves_fraction_key = (
        "direct_improves_fraction" if curve_key == "direct_mean_curve" else "fgmres_improves_fraction"
    )
    improves_fraction = float(best["summary"].get(improves_fraction_key, 0.0))
    ok = step_val < 1.0
    return {
        "pass": ok,
        "reason": (
            f"best damping={best['damping']}, "
            f"step{step_idx}={step_val:.4e}, final={final_val:.4e}, "
            f"improves_fraction={improves_fraction:.2f}"
        ),
        "best_damping": best["damping"],
        "gate_step": step_idx,
        "gate_value": step_val,
        "final_value": final_val,
        "improves_fraction": improves_fraction,
        "eval_dir": best["outdir"],
    }


def stage_eval(
    run_root: Path,
    omegas: list[int],
    gpu_map: dict[int, str],
    dampings: list[float],
    steps: int,
    gate_step: int,
) -> dict:
    logs_dir = run_root / "logs" / "eval"
    all_results: dict[int, list[dict]] = {omega: [] for omega in omegas}
    for damping in dampings:
        commands = []
        for omega in omegas:
            outdir = run_root / f"omega{omega}_eval_damp_{str(damping).replace('.', 'p')}"
            env_update = {"CUDA_VISIBLE_DEVICES": gpu_map[omega]}
            commands.append(
                (
                    f"omega{omega}_d{str(damping).replace('.', 'p')}",
                    [
                        sys.executable,
                        str(CODEX_DIR / "eval_iterative.py"),
                        "--checkpoint",
                        str(run_root / f"omega{omega}_train" / "checkpoints" / "best.pt"),
                        "--outdir",
                        str(outdir),
                        "--omega",
                        str(omega),
                        "--n-problems",
                        "6",
                        "--steps",
                        str(steps),
                        "--gate-step",
                        str(gate_step),
                        "--damping",
                        str(damping),
                        "--device",
                        "cuda",
                    ],
                    env_update,
                )
            )
        run_parallel(commands, logs_dir=logs_dir)
        for omega in omegas:
            outdir = run_root / f"omega{omega}_eval_damp_{str(damping).replace('.', 'p')}"
            summary = read_json(outdir / "summary.json")
            all_results[omega].append({"damping": damping, "summary": summary, "outdir": str(outdir)})

    gates = {}
    for omega in omegas:
        best_direct = choose_best_result(all_results[omega], curve_key="direct_mean_curve", gate_step=gate_step)
        best_fgmres = choose_best_result(all_results[omega], curve_key="fgmres_mean_curve", gate_step=gate_step)
        direct_gate = build_mode_gate(best_direct, curve_key="direct_mean_curve", gate_step=gate_step)
        fgmres_gate = build_mode_gate(best_fgmres, curve_key="fgmres_mean_curve", gate_step=gate_step)
        gates[str(omega)] = {
            "pass": direct_gate["pass"],
            "reason": direct_gate["reason"],
            "direct": direct_gate,
            "fgmres": fgmres_gate,
        }
    return gates


def parse_gpu_map(values: list[str], omegas: list[int]) -> dict[int, str]:
    out = {}
    for value in values:
        omega_str, gpu_str = value.split(":")
        out[int(omega_str)] = gpu_str
    missing = [omega for omega in omegas if omega not in out]
    if missing:
        raise ValueError(f"missing gpu-map entries for omegas: {missing}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the codex gated unattended pipeline.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--omegas", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--gpu-map", type=str, nargs="+", required=True, help="Entries like 16:2 32:6")
    parser.add_argument("--dataset-map", type=str, nargs="+", default=None, help="Optional entries like 64:/path/to/dataset")
    parser.add_argument("--skip-generate", action="store_true", help="Reuse existing datasets from --dataset-map")
    parser.add_argument("--n-problems", type=int, default=20)
    parser.add_argument("--gmres-iters", type=int, default=4)
    parser.add_argument("--save-stages", type=str, default="0,1,2,4")
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-stage", type=int, default=0)
    parser.add_argument("--eval-steps", type=int, default=6)
    parser.add_argument("--eval-gate-step", type=int, default=1)
    parser.add_argument("--dampings", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    args = parser.parse_args()

    run_root = args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    gpu_map = parse_gpu_map(args.gpu_map, args.omegas)
    dataset_map = parse_path_map(args.dataset_map, args.omegas) if args.skip_generate else {}

    summary = {
        "config": {
            "omegas": args.omegas,
            "gpu_map": gpu_map,
            "skip_generate": args.skip_generate,
            "dataset_map": {str(k): str(v) for k, v in dataset_map.items()},
            "n_problems": args.n_problems,
            "gmres_iters": args.gmres_iters,
            "save_stages": args.save_stages,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_stage": args.max_stage,
            "eval_steps": args.eval_steps,
            "eval_gate_step": args.eval_gate_step,
            "dampings": args.dampings,
        },
        "gates": {},
    }
    write_json(run_root / "pipeline_config.json", summary["config"])

    summary["gates"]["generate"], dataset_dirs = stage_generate(
        run_root=run_root,
        omegas=args.omegas,
        dataset_dirs=dataset_map,
        n_problems=args.n_problems,
        gmres_iters=args.gmres_iters,
        save_stages=args.save_stages,
        preview_count=args.preview_count,
        skip_generate=args.skip_generate,
    )
    write_json(run_root / "gate_generate.json", summary["gates"]["generate"])
    if not all(v["pass"] for v in summary["gates"]["generate"].values()):
        write_json(run_root / "pipeline_summary.json", summary)
        return

    summary["gates"]["inspect"] = stage_inspect(run_root=run_root, omegas=args.omegas, dataset_dirs=dataset_dirs)
    write_json(run_root / "gate_inspect.json", summary["gates"]["inspect"])
    if not all(v["pass"] for v in summary["gates"]["inspect"].values()):
        write_json(run_root / "pipeline_summary.json", summary)
        return

    summary["gates"]["train"] = stage_train(
        run_root=run_root,
        omegas=args.omegas,
        dataset_dirs=dataset_dirs,
        gpu_map=gpu_map,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_stage=args.max_stage,
    )
    write_json(run_root / "gate_train.json", summary["gates"]["train"])
    if not all(v["pass"] for v in summary["gates"]["train"].values()):
        write_json(run_root / "pipeline_summary.json", summary)
        return

    summary["gates"]["eval"] = stage_eval(
        run_root=run_root,
        omegas=args.omegas,
        gpu_map=gpu_map,
        dampings=args.dampings,
        steps=args.eval_steps,
        gate_step=args.eval_gate_step,
    )
    write_json(run_root / "gate_eval.json", summary["gates"]["eval"])

    summary["overall_pass"] = all(
        gate["pass"]
        for stage in summary["gates"].values()
        for gate in stage.values()
    )
    write_json(run_root / "pipeline_summary.json", summary)


if __name__ == "__main__":
    main()
