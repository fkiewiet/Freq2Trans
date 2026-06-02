from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

from common import (
    TransferWarmStart,
    _gaussian_source,
    auto_ckpt_path,
    build_csl_preconditioner,
    generate_test_problems,
    interior_rel_error,
    relative_residual,
    run_fgmres_to_tol,
    N,
    NPML,
    INTERIOR,
)


def _solve_reference(a, b):
    return spla.spsolve(a, b)


def _free_space_low_solve(src_field: np.ndarray, warm: TransferWarmStart):
    from common import _solve_helmholtz_green  # local import to avoid extra surface

    t0 = time.time()
    u_low = _solve_helmholtz_green(warm.omega_low, src_field).astype(np.complex128)
    return u_low.flatten(), {"predict_time_s": time.time() - t0}


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _resolve_dataset_dir(raw_path: str | Path) -> Path:
    raw = Path(raw_path)
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        root = Path(__file__).resolve().parents[2]
        candidates.append(root / raw)

    name = raw.name
    if name:
        candidates.extend(
            [
                Path("/orcd/pool/006/fkiewiet/freq2transfer/datasets_N9600") / name,
                Path("/scratch/fkiewiet/datasets_N9600") / name,
                Path("/tmp/fkiewiet/datasets_N9600") / name,
                Path(__file__).resolve().parents[2] / "experiments" / "claude" / "datasets_persistent" / name,
                Path(__file__).resolve().parents[2] / "experiments" / "claude" / "datasets" / name,
            ]
        )

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"  - {c}" for c in unique_candidates)
    raise FileNotFoundError(
        f"Dataset directory not found for requested path '{raw_path}'.\nSearched:\n{searched}"
    )


def _load_dataset_eval_context(
    ckpt_path: Path,
    dataset_dir_override: str | None,
    split_name: str,
    max_samples: int | None,
    sample_offset: int,
) -> dict:
    run_dir = ckpt_path.parent
    split_npz = run_dir / "split_indices.npz"
    split_summary_path = run_dir / "split_summary.json"
    summary_path = run_dir / "summary.json"

    if not split_npz.exists():
        raise FileNotFoundError(
            f"Dataset-backed eval requested, but split file is missing: {split_npz}"
        )

    summary = _load_json(summary_path) if summary_path.exists() else {}
    split_summary = _load_json(split_summary_path) if split_summary_path.exists() else {}

    dataset_raw = dataset_dir_override or summary.get("dataset") or split_summary.get("dataset")
    if dataset_raw is None:
        raise FileNotFoundError(
            "Could not determine dataset directory from checkpoint run artifacts. "
            "Pass --dataset_dir explicitly."
        )
    ds_dir = _resolve_dataset_dir(dataset_raw)
    ds_meta = _load_json(ds_dir / "metadata.json")

    pair_idx = int(summary.get("pair_idx", split_summary.get("pair_idx")))
    pair = summary.get("pair", split_summary.get("pair"))

    split_indices_npz = np.load(split_npz)
    if split_name not in split_indices_npz:
        raise KeyError(f"Split '{split_name}' not found in {split_npz}")

    raw_indices = np.asarray(split_indices_npz[split_name], dtype=np.int64)
    raw_indices = np.sort(raw_indices)
    if sample_offset:
        raw_indices = raw_indices[sample_offset:]
    if max_samples is not None:
        raw_indices = raw_indices[:max_samples]

    arrays = {
        "u_low_re": np.load(ds_dir / "u_low_re.npy", mmap_mode="r"),
        "u_low_im": np.load(ds_dir / "u_low_im.npy", mmap_mode="r"),
        "u_high_re": np.load(ds_dir / "u_high_re.npy", mmap_mode="r"),
        "u_high_im": np.load(ds_dir / "u_high_im.npy", mmap_mode="r"),
        "rms": np.load(ds_dir / "rms.npy", mmap_mode="r"),
    }

    return {
        "run_dir": run_dir,
        "summary": summary,
        "split_summary": split_summary,
        "dataset_dir": ds_dir,
        "dataset_meta": ds_meta,
        "pair_idx": pair_idx,
        "pair": pair,
        "split_name": split_name,
        "raw_indices": raw_indices,
        "arrays": arrays,
    }


def _regenerate_source_field(meta: dict, pair_idx: int, raw_idx: int) -> tuple[np.ndarray, int]:
    n_per_pair = int(meta["n_per_pair"])
    seed = int(meta["seed"])
    sample_idx = int(raw_idx) - pair_idx * n_per_pair
    if sample_idx < 0 or sample_idx >= n_per_pair:
        raise ValueError(
            f"raw_idx={raw_idx} is inconsistent with pair_idx={pair_idx} and n_per_pair={n_per_pair}"
        )

    rng = np.random.default_rng(seed + pair_idx * n_per_pair + sample_idx)
    n_src = int(rng.integers(3, 7))
    px = rng.integers(NPML, NPML + INTERIOR, size=n_src)
    py = rng.integers(NPML, NPML + INTERIOR, size=n_src)
    amps = rng.uniform(1.0, 2.0, size=n_src)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_src)

    src = np.zeros((N, N), dtype=np.complex128)
    for s in range(n_src):
        src += _gaussian_source(N, px[s], py[s], amps[s] * np.exp(1j * phases[s]))

    return src, n_src


def _build_dataset_split_problems(ctx: dict) -> list[dict]:
    problems = []
    pair_idx = int(ctx["pair_idx"])
    n_per_pair = int(ctx["dataset_meta"]["n_per_pair"])
    arrays = ctx["arrays"]

    for idx, raw_idx in enumerate(ctx["raw_indices"]):
        raw = int(raw_idx)
        sample_idx = raw - pair_idx * n_per_pair
        src_field, n_src = _regenerate_source_field(ctx["dataset_meta"], pair_idx, raw)
        u_low_norm = (
            arrays["u_low_re"][raw].astype(np.float32)
            + 1j * arrays["u_low_im"][raw].astype(np.float32)
        ).astype(np.complex64)
        u_high_norm = (
            arrays["u_high_re"][raw].astype(np.float32)
            + 1j * arrays["u_high_im"][raw].astype(np.float32)
        ).astype(np.complex64)
        problems.append(
            {
                "idx": int(idx),
                "raw_idx": raw,
                "sample_idx": int(sample_idx),
                "n_src": int(n_src),
                "b": src_field.flatten(),
                "src_field": src_field,
                "u_low_norm": u_low_norm,
                "u_high_norm": u_high_norm,
                "rms_low": float(arrays["rms"][raw]),
            }
        )
    return problems


def _summarize(problems: list[dict], arms: list[str]) -> dict:
    out: dict[str, dict] = {}

    for arm in arms:
        field_key = f"{arm}_field_err_k0"
        resid_key = f"{arm}_rel_res_k0"
        iter_key = f"{arm}_fgmres_iters"
        time_key = f"{arm}_fgmres_time_s"
        total_key = f"{arm}_total_time_s"
        conv_key = f"{arm}_converged"

        vals_field = np.array([p[field_key] for p in problems], dtype=float)
        vals_resid = np.array([p[resid_key] for p in problems], dtype=float)
        vals_iter = np.array([p[iter_key] for p in problems], dtype=float)
        vals_time = np.array([p[time_key] for p in problems], dtype=float)
        vals_total = np.array([p[total_key] for p in problems], dtype=float)
        vals_conv = np.array([1.0 if p[conv_key] else 0.0 for p in problems], dtype=float)

        out[arm] = {
            "mean_field_err_k0": float(vals_field.mean()),
            "median_field_err_k0": float(np.median(vals_field)),
            "mean_rel_res_k0": float(vals_resid.mean()),
            "mean_fgmres_iters": float(vals_iter.mean()),
            "median_fgmres_iters": float(np.median(vals_iter)),
            "mean_fgmres_time_s": float(vals_time.mean()),
            "mean_total_time_s": float(vals_total.mean()),
            "convergence_rate": float(vals_conv.mean()),
        }

    warm_vs_zero_iters = np.array(
        [p["warm_fgmres_iters"] - p["zero_fgmres_iters"] for p in problems], dtype=float
    )
    warm_vs_zero_resid = np.array(
        [p["warm_rel_res_k0"] / max(p["zero_rel_res_k0"], 1e-12) for p in problems], dtype=float
    )

    paired = {
        "warm_better_than_zero_on_field_err_frac": float(
            np.mean([p["warm_field_err_k0"] < p["zero_field_err_k0"] for p in problems])
        ),
        "warm_better_than_zero_on_rel_res_frac": float(
            np.mean([p["warm_rel_res_k0"] < p["zero_rel_res_k0"] for p in problems])
        ),
        "warm_hits_rel_res_lt_0p1_frac": float(
            np.mean([p["warm_rel_res_k0"] < 1e-1 for p in problems])
        ),
        "warm_hits_rel_res_lt_0p01_frac": float(
            np.mean([p["warm_rel_res_k0"] < 1e-2 for p in problems])
        ),
        "warm_rel_res_ratio_to_zero_mean": float(warm_vs_zero_resid.mean()),
        "warm_rel_res_ratio_to_zero_median": float(np.median(warm_vs_zero_resid)),
        "warm_better_than_zero_on_iters_frac": float(
            np.mean([p["warm_fgmres_iters"] < p["zero_fgmres_iters"] for p in problems])
        ),
        "warm_minus_zero_iters_mean": float(warm_vs_zero_iters.mean()),
    }

    if "copy_low" in arms:
        warm_vs_copy_iters = np.array(
            [p["warm_fgmres_iters"] - p["copy_low_fgmres_iters"] for p in problems], dtype=float
        )
        paired.update(
            {
                "warm_better_than_copy_on_field_err_frac": float(
                    np.mean([p["warm_field_err_k0"] < p["copy_low_field_err_k0"] for p in problems])
                ),
                "warm_better_than_copy_on_iters_frac": float(
                    np.mean([p["warm_fgmres_iters"] < p["copy_low_fgmres_iters"] for p in problems])
                ),
                "warm_minus_copy_iters_mean": float(warm_vs_copy_iters.mean()),
            }
        )

    out["paired"] = paired
    return out


def main():
    p = argparse.ArgumentParser(description="Rigorous warm-start evaluation")
    p.add_argument("--omega", type=float, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument(
        "--model_family",
        choices=["cnn", "precond_v3_unet"],
        default="precond_v3_unet",
        help="Which checkpoint family to use for the learned warm-start model.",
    )
    p.add_argument(
        "--eval_mode",
        choices=["dataset_split", "random_rhs"],
        default="dataset_split",
        help="Use the fixed ORCD split from the training run, or generate fresh random RHS samples.",
    )
    p.add_argument("--dataset_dir", type=str, default=None, help="Optional override for the dataset directory.")
    p.add_argument("--split_name", choices=["train", "val", "test"], default="test")
    p.add_argument("--n_problems", type=int, default=8, help="Used only for random_rhs mode.")
    p.add_argument("--max_samples", type=int, default=None, help="Cap the number of split samples to evaluate.")
    p.add_argument("--sample_offset", type=int, default=0, help="Skip this many split samples before evaluation.")
    p.add_argument("--seed", type=int, default=77777, help="Used only for random_rhs mode.")
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--restart", type=int, default=20)
    p.add_argument("--maxiter", type=int, default=50)
    p.add_argument("--preconditioner", choices=["none", "csl"], default="csl")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument(
        "--include_copy_low",
        action="store_true",
        help="Include the raw half-frequency field as an auxiliary baseline.",
    )
    args = p.parse_args()

    omega = float(args.omega)
    ckpt_path = (
        Path(args.ckpt)
        if args.ckpt
        else auto_ckpt_path(omega, model_family=args.model_family)
    )
    warm = TransferWarmStart(
        ckpt_path,
        omega_target=omega,
        device=args.device,
        model_family=args.model_family,
    )

    dataset_ctx = None
    if args.eval_mode == "dataset_split":
        dataset_ctx = _load_dataset_eval_context(
            ckpt_path=ckpt_path,
            dataset_dir_override=args.dataset_dir,
            split_name=args.split_name,
            max_samples=args.max_samples,
            sample_offset=args.sample_offset,
        )
        problems_local = _build_dataset_split_problems(dataset_ctx)
    else:
        problems_local, _ = generate_test_problems(omega, args.n_problems, args.seed)

    _, a = generate_test_problems(omega, 1, 0)
    m = (
        build_csl_preconditioner(a, omega, beta=args.beta)
        if args.preconditioner == "csl"
        else None
    )
    arms = ["zero", "warm"]
    if args.include_copy_low:
        arms.insert(1, "copy_low")

    field_error_reference = "dataset_u_high" if args.eval_mode == "dataset_split" else "pml_x_true"
    problem_rows: list[dict] = []

    for problem in problems_local:
        b = problem["b"]
        src_field = problem["src_field"]
        x_zero = np.zeros_like(b)

        if args.eval_mode == "dataset_split":
            x_target = (problem["u_high_norm"].astype(np.complex128) * problem["rms_low"]).flatten()
            x_warm, warm_meta = warm.predict_from_dataset(problem["u_low_norm"], problem["rms_low"])
            x_copy_low = (problem["u_low_norm"].astype(np.complex128) * problem["rms_low"]).flatten()
            copy_meta = {"predict_time_s": 0.0}
        else:
            x_target = _solve_reference(a, b)
            x_warm, warm_meta = warm.predict(b)
            x_copy_low, copy_meta = _free_space_low_solve(src_field, warm)

        row = {
            "idx": int(problem["idx"]),
            "n_src": int(problem["n_src"]),
        }
        if args.eval_mode == "dataset_split":
            row.update(
                {
                    "raw_idx": int(problem["raw_idx"]),
                    "sample_idx": int(problem["sample_idx"]),
                }
            )

        row["zero_field_err_k0"] = interior_rel_error(x_zero, x_target)
        row["warm_field_err_k0"] = interior_rel_error(x_warm, x_target)

        row["zero_rel_res_k0"] = relative_residual(a, b, x_zero)
        row["warm_rel_res_k0"] = relative_residual(a, b, x_warm)
        row["warm_rel_res_ratio_to_zero"] = row["warm_rel_res_k0"] / max(row["zero_rel_res_k0"], 1e-12)

        zero = run_fgmres_to_tol(a, b, x_zero, args.tol, args.restart, args.maxiter, m)
        warm_run = run_fgmres_to_tol(a, b, x_warm, args.tol, args.restart, args.maxiter, m)

        row["zero_fgmres_iters"] = zero["iters"]
        row["warm_fgmres_iters"] = warm_run["iters"]
        row["zero_fgmres_time_s"] = zero["time_s"]
        row["warm_fgmres_time_s"] = warm_run["time_s"]
        row["zero_total_time_s"] = zero["time_s"]
        row["warm_total_time_s"] = warm_meta["predict_time_s"] + warm_run["time_s"]
        row["zero_converged"] = bool(zero["converged"])
        row["warm_converged"] = bool(warm_run["converged"])
        row["zero_curve"] = zero["residual_curve"]
        row["warm_curve"] = warm_run["residual_curve"]
        row["zero_predict_time_s"] = 0.0
        row["warm_predict_time_s"] = float(warm_meta["predict_time_s"])

        if args.include_copy_low:
            row["copy_low_field_err_k0"] = interior_rel_error(x_copy_low, x_target)
            row["copy_low_rel_res_k0"] = relative_residual(a, b, x_copy_low)
            copy_low = run_fgmres_to_tol(a, b, x_copy_low, args.tol, args.restart, args.maxiter, m)
            row["copy_low_fgmres_iters"] = copy_low["iters"]
            row["copy_low_fgmres_time_s"] = copy_low["time_s"]
            row["copy_low_total_time_s"] = copy_meta["predict_time_s"] + copy_low["time_s"]
            row["copy_low_converged"] = bool(copy_low["converged"])
            row["copy_low_curve"] = copy_low["residual_curve"]
            row["copy_low_predict_time_s"] = float(copy_meta["predict_time_s"])

        problem_rows.append(row)

    summary = _summarize(problem_rows, arms)

    if args.eval_mode == "dataset_split":
        suffix = f"omega{int(round(omega))}_{args.preconditioner}_{args.model_family}_{args.split_name}"
        if args.max_samples is not None:
            suffix += f"_N{args.max_samples}"
    else:
        suffix = f"omega{int(round(omega))}_{args.preconditioner}_{args.model_family}_seed{args.seed}"
    run_dir = Path("experiments/warmstart_gmres/runs") / suffix
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "omega": omega,
        "ckpt": str(ckpt_path),
        "device": args.device,
        "solver": "fgmres",
        "eval_mode": args.eval_mode,
        "field_error_reference": field_error_reference,
        "arms": arms,
        "fgmres_tol": args.tol,
        "fgmres_restart": args.restart,
        "fgmres_maxiter": args.maxiter,
        "preconditioner": args.preconditioner,
        "csl_beta": args.beta if args.preconditioner == "csl" else None,
        "benchmark_baseline": "zero",
        "model_family": args.model_family,
        "warm_model_metadata": warm.metadata,
        "problems": problem_rows,
        "summary": summary,
    }

    if args.eval_mode == "dataset_split":
        payload.update(
            {
                "n_problems": len(problem_rows),
                "dataset_dir": str(dataset_ctx["dataset_dir"]),
                "dataset_pair": dataset_ctx["pair"],
                "dataset_pair_idx": int(dataset_ctx["pair_idx"]),
                "split_name": args.split_name,
                "split_file": str(dataset_ctx["run_dir"] / "split_indices.npz"),
                "split_sample_offset": int(args.sample_offset),
                "split_max_samples": None if args.max_samples is None else int(args.max_samples),
                "run_dir": str(dataset_ctx["run_dir"]),
                "training_summary_path": str(dataset_ctx["run_dir"] / "summary.json"),
            }
        )
    else:
        payload.update(
            {
                "n_problems": args.n_problems,
                "seed": args.seed,
            }
        )

    out_json = run_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {out_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
