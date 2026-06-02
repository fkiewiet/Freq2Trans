from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from codex_common import (
    DEFAULT_STAGES,
    GRID_N,
    NPML,
    SIGMA0_MAP,
    build_solver_bundle,
    choose_valid_stages,
    ensure_dir,
    gmres_trajectory,
    interior_view,
    parse_stages,
    random_multi_source_rhs,
    rel_l2,
    set_seed,
    solve_field,
    write_json,
)


def _plot_problem_summary(
    save_path: Path,
    omega: int,
    rhs: np.ndarray,
    u_true: np.ndarray,
    residuals: list[np.ndarray],
    corrections: list[np.ndarray],
    stages: list[int],
    rel_residuals: list[float],
) -> None:
    fig, axes = plt.subplots(6, len(stages), figsize=(4 * len(stages), 16))
    if len(stages) == 1:
        axes = np.array(axes).reshape(6, 1)

    for col, stage in enumerate(stages):
        r = residuals[stage]
        z = corrections[stage]
        axes[0, col].imshow(rhs.real, cmap="RdBu_r")
        axes[0, col].set_title(f"RHS Re(f)  omega={omega}")
        axes[1, col].imshow(rhs.imag, cmap="RdBu_r")
        axes[1, col].set_title("RHS Im(f)")
        axes[2, col].imshow(r.real, cmap="RdBu_r")
        axes[2, col].set_title(f"Stage {stage}  Re(r)  rr={rel_residuals[stage]:.2e}")
        axes[3, col].imshow(r.imag, cmap="RdBu_r")
        axes[3, col].set_title(f"Stage {stage}  Im(r)")
        axes[4, col].imshow(z.real, cmap="RdBu_r")
        axes[4, col].set_title(f"Stage {stage}  Re(z)")
        axes[5, col].imshow(z.imag, cmap="RdBu_r")
        axes[5, col].set_title(f"Stage {stage}  Im(z)")

    for row in range(6):
        for col in range(len(stages)):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, axes2 = plt.subplots(4, len(stages), figsize=(4 * len(stages), 11))
    if len(stages) == 1:
        axes2 = np.array(axes2).reshape(4, 1)
    for col, stage in enumerate(stages):
        axes2[0, col].imshow(interior_view(residuals[stage]).real, cmap="RdBu_r")
        axes2[0, col].set_title(f"Interior Re(r)  k={stage}")
        axes2[1, col].imshow(interior_view(residuals[stage]).imag, cmap="RdBu_r")
        axes2[1, col].set_title(f"Interior Im(r)  k={stage}")
        axes2[2, col].imshow(interior_view(corrections[stage]).real, cmap="RdBu_r")
        axes2[2, col].set_title(f"Interior Re(z)  k={stage}")
        axes2[3, col].imshow(interior_view(corrections[stage]).imag, cmap="RdBu_r")
        axes2[3, col].set_title(f"Interior Im(z)  k={stage}")
    for row in range(4):
        for col in range(len(stages)):
            axes2[row, col].set_xticks([])
            axes2[row, col].set_yticks([])
    save_path2 = save_path.with_name(save_path.stem + "_interior.png")
    plt.tight_layout()
    fig2.savefig(save_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)


def generate_problem(
    bundle,
    rng: np.random.Generator,
    gmres_iters: int,
    save_stages: list[int],
    sigma: float,
    min_sources: int,
    max_sources: int,
    greens_cache: dict | None = None,
) -> tuple[dict, dict]:
    if greens_cache is None:
        rhs_field, meta = random_multi_source_rhs(
            rng=rng,
            n=GRID_N,
            n_pml=NPML,
            min_sources=min_sources,
            max_sources=max_sources,
            sigma=sigma,
        )
        u_true = solve_field(bundle, rhs_field)
    else:
        positions = greens_cache["positions"]
        fields = greens_cache["fields"]
        n_sources = int(rng.integers(min_sources, max_sources + 1))
        idxs = rng.integers(0, len(positions), size=n_sources)
        amps = rng.uniform(1.0, 2.0, size=n_sources)
        phases = rng.uniform(0.0, 2 * np.pi, size=n_sources)

        rhs_field = np.zeros((GRID_N, GRID_N), dtype=np.complex128)
        u_true = np.zeros((GRID_N, GRID_N), dtype=np.complex128)
        for i, src_idx in enumerate(idxs):
            x, y = positions[int(src_idx)]
            rhs_field[y, x] += amps[i] * np.exp(1j * phases[i])
            u_true += (amps[i] * np.exp(1j * phases[i])) * (
                fields[int(src_idx), 0].astype(np.float32) + 1j * fields[int(src_idx), 1].astype(np.float32)
            )

        meta = {
            "n_sources": n_sources,
            "px": np.array([positions[int(i)][0] for i in idxs], dtype=np.int32),
            "py": np.array([positions[int(i)][1] for i in idxs], dtype=np.int32),
            "amps": amps.astype(np.float32),
            "phases": phases.astype(np.float32),
            "sigma": float(sigma),
            "greens_cache": True,
        }
    traj = gmres_trajectory(
        A=bundle.A,
        b=rhs_field.ravel().astype(np.complex128),
        x_true=u_true.ravel().astype(np.complex128),
        max_iter=gmres_iters,
    )

    valid_stages = choose_valid_stages(save_stages, len(traj["residuals"]))
    residuals = np.stack(
        [traj["residuals"][idx].reshape(GRID_N, GRID_N).astype(np.complex64) for idx in valid_stages],
        axis=0,
    )
    corrections = np.stack(
        [traj["corrections"][idx].reshape(GRID_N, GRID_N).astype(np.complex64) for idx in valid_stages],
        axis=0,
    )
    rel_residuals = np.array([traj["rel_residuals"][idx] for idx in valid_stages], dtype=np.float32)
    interior_rel = np.array(
        [
            rel_l2(
                np.zeros_like(interior_view(corrections[i])),
                interior_view(corrections[i]),
            )
            for i in range(len(valid_stages))
        ],
        dtype=np.float32,
    )

    payload = {
        "omega": np.int32(bundle.omega),
        "rhs_re": rhs_field.real.astype(np.float32),
        "rhs_im": rhs_field.imag.astype(np.float32),
        "true_re": u_true.real.astype(np.float32),
        "true_im": u_true.imag.astype(np.float32),
        "residual_re": residuals.real.astype(np.float32),
        "residual_im": residuals.imag.astype(np.float32),
        "correction_re": corrections.real.astype(np.float32),
        "correction_im": corrections.imag.astype(np.float32),
        "stages": np.array(valid_stages, dtype=np.int32),
        "rel_residuals": rel_residuals,
        "interior_correction_norms": interior_rel,
        "n_sources": np.int32(meta["n_sources"]),
        "px": meta["px"],
        "py": meta["py"],
        "amps": meta["amps"],
        "phases": meta["phases"],
        "source_sigma": np.float32(meta["sigma"]),
    }
    summary = {
        "omega": int(bundle.omega),
        "n_sources": int(meta["n_sources"]),
        "stages": valid_stages,
        "rel_residuals": [float(x) for x in rel_residuals],
    }
    return payload, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate codex GMRES residual dataset.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--omegas", type=int, nargs="+", default=[32])
    parser.add_argument("--n-problems", type=int, default=12)
    parser.add_argument("--gmres-iters", type=int, default=8)
    parser.add_argument("--save-stages", type=str, default="0,1,2,4,7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-sigma", type=float, default=2.0)
    parser.add_argument("--min-sources", type=int, default=3)
    parser.add_argument("--max-sources", type=int, default=6)
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--greens-cache", type=Path, default=None, help="Directory with greens_meta.json + fields file.")
    args = parser.parse_args()

    if args.max_sources < args.min_sources:
        raise ValueError("max-sources must be >= min-sources")

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    save_stages = parse_stages(args.save_stages)
    outdir = ensure_dir(args.outdir)
    problems_dir = ensure_dir(outdir / "problems")
    previews_dir = ensure_dir(outdir / "previews")
    greens_cache = None
    if args.greens_cache is not None:
        meta_path = args.greens_cache / "greens_meta.json"
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        fields_path = args.greens_cache / meta["fields_file"]
        fields = np.memmap(
            fields_path,
            mode="r",
            dtype=np.float16 if meta["dtype"] == "float16" else np.float32,
            shape=(len(meta["positions"]), 2, GRID_N, GRID_N),
        )
        greens_cache = {"positions": meta["positions"], "fields": fields}

    metadata = {
        "kind": "codex_residual_dataset",
        "grid_n": GRID_N,
        "n_pml": NPML,
        "interior_n": GRID_N - 2 * NPML,
        "omegas": args.omegas,
        "sigma0_map": {str(k): float(v) for k, v in SIGMA0_MAP.items()},
        "n_problems_per_omega": args.n_problems,
        "gmres_iters": args.gmres_iters,
        "save_stages": save_stages,
        "seed": args.seed,
        "source_sigma": args.source_sigma,
        "min_sources": args.min_sources,
        "max_sources": args.max_sources,
    }
    write_json(outdir / "metadata.json", metadata)

    manifest: list[dict] = []
    for omega in args.omegas:
        omega_dir = ensure_dir(problems_dir / f"omega_{omega:03d}")
        bundle = build_solver_bundle(omega=omega, n=GRID_N, n_pml=NPML)
        print(f"[omega={omega}] matrix ready, generating {args.n_problems} problems")
        for problem_idx in range(args.n_problems):
            payload, summary = generate_problem(
                bundle=bundle,
                rng=rng,
                gmres_iters=args.gmres_iters,
                save_stages=save_stages,
                sigma=args.source_sigma,
                min_sources=args.min_sources,
                max_sources=args.max_sources,
                greens_cache=greens_cache,
            )
            file_path = omega_dir / f"problem_{problem_idx:05d}.npz"
            np.savez_compressed(file_path, **payload)

            entry = {
                "omega": int(omega),
                "problem_idx": int(problem_idx),
                "path": str(file_path.relative_to(outdir)),
                "n_sources": int(summary["n_sources"]),
                "stages": summary["stages"],
                "rel_residuals": summary["rel_residuals"],
            }
            manifest.append(entry)
            print(
                f"  saved {file_path.name}  "
                f"sources={entry['n_sources']}  stages={entry['stages']}  "
                f"rr0={entry['rel_residuals'][0]:.2e} rr_last={entry['rel_residuals'][-1]:.2e}"
            )

            if problem_idx < args.preview_count:
                residuals = [
                    payload["residual_re"][i].astype(np.float32) + 1j * payload["residual_im"][i].astype(np.float32)
                    for i in range(payload["residual_re"].shape[0])
                ]
                corrections = [
                    payload["correction_re"][i].astype(np.float32) + 1j * payload["correction_im"][i].astype(np.float32)
                    for i in range(payload["correction_re"].shape[0])
                ]
                preview_path = previews_dir / f"omega_{omega:03d}_problem_{problem_idx:05d}.png"
                _plot_problem_summary(
                    save_path=preview_path,
                    omega=int(omega),
                    rhs=payload["rhs_re"] + 1j * payload["rhs_im"],
                    u_true=payload["true_re"] + 1j * payload["true_im"],
                    residuals=residuals,
                    corrections=corrections,
                    stages=list(range(len(residuals))),
                    rel_residuals=payload["rel_residuals"].tolist(),
                )

    with (outdir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row) + "\n")
    print(f"wrote manifest with {len(manifest)} problems to {outdir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
