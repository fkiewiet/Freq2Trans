#!/usr/bin/env python3
"""Patch experiments/2d/evaluate_warmstarts_2d.py to write iteration_metrics.csv."""

from pathlib import Path


PATH = Path("experiments/2d/evaluate_warmstarts_2d.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Expected one {label} block, found {count}")
    return text.replace(old, new)


def main() -> None:
    text = PATH.read_text()
    if "iteration_metrics.csv" in text and "def iteration_diagnostics(" in text:
        print(f"{PATH} already has iteration_metrics support")
        return

    residual_curve = '''def true_residual_curve(A, b: np.ndarray, x0: np.ndarray, M_lu, steps: int) -> list[float]:
    curve = []
    for k in range(steps + 1):
        xk = fgmres_solution_after_k(A, b, x0, M_lu, k)
        curve.append(true_residual(A, b, xk))
    return curve
'''

    diagnostics = residual_curve + '''

def iteration_diagnostics(
    A,
    b: np.ndarray,
    x0: np.ndarray,
    u_high: np.ndarray,
    cfg: Eval2DConfig,
    M_lu,
    Mb_norm: float,
    steps: int,
) -> tuple[list[float], list[dict], np.ndarray]:
    rows = []
    curve = []
    x_final = x0.reshape(-1).astype(np.complex128)
    for k in range(steps + 1):
        xk = fgmres_solution_after_k(A, b, x0, M_lu, k)
        xk_grid = xk.reshape(cfg.n, cfg.n)
        r = b - A @ xk
        true_res = float(np.linalg.norm(r) / max(np.linalg.norm(b), 1e-30))
        pre_res = float(np.linalg.norm(M_lu.solve(r)) / max(Mb_norm, 1e-30))
        curve.append(true_res)
        rows.append(
            {
                "iteration": k,
                "true_residual": true_res,
                "precond_residual": pre_res,
                "interior_error": rel_l2_2d(xk_grid, u_high, cfg, full=False),
                "full_error": rel_l2_2d(xk_grid, u_high, cfg, full=True),
            }
        )
        x_final = xk
    return curve, rows, x_final
'''
    text = replace_once(text, residual_curve, diagnostics, "true_residual_curve")

    text = replace_once(
        text,
        "    sample_rows = []\n",
        "    sample_rows = []\n    iteration_rows = []\n",
        "sample_rows",
    )

    old_loop = '''            curve = true_residual_curve(A_h, b, x0, csl_lu, cfg.gmres_steps)
            curves[method].append(curve)
            x_final = fgmres_solution_after_k(A_h, b, x0, csl_lu, cfg.gmres_steps)
            sample_rows.append(
'''
    new_loop = '''            curve, method_iteration_rows, x_final = iteration_diagnostics(
                A_h, b, x0, u_high, cfg, csl_lu, Mb_norm, cfg.gmres_steps
            )
            curves[method].append(curve)
            for row in method_iteration_rows:
                iteration_rows.append(
                    {
                        "pair": pair_tag,
                        "sample": sample,
                        "method": method,
                        **row,
                    }
                )
            sample_rows.append(
'''
    text = replace_once(text, old_loop, new_loop, "curve computation")

    old_write = '''    with (outdir / "sample_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)

    summary_rows = []
'''
    new_write = '''    with (outdir / "sample_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)

    with (outdir / "iteration_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(iteration_rows[0].keys()))
        writer.writeheader()
        writer.writerows(iteration_rows)

    summary_rows = []
'''
    text = replace_once(text, old_write, new_write, "sample_metrics write")

    text = replace_once(
        text,
        '                    "Preconditioned residuals are ||M_CSL^{-1}(b - A_high x_k)|| / ||M_CSL^{-1}b||.",\n',
        '                    "Preconditioned residuals are ||M_CSL^{-1}(b - A_high x_k)|| / ||M_CSL^{-1}b||.",\n'
        '                    "iteration_metrics.csv stores every sample/method/iteration used to make convergence plots.",\n',
        "config note",
    )

    PATH.write_text(text)
    print(f"patched {PATH}")


if __name__ == "__main__":
    main()
