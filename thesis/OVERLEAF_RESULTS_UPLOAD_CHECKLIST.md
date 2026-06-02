# Results chapter Overleaf upload checklist

Use `thesis/chapter_results_overleaf_pending_64_128.tex` as the new chapter draft.

## Upload now

Upload these into Overleaf under the same relative paths used in the chapter.

- `figures/ch7/training_16_32.png`
- `figures/ch7/training_32_64.png`
- `figures/ch7/training_64_128.png`
- `figures/ch7/fig_voronoi_prediction.png`
- `figures/ch7/fig_atom_decomposition.png`
- `figures/ch7/gmres_clean_64_128.png`
- `figures/ch7/2d_final_true_residual_bars.png`
- `figures/ch7/2d_initial_residual_comparison.png`
- `figures/ch7/2d_adaptive_16_32/2d_16_32_iterations_to_convergence.png`
- `figures/ch7/2d_adaptive_16_32/2d_16_32_true_residual_convergence_mean.png`

Keep the PDF versions too if Overleaf prefers vector output, but the chapter currently references PNG files.

## Referenced but not currently local

The chapter still references these 1D figures through the compile-safe `\resultsfigure` wrapper. Overleaf will show labelled placeholders unless you upload the real files to these paths or temporarily remove/comment these figures. I did not find local copies in this workspace.

- `figures/ch7/01_true_residual_vs_iteration.png`
- `figures/ch7/03_field_error_vs_iteration.png`
- `figures/ch7/01_left_vs_right_true_residual.png`
- `figures/ch7/02_left_vs_right_precond_residual.png`
- `figures/ch7/combined_1d_dirichlet_pml_true_residual.png`
- `figures/ch7/03_pml_field_error_vs_iteration.png`

## Download when the 32/64 and 64/128 adaptive runs finish

For each pending pair, download the generated adaptive-convergence folder. Upload the plots below, then paste the summary numbers into the inline LaTeX table in the chapter:

- `figures/ch7/2d_adaptive_32_64/2d_32_64_iterations_to_convergence.png`
- `figures/ch7/2d_adaptive_32_64/2d_32_64_true_residual_convergence_mean.png`
- `figures/ch7/2d_adaptive_32_64/2d_32_64_true_residual_initial_final.png`
- `figures/ch7/2d_adaptive_64_128/2d_64_128_iterations_to_convergence.png`
- `figures/ch7/2d_adaptive_64_128/2d_64_128_true_residual_convergence_mean.png`
- `figures/ch7/2d_adaptive_64_128/2d_64_128_true_residual_initial_final.png`

The chapter already has a paragraph marking these higher-frequency adaptive runs as pending. Once the files exist, replace that paragraph with the actual pairwise iteration savings.
