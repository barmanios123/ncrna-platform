# Run comparison summary

## Metrics comparison

| run_name | best_model | cv_mean_r2 | final_rmse | final_mae | final_r2 | final_pearson_r | confidence_tier_excluded | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_with_confidence_tier | RandomForestRegressor | 0.765232 | 0.014682 | 0.001075 | 0.508947 | 0.718093 | False | Recovered baseline run from handoff; feature importance was dominated by confidence tier. |
| ablation_without_confidence_tier | ExtraTreesRegressor | 0.542211 | 0.015938 | 0.001670 | 0.421345 | 0.649375 | True | Current ablation run from output files. |

## Metric deltas

| metric | baseline | ablation | delta_ablation_minus_baseline |
| --- | --- | --- | --- |
| cv_mean_r2 | 0.765232 | 0.542211 | -0.223021 |
| final_rmse | 0.014682 | 0.015938 | 0.001256 |
| final_mae | 0.001075 | 0.001670 | 0.000595 |
| final_r2 | 0.508947 | 0.421345 | -0.087602 |
| final_pearson_r | 0.718093 | 0.649375 | -0.068718 |

## Top ablation features

| feature | importance |
| --- | --- |
| gf_disease_shift | 0.276655 |
| mean_abs_log2fc | 0.178173 |
| relevance_score | 0.093671 |
| risk_score | 0.043217 |
| gf_perturbation_impact | 0.038708 |
| gf_risk_adjustment | 0.035784 |
| symbol_TIMD4 | 0.033835 |
| symbol_ANKRD36 | 0.032200 |
| gf_geneformer_like_score | 0.026261 |
| symbol_PNISR | 0.024965 |
| symbol_MIAT | 0.021879 |
| mean_tau | 0.020293 |
| symbol_MALAT1 | 0.017618 |
| translational_score | 0.014383 |
| specificity_score | 0.011503 |
| expr_std_tcga_pancan | 0.009868 |
| expr_max_tcga_pancan | 0.008753 |
| symbol_ANKRD26 | 0.007909 |
| expr_q3_tcga_pancan | 0.007686 |
| symbol_ZMAT1 | 0.007432 |
| expr_iqr_tcga_pancan | 0.006695 |
| expr_mean_tcga_pancan | 0.006637 |
| expr_min_tcga_pancan | 0.006342 |
| log_tpm_disease | 0.006208 |
| symbol_SNHG12 | 0.004586 |

## Interpretation
The baseline run achieved higher apparent performance, but it included confidence-tier information that likely introduced label leakage. The ablation run removed confidence_tier and shifted feature importance toward biologically meaningful signals such as disease shift, perturbation impact, expression change, and relevance/specificity scores.