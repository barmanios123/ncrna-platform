from pathlib import Path
import json
import pandas as pd

OUTPUT_DIR = Path("output")
COMPARE_DIR = OUTPUT_DIR / "comparisons"
COMPARE_DIR.mkdir(parents=True, exist_ok=True)

BASELINE = {
    "run_name": "baseline_with_confidence_tier",
    "best_model": "RandomForestRegressor",
    "cv_mean_r2": 0.765232,
    "final_rmse": 0.014682385516136142,
    "final_mae": 0.0010746666666666869,
    "final_r2": 0.5089465957985322,
    "final_pearson_r": 0.7180931913106668,
    "confidence_tier_excluded": False,
    "notes": "Recovered baseline run from handoff; feature importance was dominated by confidence tier.",
}

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def load_current_metrics():
    metrics_path = OUTPUT_DIR / "training_metrics.json"
    benchmark_path = OUTPUT_DIR / "model_benchmark.csv"
    metadata_path = OUTPUT_DIR / "training_metadata.json"
    fi_path = OUTPUT_DIR / "feature_importance.csv"

    metrics = load_json(metrics_path)
    benchmark = pd.read_csv(benchmark_path)
    feature_importance = pd.read_csv(fi_path)
    metadata = load_json(metadata_path)

    best_row = benchmark.sort_values("mean_r2", ascending=False).iloc[0]

    current = {
        "run_name": "ablation_without_confidence_tier",
        "best_model": str(best_row["model_name"]),
        "cv_mean_r2": float(best_row["mean_r2"]),
        "final_rmse": float(metrics["rmse"]),
        "final_mae": float(metrics["mae"]),
        "final_r2": float(metrics["r2"]),
        "final_pearson_r": float(metrics["pearson_r"]),
        "confidence_tier_excluded": bool(metadata.get("confidence_tier_excluded", False)),
        "notes": "Current ablation run from output files.",
    }

    return current, benchmark, feature_importance, metadata

def build_metrics_comparison(current):
    df = pd.DataFrame([BASELINE, current])
    metric_cols = [
        "run_name",
        "best_model",
        "cv_mean_r2",
        "final_rmse",
        "final_mae",
        "final_r2",
        "final_pearson_r",
        "confidence_tier_excluded",
        "notes",
    ]
    return df[metric_cols]

def build_delta_summary(metrics_df):
    base = metrics_df.loc[
        metrics_df["run_name"] == "baseline_with_confidence_tier"
    ].iloc[0]
    abl = metrics_df.loc[
        metrics_df["run_name"] == "ablation_without_confidence_tier"
    ].iloc[0]

    rows = []
    for metric in [
        "cv_mean_r2",
        "final_rmse",
        "final_mae",
        "final_r2",
        "final_pearson_r",
    ]:
        rows.append(
            {
                "metric": metric,
                "baseline": base[metric],
                "ablation": abl[metric],
                "delta_ablation_minus_baseline": abl[metric] - base[metric],
            }
        )
    return pd.DataFrame(rows)

def save_top_features(feature_importance, n=25):
    top = feature_importance.head(n).copy()
    top.to_csv(COMPARE_DIR / "ablation_top_features.csv", index=False)
    return top

def df_to_md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows = []
    for _, row in df.iterrows():
        vals = []
        for v in row.tolist():
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v).replace("\n", " "))
        rows.append("| " + " | ".join(vals) + " |")

    return "\n".join([header, sep] + rows)

def write_summary_md(metrics_df, delta_df, top_features):
    lines = []
    lines.append("# Run comparison summary\n")
    lines.append("## Metrics comparison\n")
    lines.append(df_to_md_table(metrics_df))
    lines.append("\n## Metric deltas\n")
    lines.append(df_to_md_table(delta_df))
    lines.append("\n## Top ablation features\n")
    lines.append(df_to_md_table(top_features))
    lines.append(
        "\n## Interpretation\n"
        "The baseline run achieved higher apparent performance, but it included confidence-tier information "
        "that likely introduced label leakage. The ablation run removed confidence_tier and shifted feature "
        "importance toward biologically meaningful signals such as disease shift, perturbation impact, "
        "expression change, and relevance/specificity scores."
    )

    out_path = COMPARE_DIR / "run_comparison_summary.md"
    out_path.write_text("\n".join(lines))
    return out_path

def main():
    current, benchmark, feature_importance, metadata = load_current_metrics()
    metrics_df = build_metrics_comparison(current)
    delta_df = build_delta_summary(metrics_df)
    top_features = save_top_features(feature_importance, n=25)

    metrics_df.to_csv(COMPARE_DIR / "metrics_comparison.csv", index=False)
    delta_df.to_csv(COMPARE_DIR / "metrics_delta.csv", index=False)
    benchmark.to_csv(COMPARE_DIR / "current_model_benchmark.csv", index=False)

    summary_path = write_summary_md(metrics_df, delta_df, top_features)

    print("Wrote:")
    print(COMPARE_DIR / "metrics_comparison.csv")
    print(COMPARE_DIR / "metrics_delta.csv")
    print(COMPARE_DIR / "ablation_top_features.csv")
    print(COMPARE_DIR / "current_model_benchmark.csv")
    print(summary_path)

if __name__ == "__main__":
    main()