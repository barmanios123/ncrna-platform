from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPR_PATH = PROJECT_ROOT / "data" / "processed" / "gse126848_expression.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "expression_summary_features.csv"


def normalize_symbol_column(df: pd.DataFrame) -> pd.DataFrame:
    first_col = df.columns[0]
    if first_col != "symbol":
        df = df.rename(columns={first_col: "symbol"})
    return df


def main():
    if not EXPR_PATH.exists():
        raise FileNotFoundError(f"Expression file not found: {EXPR_PATH}")

    df = pd.read_csv(EXPR_PATH)
    df = normalize_symbol_column(df)

    sample_cols = [c for c in df.columns if c != "symbol"]
    if not sample_cols:
        raise ValueError("No sample columns found in expression matrix")

    for c in sample_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    expr = df[sample_cols]

    out = pd.DataFrame({
        "symbol": df["symbol"].astype(str),
        "expr_mean": expr.mean(axis=1),
        "expr_std": expr.std(axis=1),
        "expr_median": expr.median(axis=1),
        "expr_max": expr.max(axis=1),
        "expr_nonzero_fraction": (expr > 0).mean(axis=1),
    })

    out["expr_cv"] = np.where(out["expr_mean"] > 0, out["expr_std"] / out["expr_mean"], 0.0)
    out["expr_log1p_mean"] = np.log1p(out["expr_mean"])
    out["expr_log1p_max"] = np.log1p(out["expr_max"])

    out = out.drop_duplicates(subset=["symbol"]).sort_values("symbol").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Wrote {OUT_PATH} with shape {out.shape}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()