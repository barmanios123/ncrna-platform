from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "ncrna_platform.db"
EXPR_PATH = PROJECT_ROOT / "data" / "processed" / "gse126848_expression.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "expression_summary_features.csv"


def normalize_symbol_column(df: pd.DataFrame) -> pd.DataFrame:
    first_col = df.columns[0]
    if first_col != "symbol":
        df = df.rename(columns={first_col: "symbol"})
    return df


def build_matrix_summary_features() -> pd.DataFrame:
    if not EXPR_PATH.exists():
        print(f"⚠️ Expression matrix file not found, skipping CSV summary step: {EXPR_PATH}")
        return pd.DataFrame()

    df = pd.read_csv(EXPR_PATH)
    df = normalize_symbol_column(df)

    sample_cols = [c for c in df.columns if c != "symbol"]
    if not sample_cols:
        raise ValueError("No sample columns found in expression matrix")

    for c in sample_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    expr = df[sample_cols]

    out = pd.DataFrame(
        {
            "symbol": df["symbol"].astype(str),
            "expr_mean": expr.mean(axis=1),
            "expr_std": expr.std(axis=1),
            "expr_median": expr.median(axis=1),
            "expr_max": expr.max(axis=1),
            "expr_nonzero_fraction": (expr > 0).mean(axis=1),
        }
    )

    out["expr_cv"] = np.where(out["expr_mean"] > 0, out["expr_std"] / out["expr_mean"], 0.0)
    out["expr_log1p_mean"] = np.log1p(out["expr_mean"])
    out["expr_log1p_max"] = np.log1p(out["expr_max"])

    out = out.drop_duplicates(subset=["symbol"]).sort_values("symbol").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"✅ Wrote CSV summary: {OUT_PATH} with shape {out.shape}")
    print(out.head(10).to_string(index=False))
    return out


def compute_de_consistency(group: pd.DataFrame) -> float:
    signif = group[group["padj"] <= 0.05].copy()
    if signif.empty:
        return 0.0

    n_up = (signif["direction"].astype(str).str.lower() == "up").sum()
    n_down = (signif["direction"].astype(str).str.lower() == "down").sum()
    total = n_up + n_down

    if total == 0:
        return 0.0

    return float(max(n_up, n_down) / total)


def build_expression_feature_table() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    expr = pd.read_sql_query(
        """
        SELECT
            ncrna_id,
            disease_id,
            context_id,
            log2fc,
            pvalue,
            padj,
            basemean,
            tpm_disease,
            tpm_healthy,
            specificity_tau,
            direction,
            dataset_id
        FROM expression_evidence
        """,
        conn,
    )

    if expr.empty:
        conn.close()
        raise ValueError("expression_evidence is empty; cannot build expression features")

    numeric_cols = [
        "log2fc",
        "pvalue",
        "padj",
        "basemean",
        "tpm_disease",
        "tpm_healthy",
        "specificity_tau",
    ]
    for col in numeric_cols:
        expr[col] = pd.to_numeric(expr[col], errors="coerce")

    grouped_rows = []
    for (ncrna_id, disease_id, context_id), g in expr.groupby(
        ["ncrna_id", "disease_id", "context_id"], dropna=False
    ):
        log2fc_vals = g["log2fc"].dropna()
        padj_vals = g["padj"].dropna()
        tpm_disease_vals = g["tpm_disease"].dropna()
        tau_vals = g["specificity_tau"].dropna()

        mean_abs_log2fc = float(np.mean(np.abs(log2fc_vals))) if len(log2fc_vals) else 0.0
        sig_rate = float((g["padj"] <= 0.05).fillna(False).mean()) if len(g) else 0.0
        de_consistency = compute_de_consistency(g)

        mean_log2fc = float(log2fc_vals.mean()) if len(log2fc_vals) else 0.0
        max_abs_log2fc = float(np.max(np.abs(log2fc_vals))) if len(log2fc_vals) else 0.0
        min_padj = float(padj_vals.min()) if len(padj_vals) else 1.0
        n_expression_studies = int(g["dataset_id"].nunique()) if "dataset_id" in g.columns else int(len(g))
        mean_tpm_disease = float(tpm_disease_vals.mean()) if len(tpm_disease_vals) else 0.0
        mean_tpm_healthy = float(g["tpm_healthy"].dropna().mean()) if g["tpm_healthy"].notna().any() else 0.0
        mean_tau = float(tau_vals.mean()) if len(tau_vals) else 0.0
        log_tpm_disease = float(np.log1p(mean_tpm_disease)) if mean_tpm_disease > 0 else 0.0

        n_sig = int((g["padj"] <= 0.05).fillna(False).sum())
        n_up_sig = int(((g["padj"] <= 0.05) & (g["direction"].astype(str).str.lower() == "up")).sum())
        n_down_sig = int(((g["padj"] <= 0.05) & (g["direction"].astype(str).str.lower() == "down")).sum())

        grouped_rows.append(
            {
                "ncrna_id": ncrna_id,
                "disease_id": disease_id,
                "context_id": context_id,
                "n_expression_studies": n_expression_studies,
                "mean_abs_log2fc": mean_abs_log2fc,
                "mean_log2fc": mean_log2fc,
                "max_abs_log2fc": max_abs_log2fc,
                "sig_rate": sig_rate,
                "de_consistency": de_consistency,
                "min_padj": min_padj,
                "n_sig_expression_studies": n_sig,
                "n_up_sig_expression_studies": n_up_sig,
                "n_down_sig_expression_studies": n_down_sig,
                "mean_tpm_disease": mean_tpm_disease,
                "mean_tpm_healthy": mean_tpm_healthy,
                "mean_tau": mean_tau,
                "log_tpm_disease": log_tpm_disease,
            }
        )

    feat_df = pd.DataFrame(grouped_rows).sort_values(
        ["disease_id", "context_id", "ncrna_id"]
    ).reset_index(drop=True)

    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS expression_features (
            ncrna_id TEXT,
            disease_id TEXT,
            context_id TEXT,
            n_expression_studies INTEGER,
            mean_abs_log2fc REAL,
            mean_log2fc REAL,
            max_abs_log2fc REAL,
            sig_rate REAL,
            de_consistency REAL,
            min_padj REAL,
            n_sig_expression_studies INTEGER,
            n_up_sig_expression_studies INTEGER,
            n_down_sig_expression_studies INTEGER,
            mean_tpm_disease REAL,
            mean_tpm_healthy REAL,
            mean_tau REAL,
            log_tpm_disease REAL,
            PRIMARY KEY (ncrna_id, disease_id, context_id)
        )
        """
    )

    cur.execute("DELETE FROM expression_features")

    feat_df.to_sql("expression_features", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()

    print(f"✅ Built expression_features table with shape {feat_df.shape}")
    print(feat_df.head(10).to_string(index=False))
    return feat_df


def main():
    build_matrix_summary_features()
    build_expression_feature_table()


if __name__ == "__main__":
    main()