"""
ncRNA Target Intelligence Platform — Feature Engineering
Phase 2: includes curated target evidence + GEO/TCGA expression summary features
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPR_SUMMARY_PATH = (
    PROJECT_ROOT / "data" / "processed" / "expression_summary_features_merged.csv"
)


def _safe_read_sql(query: str, conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        return pd.read_sql_query(query, conn)
    except Exception:
        return pd.DataFrame()


def _load_expression_summary_features() -> pd.DataFrame:
    """
    Load merged expression summary features.

    Expected columns include:
      - ensembl_id
      - symbol or gene_symbol
      - expr_mean
      - expr_std
      - expr_median
      - expr_max
      - expr_nonzero_fraction
      - expr_cv
      - expr_log1p_mean
      - expr_log1p_max
      - expr_mean_tcga_pancan
      - expr_median_tcga_pancan
      - expr_std_tcga_pancan
      - expr_min_tcga_pancan
      - expr_max_tcga_pancan
      - expr_q1_tcga_pancan
      - expr_q3_tcga_pancan
      - expr_iqr_tcga_pancan
      - expr_prevalence_tcga_pancan
      - in_tcga_pancan_expr
    """
    if not EXPR_SUMMARY_PATH.exists():
        return pd.DataFrame(
            columns=[
                "ensembl_id",
                "symbol",
                "expr_mean",
                "expr_std",
                "expr_median",
                "expr_max",
                "expr_nonzero_fraction",
                "expr_cv",
                "expr_log1p_mean",
                "expr_log1p_max",
                "expr_mean_tcga_pancan",
                "expr_median_tcga_pancan",
                "expr_std_tcga_pancan",
                "expr_min_tcga_pancan",
                "expr_max_tcga_pancan",
                "expr_q1_tcga_pancan",
                "expr_q3_tcga_pancan",
                "expr_iqr_tcga_pancan",
                "expr_prevalence_tcga_pancan",
                "in_tcga_pancan_expr",
            ]
        )

    df = pd.read_csv(EXPR_SUMMARY_PATH)

    if "symbol" not in df.columns:
        for cand in ["gene_symbol", "GeneSymbol"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "symbol"})
                break

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip()

    if "ensembl_id" in df.columns:
        df["ensembl_id"] = df["ensembl_id"].astype(str).str.strip()

    return df


def build_feature_matrix(
    db_path: str = "ncrna_platform.db",
    disease_id: str = "DIS_001",
    context_id: str = "CTX_001",
) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)

    # Expression / GEO-like evidence
    expr_q = """
        SELECT
            e.ncrna_id,
            AVG(ABS(e.log2fc)) AS mean_abs_log2fc,
            AVG(e.specificity_tau) AS mean_tau,
            SUM(CASE WHEN e.padj < 0.05 THEN 1 ELSE 0 END) AS n_sig_datasets,
            COUNT(*) AS n_datasets,
            AVG(e.tpm_disease) AS mean_tpm_disease,
            SUM(CASE WHEN e.direction = 'up' THEN 1 ELSE 0 END) AS n_up,
            SUM(CASE WHEN e.direction = 'down' THEN 1 ELSE 0 END) AS n_down
        FROM expression_evidence e
        WHERE e.disease_id = :disease_id
          AND e.context_id = :context_id
        GROUP BY e.ncrna_id
    """
    expr_df = _safe_read_sql(expr_q, conn)
    if not expr_df.empty:
        expr_df["de_consistency"] = expr_df.apply(
            lambda r: (
                max(r["n_up"], r["n_down"]) / r["n_datasets"]
                if r["n_datasets"] > 0
                else 0.0
            ),
            axis=1,
        )
        expr_df["log_tpm_disease"] = np.log1p(expr_df["mean_tpm_disease"].fillna(0.0))
        expr_df["sig_rate"] = (
            expr_df["n_sig_datasets"]
            / expr_df["n_datasets"].clip(lower=1)
        )

    # Tractability / modality
    tract_q = """
        SELECT
            ncrna_id,
            localization,
            isoform_count,
            gc_content,
            secondary_structure_score,
            aso_accessible,
            sirna_compatible,
            small_mol_bindable,
            crispr_feasible,
            best_modality
        FROM tractability_features
    """
    tract_df = _safe_read_sql(tract_q, conn)
    if not tract_df.empty:
        tract_df["modality_breadth"] = (
            tract_df["aso_accessible"].fillna(0)
            + tract_df["sirna_compatible"].fillna(0)
            + tract_df["small_mol_bindable"].fillna(0)
            + tract_df["crispr_feasible"].fillna(0)
        )
        tract_df["nuclear_flag"] = (
            tract_df["localization"].fillna("").astype(str).str.lower()
            == "nuclear"
        ).astype(int)
        tract_df["isoform_complexity"] = np.log1p(
            tract_df["isoform_count"].fillna(0)
        )

    # Perturbation evidence
    pert_q = """
        SELECT
            ncrna_id,
            COUNT(*) AS n_perturbation_studies,
            AVG(ABS(effect_size)) AS mean_pert_effect,
            SUM(CASE WHEN confidence = 'high' THEN 1 ELSE 0 END) AS n_high_conf_perts
        FROM perturbation_evidence
        GROUP BY ncrna_id
    """
    pert_df = _safe_read_sql(pert_q, conn)

    # Clinical links
    clin_q = """
        SELECT
            ncrna_id,
            COUNT(*) AS n_clinical_studies,
            AVG(ABS(correlation_r)) AS mean_clinical_r,
            SUM(CASE WHEN pvalue < 0.01 THEN 1 ELSE 0 END) AS n_sig_clinical,
            SUM(CASE WHEN biomarker_type = 'prognostic' THEN 1 ELSE 0 END) AS n_prognostic
        FROM clinical_links
        GROUP BY ncrna_id
    """
    clin_df = _safe_read_sql(clin_q, conn)

    # Pathway / network
    pw_q = """
        SELECT
            ncrna_id,
            COUNT(*) AS n_pathways,
            MIN(enrichment_fdr) AS best_pathway_fdr,
            MAX(ABS(pearson_r)) AS max_hub_correlation
        FROM pathway_links
        GROUP BY ncrna_id
    """
    pw_df = _safe_read_sql(pw_q, conn)

    # Literature
    lit_q = """
        SELECT
            ncrna_id,
            COUNT(*) AS n_lit_statements,
            AVG(confidence_score) AS mean_lit_confidence,
            SUM(is_contradictory) AS n_contradictory,
            COUNT(DISTINCT pubmed_id) AS n_unique_papers,
            MAX(year) AS most_recent_year
        FROM literature_evidence
        GROUP BY ncrna_id
    """
    lit_df = _safe_read_sql(lit_q, conn)

    # Master ncRNA registry
    master_q = """
        SELECT
            ncrna_id,
            symbol,
            biotype,
            conservation_score,
            transcript_count
        FROM ncrna_master
    """
    master_df = pd.read_sql_query(master_q, conn)

    # Curated liver-disease targets
    curated_q = """
        SELECT
            symbol,
            MAX(
                CASE
                    WHEN evidence_tier = 'Tier1' THEN 1.00
                    WHEN evidence_tier = 'Tier2' THEN 0.66
                    WHEN evidence_tier = 'Tier3' THEN 0.33
                    ELSE 0
                END
            ) AS curated_tier_score,
            MAX(CASE WHEN is_contradictory = 1 THEN 1 ELSE 0 END) AS curated_contradiction_flag,
            MAX(CASE WHEN disease_stage = 'HCC' THEN 1 ELSE 0 END) AS curated_hcc_flag,
            MAX(CASE WHEN disease_stage = 'MASLD' THEN 1 ELSE 0 END) AS curated_masld_flag,
            MAX(CASE WHEN disease_stage = 'Fibrosis' THEN 1 ELSE 0 END) AS curated_fibrosis_flag,
            MAX(CASE WHEN disease_stage = 'MASH' THEN 1 ELSE 0 END) AS curated_mash_flag,
            COUNT(*) AS curated_record_count
        FROM curated_targets
        GROUP BY symbol
    """
    curated_df = _safe_read_sql(curated_q, conn)

    conn.close()

    # Start from master list
    df = master_df.copy()

    # Merge by ncrna_id
    for sub_df in [expr_df, tract_df, pert_df, clin_df, pw_df, lit_df]:
        if not sub_df.empty:
            df = df.merge(sub_df, on="ncrna_id", how="left")

    # Merge curated evidence by symbol
    if not curated_df.empty:
        curated_df["symbol"] = curated_df["symbol"].astype(str).str.strip()
        df["symbol"] = df["symbol"].astype(str).str.strip()
        df = df.merge(curated_df, on="symbol", how="left")

    # Merge expression summary (GEO/TCGA) by symbol
    expr_summary_df = _load_expression_summary_features()
    if not expr_summary_df.empty:
        df["symbol"] = df["symbol"].astype(str).str.strip()
        expr_summary_df["symbol"] = expr_summary_df["symbol"].astype(str).str.strip()
        df = df.merge(expr_summary_df, on="symbol", how="left", suffixes=("", "_expr"))

        if "ensembl_id_expr" in df.columns:
            df = df.drop(columns=["ensembl_id_expr"])

    # Basic cleaning
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    if "in_tcga_pancan_expr" in df.columns:
        df["in_tcga_pancan_expr"] = (
            df["in_tcga_pancan_expr"].fillna(False).astype(int)
        )

    df["curated_exists"] = (
        (df["curated_record_count"] > 0).astype(int)
        if "curated_record_count" in df.columns
        else 0
    )

    # Backfill key columns used in risk and scoring
    if "mean_tau" not in df.columns:
        df["mean_tau"] = 0.0
    if "n_contradictory" not in df.columns:
        df["n_contradictory"] = 0.0
    if "n_lit_statements" not in df.columns:
        df["n_lit_statements"] = 0.0
    if "isoform_count" not in df.columns:
        df["isoform_count"] = 0.0

    # Risk flags
    df["risk_ubiquitous"] = (df["mean_tau"] < 0.4).astype(int)
    df["risk_contradictory_lit"] = (
        (df["n_contradictory"] / df["n_lit_statements"].clip(lower=1)) > 0.3
    ).astype(int)
    df["risk_high_isoforms"] = (df["isoform_count"] > 5).astype(int)

    print(f"✅ Feature matrix: {df.shape[0]} ncRNAs × {df.shape[1]} features")
    return df


if __name__ == "__main__":
    feat_df = build_feature_matrix()
    print(feat_df.head())