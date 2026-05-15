"""
ncRNA Target Intelligence Platform — Translational Scoring Engine
Phase 2: curated target evidence integrated + raw provenance persistence
"""

import json
import sqlite3

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

try:
    import xgboost as xgb

    HAS_XGB = True
except Exception as e:
    xgb = None
    HAS_XGB = False
    print(f"⚠️ XGBoost unavailable; using GradientBoostingRegressor instead. Reason: {e}")

from models.features import build_feature_matrix

SCORE_WEIGHTS = {
    "relevance": 0.22,
    "specificity": 0.12,
    "mechanism": 0.18,
    "tractability": 0.12,
    "human_evidence": 0.16,
    "curated": 0.15,
    "risk": 0.05,
}

RELEVANCE_FEATURES = [
    "mean_abs_log2fc",
    "sig_rate",
    "de_consistency",
]

SPECIFICITY_FEATURES = [
    "mean_tau",
    "log_tpm_disease",
]

MECHANISM_FEATURES = [
    "n_perturbation_studies",
    "mean_pert_effect",
    "n_high_conf_perts",
    "n_pathways",
    "max_hub_correlation",
]

TRACTABILITY_FEATURES = [
    "modality_breadth",
    "aso_accessible",
    "sirna_compatible",
    "crispr_feasible",
]

HUMAN_EVIDENCE_FEATURES = [
    "n_clinical_studies",
    "mean_clinical_r",
    "n_sig_clinical",
    "n_prognostic",
    "mean_lit_confidence",
    "n_unique_papers",
]

CURATED_FEATURES = [
    "curated_exists",
    "curated_tier_score",
    "curated_hcc_flag",
    "curated_masld_flag",
    "curated_fibrosis_flag",
    "curated_mash_flag",
]

RISK_FEATURES = [
    "risk_ubiquitous",
    "risk_contradictory_lit",
    "risk_high_isoforms",
    "curated_contradiction_flag",
]

ALL_FEATURE_COLS = (
    RELEVANCE_FEATURES
    + SPECIFICITY_FEATURES
    + MECHANISM_FEATURES
    + TRACTABILITY_FEATURES
    + HUMAN_EVIDENCE_FEATURES
    + CURATED_FEATURES
    + RISK_FEATURES
)

TCGA_FEATURES = [
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

GF_REGULATORY_FEATURES = [
    "n_pathways",
    "max_hub_correlation",
    "mean_lit_confidence",
    "n_unique_papers",
    "conservation_score",
]

GF_PERTURBATION_FEATURES = [
    "n_perturbation_studies",
    "n_high_conf_perts",
    "mean_pert_effect",
    "de_consistency",
]

GF_DISEASE_SHIFT_FEATURES = [
    "mean_abs_log2fc",
    "sig_rate",
    "mean_clinical_r",
    "n_sig_clinical",
]

GF_CONTEXT_FEATURES = [
    "mean_tau",
    "log_tpm_disease",
    "expr_prevalence_tcga_pancan",
    "in_tcga_pancan_expr",
]

GF_RISK_FEATURES = [
    "risk_ubiquitous",
    "risk_contradictory_lit",
    "risk_high_isoforms",
    "curated_contradiction_flag",
]


def compute_component_score(df: pd.DataFrame, feature_cols: list) -> pd.Series:
    cols = [c for c in feature_cols if c in df.columns]
    if len(cols) == 0:
        return pd.Series(0.0, index=df.index)

    sub = df[cols].copy().fillna(0)
    if sub.shape[1] == 0:
        return pd.Series(0.0, index=df.index)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(sub)
    return pd.Series(scaled.mean(axis=1), index=df.index)


def compute_risk_penalty(df: pd.DataFrame) -> pd.Series:
    risk_raw = compute_component_score(df, RISK_FEATURES)
    return 1.0 - risk_raw


def compute_translational_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().fillna(0)

    df["relevance_score"] = compute_component_score(df, RELEVANCE_FEATURES)
    df["specificity_score"] = compute_component_score(df, SPECIFICITY_FEATURES)
    df["mechanism_score"] = compute_component_score(df, MECHANISM_FEATURES)
    df["tractability_score"] = compute_component_score(df, TRACTABILITY_FEATURES)
    df["human_evidence_score"] = compute_component_score(df, HUMAN_EVIDENCE_FEATURES)
    df["curated_score"] = compute_component_score(df, CURATED_FEATURES)
    df["risk_score"] = compute_risk_penalty(df)

    df["translational_score"] = (
        SCORE_WEIGHTS["relevance"] * df["relevance_score"]
        + SCORE_WEIGHTS["specificity"] * df["specificity_score"]
        + SCORE_WEIGHTS["mechanism"] * df["mechanism_score"]
        + SCORE_WEIGHTS["tractability"] * df["tractability_score"]
        + SCORE_WEIGHTS["human_evidence"] * df["human_evidence_score"]
        + SCORE_WEIGHTS["curated"] * df["curated_score"]
        + SCORE_WEIGHTS["risk"] * df["risk_score"]
    ).clip(0, 1)

    def assign_tier(score):
        if score >= 0.68:
            return "Tier 1 — High Confidence"
        if score >= 0.42:
            return "Tier 2 — Moderate Confidence"
        return "Tier 3 — Exploratory"

    df["confidence_tier"] = df["translational_score"].apply(assign_tier)
    return df.sort_values("translational_score", ascending=False)


def compute_geneformer_like_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().fillna(0)

    df["gf_regulatory_centrality"] = compute_component_score(df, GF_REGULATORY_FEATURES)
    df["gf_perturbation_impact"] = compute_component_score(df, GF_PERTURBATION_FEATURES)
    df["gf_disease_shift"] = compute_component_score(df, GF_DISEASE_SHIFT_FEATURES)
    df["gf_context_support"] = compute_component_score(df, GF_CONTEXT_FEATURES)

    risk_raw = compute_component_score(df, GF_RISK_FEATURES)
    df["gf_risk_adjustment"] = 1.0 - risk_raw

    df["gf_geneformer_like_score"] = (
        0.30 * df["gf_regulatory_centrality"].fillna(0.0)
        + 0.30 * df["gf_perturbation_impact"].fillna(0.0)
        + 0.25 * df["gf_disease_shift"].fillna(0.0)
        + 0.10 * df["gf_context_support"].fillna(0.0)
        + 0.05 * df["gf_risk_adjustment"].fillna(0.0)
    ).clip(0, 1)

    return df


def train_ranking_model(df: pd.DataFrame, label_col: str = "translational_score") -> dict:
    feature_cols = [c for c in ALL_FEATURE_COLS if c in df.columns]
    X = df[feature_cols].fillna(0).values
    y = df[label_col].values

    if HAS_XGB and xgb is not None:
        try:
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
            )
            model_name = "XGBoost"
        except Exception:
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
            model_name = "GradientBoostingRegressor"
    else:
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        model_name = "GradientBoostingRegressor"

    cv = min(5, len(df))
    if cv >= 2:
        try:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        except Exception:
            cv_mean = 0.0
            cv_std = 0.0
    else:
        cv_mean = 0.0
        cv_std = 0.0

    model.fit(X, y)

    importances = getattr(model, "feature_importances_", [0] * len(feature_cols))
    importances = dict(zip(feature_cols, importances))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    print(f"✅ Model trained with {model_name} | CV R² = {cv_mean:.3f} ± {cv_std:.3f}")

    return {
        "model": model,
        "model_name": model_name,
        "feature_importances": importances,
        "cv_r2_mean": cv_mean,
        "cv_r2_std": cv_std,
        "feature_cols": feature_cols,
    }


EXPERIMENT_LOGIC = {
    "high_curated_high_relevance": [
        "Validate expression in independent liver cohort",
        "ASO knockdown in PHH or HepG2 with lipid accumulation readout",
        "qPCR confirmation across disease-stage liver samples",
    ],
    "high_mechanism_low_clinical": [
        "Add orthogonal clinical validation cohort",
        "Run perturbation RNA-seq to connect target to pathway mechanism",
        "Test biomarker detectability in serum/exosomes",
    ],
    "contradictory_target": [
        "Manual literature review before wet-lab commitment",
        "Stage-specific validation in hepatocyte and stellate models",
        "Check isoform-specific expression before ASO design",
    ],
    "default": [
        "Validate differential expression in an independent cohort",
        "Review pathway support and perturbation evidence",
        "Prioritize for targeted follow-up assay panel",
    ],
}


def recommend_experiments(row: pd.Series) -> list:
    if row.get("curated_tier_score", 0) >= 0.66 and row.get("relevance_score", 0) >= 0.5:
        return EXPERIMENT_LOGIC["high_curated_high_relevance"]
    if row.get("curated_contradiction_flag", 0) == 1:
        return EXPERIMENT_LOGIC["contradictory_target"]
    if row.get("mechanism_score", 0) > 0.5 and row.get("human_evidence_score", 0) < 0.3:
        return EXPERIMENT_LOGIC["high_mechanism_low_clinical"]
    return EXPERIMENT_LOGIC["default"]


def build_risk_flags(row: pd.Series) -> list:
    flags = []

    if row.get("risk_ubiquitous", 0):
        flags.append("Low tissue specificity (Tau < 0.4) — systemic liability risk")
    if row.get("risk_contradictory_lit", 0):
        flags.append("High contradictory literature fraction")
    if row.get("risk_high_isoforms", 0):
        flags.append("High isoform complexity may complicate oligo design")
    if row.get("curated_contradiction_flag", 0):
        flags.append("Manual curation flagged this target as contradictory/context-dependent")
    if row.get("curated_exists", 0) == 0:
        flags.append("No curated liver-disease evidence currently linked")

    return flags if flags else ["No major risk flags identified"]


def ensure_target_scores_schema(conn: sqlite3.Connection):
    c = conn.cursor()

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS target_scores (
            score_id TEXT PRIMARY KEY,
            ncrna_id TEXT,
            disease_id TEXT,
            context_id TEXT,
            relevance_score REAL,
            specificity_score REAL,
            mechanism_score REAL,
            tractability_score REAL,
            human_evidence_score REAL,
            risk_score REAL,
            translational_score REAL,
            confidence_tier TEXT,
            top_evidence TEXT,
            risk_flags TEXT,
            recommended_experiments TEXT,
            model_version TEXT,
            scored_date TEXT,
            expr_mean_tcga_pancan REAL,
            expr_median_tcga_pancan REAL,
            expr_std_tcga_pancan REAL,
            expr_min_tcga_pancan REAL,
            expr_max_tcga_pancan REAL,
            expr_q1_tcga_pancan REAL,
            expr_q3_tcga_pancan REAL,
            expr_iqr_tcga_pancan REAL,
            expr_prevalence_tcga_pancan REAL,
            in_tcga_pancan_expr INTEGER,
            gf_geneformer_like_score REAL,
            gf_regulatory_centrality REAL,
            gf_perturbation_impact REAL,
            gf_disease_shift REAL,
            gf_context_support REAL,
            gf_risk_adjustment REAL,
            gf_model_version TEXT,
            n_pathways REAL,
            max_hub_correlation REAL,
            mean_lit_confidence REAL,
            n_unique_papers REAL,
            conservation_score REAL,
            n_perturbation_studies REAL,
            n_high_conf_perts REAL,
            mean_pert_effect REAL,
            de_consistency REAL,
            mean_abs_log2fc REAL,
            sig_rate REAL,
            mean_clinical_r REAL,
            n_sig_clinical REAL,
            mean_tau REAL,
            log_tpm_disease REAL,
            risk_ubiquitous INTEGER,
            risk_contradictory_lit INTEGER,
            risk_high_isoforms INTEGER,
            curated_contradiction_flag INTEGER
        )
        """
    )

    existing_cols = pd.read_sql_query("PRAGMA table_info(target_scores);", conn)["name"].tolist()

    extra_schema = {
        "scored_date": "TEXT",
        "expr_mean_tcga_pancan": "REAL",
        "expr_median_tcga_pancan": "REAL",
        "expr_std_tcga_pancan": "REAL",
        "expr_min_tcga_pancan": "REAL",
        "expr_max_tcga_pancan": "REAL",
        "expr_q1_tcga_pancan": "REAL",
        "expr_q3_tcga_pancan": "REAL",
        "expr_iqr_tcga_pancan": "REAL",
        "expr_prevalence_tcga_pancan": "REAL",
        "in_tcga_pancan_expr": "INTEGER",
        "gf_geneformer_like_score": "REAL",
        "gf_regulatory_centrality": "REAL",
        "gf_perturbation_impact": "REAL",
        "gf_disease_shift": "REAL",
        "gf_context_support": "REAL",
        "gf_risk_adjustment": "REAL",
        "gf_model_version": "TEXT",
        "n_pathways": "REAL",
        "max_hub_correlation": "REAL",
        "mean_lit_confidence": "REAL",
        "n_unique_papers": "REAL",
        "conservation_score": "REAL",
        "n_perturbation_studies": "REAL",
        "n_high_conf_perts": "REAL",
        "mean_pert_effect": "REAL",
        "de_consistency": "REAL",
        "mean_abs_log2fc": "REAL",
        "sig_rate": "REAL",
        "mean_clinical_r": "REAL",
        "n_sig_clinical": "REAL",
        "mean_tau": "REAL",
        "log_tpm_disease": "REAL",
        "risk_ubiquitous": "INTEGER",
        "risk_contradictory_lit": "INTEGER",
        "risk_high_isoforms": "INTEGER",
        "curated_contradiction_flag": "INTEGER",
    }

    for col, col_type in extra_schema.items():
        if col not in existing_cols:
            c.execute(f"ALTER TABLE target_scores ADD COLUMN {col} {col_type}")


def save_scores_to_db(
    scored_df: pd.DataFrame,
    db_path="ncrna_platform.db",
    disease_id="DIS_001",
    context_id="CTX_001",
    model_version="v2.3",
    gf_model_version="gf_v0.3",
):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    ensure_target_scores_schema(conn)

    for _, row in scored_df.iterrows():
        nid = row["ncrna_id"]
        exps = recommend_experiments(row)
        flags = build_risk_flags(row)

        top_ev = []
        if row.get("curated_exists", 0):
            top_ev.append(
                f"Curated liver evidence tier score = {row.get('curated_tier_score', 0):.2f}"
            )
        if row.get("mean_abs_log2fc", 0) > 0:
            top_ev.append(f"Mean |log2FC| = {row.get('mean_abs_log2fc', 0):.2f}")
        if row.get("mean_clinical_r", 0) > 0:
            top_ev.append(f"Clinical correlation r = {row.get('mean_clinical_r', 0):.2f}")
        if row.get("n_high_conf_perts", 0) > 0:
            top_ev.append(
                f"{int(row.get('n_high_conf_perts', 0))} high-confidence perturbation studies"
            )
        if row.get("in_tcga_pancan_expr", 0):
            top_ev.append(
                f"TCGA pan-cancer mean expression = {row.get('expr_mean_tcga_pancan', 0):.2f}"
            )
        if row.get("gf_geneformer_like_score", 0) > 0:
            top_ev.append(
                f"Geneformer-like score = {row.get('gf_geneformer_like_score', 0):.2f}"
            )

        if not top_ev:
            top_ev = ["Limited evidence available"]

        insert_cols = [
            "score_id",
            "ncrna_id",
            "disease_id",
            "context_id",
            "relevance_score",
            "specificity_score",
            "mechanism_score",
            "tractability_score",
            "human_evidence_score",
            "risk_score",
            "translational_score",
            "confidence_tier",
            "top_evidence",
            "risk_flags",
            "recommended_experiments",
            "model_version",
            "scored_date",
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
            "gf_geneformer_like_score",
            "gf_regulatory_centrality",
            "gf_perturbation_impact",
            "gf_disease_shift",
            "gf_context_support",
            "gf_risk_adjustment",
            "gf_model_version",
            "n_pathways",
            "max_hub_correlation",
            "mean_lit_confidence",
            "n_unique_papers",
            "conservation_score",
            "n_perturbation_studies",
            "n_high_conf_perts",
            "mean_pert_effect",
            "de_consistency",
            "mean_abs_log2fc",
            "sig_rate",
            "mean_clinical_r",
            "n_sig_clinical",
            "mean_tau",
            "log_tpm_disease",
            "risk_ubiquitous",
            "risk_contradictory_lit",
            "risk_high_isoforms",
            "curated_contradiction_flag",
        ]

        values = (
            f"SCR_{nid}_{disease_id}_{context_id}",
            nid,
            disease_id,
            context_id,
            round(float(row.get("relevance_score", 0)), 4),
            round(float(row.get("specificity_score", 0)), 4),
            round(float(row.get("mechanism_score", 0)), 4),
            round(float(row.get("tractability_score", 0)), 4),
            round(float(row.get("human_evidence_score", 0)), 4),
            round(float(row.get("risk_score", 0)), 4),
            round(float(row.get("translational_score", 0)), 4),
            row.get("confidence_tier", "Tier 3 — Exploratory"),
            json.dumps(top_ev),
            json.dumps(flags),
            json.dumps(exps),
            model_version,
            pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M:%S"),
            round(float(row.get("expr_mean_tcga_pancan", 0)), 6),
            round(float(row.get("expr_median_tcga_pancan", 0)), 6),
            round(float(row.get("expr_std_tcga_pancan", 0)), 6),
            round(float(row.get("expr_min_tcga_pancan", 0)), 6),
            round(float(row.get("expr_max_tcga_pancan", 0)), 6),
            round(float(row.get("expr_q1_tcga_pancan", 0)), 6),
            round(float(row.get("expr_q3_tcga_pancan", 0)), 6),
            round(float(row.get("expr_iqr_tcga_pancan", 0)), 6),
            round(float(row.get("expr_prevalence_tcga_pancan", 0)), 6),
            int(row.get("in_tcga_pancan_expr", 0)),
            round(float(row.get("gf_geneformer_like_score", 0)), 6),
            round(float(row.get("gf_regulatory_centrality", 0)), 6),
            round(float(row.get("gf_perturbation_impact", 0)), 6),
            round(float(row.get("gf_disease_shift", 0)), 6),
            round(float(row.get("gf_context_support", 0)), 6),
            round(float(row.get("gf_risk_adjustment", 0)), 6),
            gf_model_version,
            round(float(row.get("n_pathways", 0)), 6),
            round(float(row.get("max_hub_correlation", 0)), 6),
            round(float(row.get("mean_lit_confidence", 0)), 6),
            round(float(row.get("n_unique_papers", 0)), 6),
            round(float(row.get("conservation_score", 0)), 6),
            round(float(row.get("n_perturbation_studies", 0)), 6),
            round(float(row.get("n_high_conf_perts", 0)), 6),
            round(float(row.get("mean_pert_effect", 0)), 6),
            round(float(row.get("de_consistency", 0)), 6),
            round(float(row.get("mean_abs_log2fc", 0)), 6),
            round(float(row.get("sig_rate", 0)), 6),
            round(float(row.get("mean_clinical_r", 0)), 6),
            round(float(row.get("n_sig_clinical", 0)), 6),
            round(float(row.get("mean_tau", 0)), 6),
            round(float(row.get("log_tpm_disease", 0)), 6),
            int(row.get("risk_ubiquitous", 0)),
            int(row.get("risk_contradictory_lit", 0)),
            int(row.get("risk_high_isoforms", 0)),
            int(row.get("curated_contradiction_flag", 0)),
        )

        sql = f"""
            INSERT OR REPLACE INTO target_scores ({', '.join(insert_cols)})
            VALUES ({', '.join(['?'] * len(insert_cols))})
        """
        c.execute(sql, values)

    conn.commit()
    conn.close()
    print(f"✅ {len(scored_df)} scores saved to database")


def run_scoring_pipeline(
    db_path="ncrna_platform.db",
    disease_id="DIS_001",
    context_id="CTX_001",
):
    print("\n── ncRNA Target Intelligence Platform ──")
    print(f"   Disease: {disease_id}  |  Context: {context_id}\n")

    feat_df = build_feature_matrix(db_path, disease_id, context_id)
    scored_df = compute_translational_scores(feat_df)
    scored_df = compute_geneformer_like_scores(scored_df)

    model_out = train_ranking_model(scored_df)
    save_scores_to_db(scored_df, db_path, disease_id, context_id)

    print("\n── Top 10 Ranked ncRNA Targets ──")
    display_cols = [
        "symbol",
        "translational_score",
        "gf_geneformer_like_score",
        "confidence_tier",
        "curated_tier_score",
        "relevance_score",
        "mechanism_score",
        "human_evidence_score",
    ]
    if "expr_mean_tcga_pancan" in scored_df.columns:
        display_cols.append("expr_mean_tcga_pancan")

    print(scored_df[display_cols].head(10).to_string(index=False))
    return scored_df, model_out


if __name__ == "__main__":
    scored_df, model_out = run_scoring_pipeline()