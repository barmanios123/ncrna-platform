"""
Liver ncRNA Translational Engine — Streamlit Dashboard
v1.5 schema-aware dashboard with:
- baseline + Geneformer-like ranking
- dossier and shortlist comparison
- methodology explainer
- raw provenance display with graceful handling of missing schema fields
- debug coverage reporting
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "ncrna_platform.db"

TCGA_COLS = [
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

GF_COLS = [
    "gf_geneformer_like_score",
    "gf_regulatory_centrality",
    "gf_perturbation_impact",
    "gf_disease_shift",
    "gf_context_support",
    "gf_risk_adjustment",
]

RAW_PROVENANCE_COLS = [
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
    "expr_prevalence_tcga_pancan",
    "in_tcga_pancan_expr",
    "risk_ubiquitous",
    "risk_contradictory_lit",
    "risk_high_isoforms",
    "curated_contradiction_flag",
]

FEATURE_MAPPING = {
    "Regulatory centrality": [
        "n_pathways",
        "max_hub_correlation",
        "mean_lit_confidence",
        "n_unique_papers",
        "conservation_score",
    ],
    "Perturbation impact": [
        "n_perturbation_studies",
        "n_high_conf_perts",
        "mean_pert_effect",
    ],
    "Disease shift": [
        "de_consistency",
        "mean_abs_log2fc",
        "sig_rate",
        "mean_clinical_r",
        "n_sig_clinical",
    ],
    "Context support": [
        "mean_tau",
        "log_tpm_disease",
        "expr_prevalence_tcga_pancan",
        "in_tcga_pancan_expr",
    ],
    "Risk adjustment": [
        "risk_ubiquitous",
        "risk_contradictory_lit",
        "risk_high_isoforms",
        "curated_contradiction_flag",
    ],
}

DISEASE_LABELS = {
    "DIS_001": "MASLD/MASH → fibrosis/HCC",
    "masld_mash_fibrosis_hcc": "MASLD/MASH → fibrosis/HCC",
    "MASLD_MASH_FIBROSIS_HCC": "MASLD/MASH → fibrosis/HCC",
    "masld_mash_hcc": "MASLD/MASH → fibrosis/HCC",
    "liver_masld_hcc": "MASLD/MASH → fibrosis/HCC",
}

CONTEXT_LABELS = {
    "CTX_001": "Liver tissue / hepatocyte cell context",
    "liver_hepatocyte": "Liver tissue / hepatocyte cell context",
    "LIVER_HEPATOCYTE": "Liver tissue / hepatocyte cell context",
    "liver_tissue_hepatocyte": "Liver tissue / hepatocyte cell context",
    "hepatocyte": "Liver tissue / hepatocyte cell context",
}


@st.cache_data(show_spinner=False)
def load_scores(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)

    try:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM vw_target_scores_enriched
            """,
            conn,
        )
    except Exception:
        df = pd.read_sql_query(
            """
            SELECT
                ts.*,
                nm.symbol,
                nm.biotype,
                nm.conservation_score AS nm_conservation_score,
                nm.transcript_count
            FROM target_scores ts
            LEFT JOIN ncrna_master nm
                ON ts.ncrna_id = nm.ncrna_id
            """,
            conn,
        )

    conn.close()

    if "conservation_score" not in df.columns and "nm_conservation_score" in df.columns:
        df["conservation_score"] = df["nm_conservation_score"]
    elif "conservation_score" in df.columns and "nm_conservation_score" in df.columns:
        df["conservation_score"] = df["conservation_score"].fillna(df["nm_conservation_score"])

    def safe_json_list(value):
        if isinstance(value, str) and value.strip().startswith("["):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return []

    for col in ["top_evidence", "risk_flags", "recommended_experiments"]:
        if col not in df.columns:
            df[col] = "[]"

    if "symbol" not in df.columns:
        df["symbol"] = df["ncrna_id"].astype(str)

    df["symbol"] = df["symbol"].fillna(df["ncrna_id"].astype(str)).astype(str)
    df["top_evidence_list"] = df["top_evidence"].apply(safe_json_list)
    df["risk_flags_list"] = df["risk_flags"].apply(safe_json_list)
    df["recommended_experiments_list"] = df["recommended_experiments"].apply(safe_json_list)

    for col in TCGA_COLS + GF_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    return df


@st.cache_data(show_spinner=False)
def load_curated(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM curated_targets", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def normalize_tier_label(value: str) -> str:
    v = str(value).strip()
    mapping = {
        "Tier1": "Tier 1 — High Confidence",
        "Tier 1": "Tier 1 — High Confidence",
        "Tier2": "Tier 2 — Moderate Confidence",
        "Tier 2": "Tier 2 — Moderate Confidence",
        "Tier3": "Tier 3 — Exploratory",
        "Tier 3": "Tier 3 — Exploratory",
    }
    return mapping.get(v, v if v else "Unspecified")


def style_confidence_tier(tier: str) -> str:
    tier = normalize_tier_label(tier)
    if "Tier 1" in tier:
        return "🟢"
    if "Tier 2" in tier:
        return "🟡"
    return "⚪️"


def clean_bullet_text(text) -> str:
    s = str(text).strip()
    for prefix in ["•", "-", "*"]:
        if s.startswith(prefix):
            s = s[1:].strip()
    return s


def render_clean_list(items, empty_message: str):
    clean_items = [clean_bullet_text(x) for x in items if str(x).strip()]
    if clean_items:
        for item in clean_items:
            st.write(f"• {item}")
    else:
        st.caption(empty_message)


def get_curated_tier_label(curated_rows: pd.DataFrame) -> str:
    if curated_rows.empty or "evidence_tier" not in curated_rows.columns:
        return "None"

    tiers = curated_rows["evidence_tier"].dropna().astype(str).unique().tolist()
    tiers = [normalize_tier_label(t) for t in tiers]
    return ", ".join(sorted(tiers)) if tiers else "None"


def make_quality_summary(row, curated_rows: pd.DataFrame) -> dict:
    top_evidence_n = len(row.get("top_evidence_list", []))
    risk_flags_n = len(row.get("risk_flags_list", []))
    recommended_experiments_n = len(row.get("recommended_experiments_list", []))
    curated_n = len(curated_rows) if not curated_rows.empty else 0

    contradictory_n = 0
    if not curated_rows.empty and "is_contradictory" in curated_rows.columns:
        contradictory_n = int(curated_rows["is_contradictory"].fillna(0).astype(float).sum())

    return {
        "Evidence statements": top_evidence_n,
        "Risk flags": risk_flags_n,
        "Recommended experiments": recommended_experiments_n,
        "Curated records": curated_n,
        "Contradictory records": contradictory_n,
    }


def display_disease_label(value: str) -> str:
    return DISEASE_LABELS.get(str(value), str(value))


def display_context_label(value: str) -> str:
    return CONTEXT_LABELS.get(str(value), str(value))


def delta_rank_display(x: int) -> str:
    if x > 0:
        return f"↑ {x}"
    if x < 0:
        return f"↓ {abs(x)}"
    return "–"


def rank_change_summary(row) -> str:
    delta = safe_int(row.get("Delta rank", 0))
    if delta > 0:
        return (
            f"{row.get('symbol', 'Target')} improved from baseline rank "
            f"{safe_int(row.get('Baseline rank', 0))} to Geneformer rank "
            f"{safe_int(row.get('Geneformer rank', 0))}."
        )
    if delta < 0:
        return (
            f"{row.get('symbol', 'Target')} fell from baseline rank "
            f"{safe_int(row.get('Baseline rank', 0))} to Geneformer rank "
            f"{safe_int(row.get('Geneformer rank', 0))}."
        )
    return (
        f"{row.get('symbol', 'Target')} stayed at the same position under both "
        f"baseline and Geneformer-like ranking."
    )


def format_feature_value(val):
    if pd.isna(val):
        return "NA"
    try:
        return round(float(val), 4)
    except Exception:
        return str(val)


def main():
    st.set_page_config(
        page_title="Liver ncRNA Translational Engine",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧬 Liver ncRNA Translational Engine")
    st.caption("MASLD/MASH → fibrosis/HCC · ncRNA target ranking with curated liver evidence")

    if not DB_PATH.exists():
        st.error("Database not found. Run the scoring pipeline first.")
        return

    scores_df = load_scores(DB_PATH)
    curated_df = load_curated(DB_PATH)

    if scores_df.empty:
        st.warning("No rows found in target_scores. Run the scoring pipeline first.")
        return

    if "confidence_tier" not in scores_df.columns:
        st.error("Missing confidence_tier in target_scores.")
        return

    scores_df["confidence_tier_norm"] = scores_df["confidence_tier"].apply(normalize_tier_label)

    with st.expander("How scoring works and how to interpret this dashboard", expanded=False):
        st.markdown(
            """
            This dashboard ranks ncRNA targets using a baseline translational score and an augmented Geneformer-like score.

            **Important note:** the current database schema contains only a subset of the raw engineered provenance features needed for full component-level explainability, so some provenance panels may be partial.
            """
        )

    with st.expander("Debug: dataframe columns and provenance coverage", expanded=False):
        st.write("**scores_df column count:**", len(scores_df.columns))
        st.code("\n".join(sorted(scores_df.columns.tolist())), language="text")

        present_cols = [c for c in RAW_PROVENANCE_COLS if c in scores_df.columns]
        missing_cols = [c for c in RAW_PROVENANCE_COLS if c not in scores_df.columns]

        d1, d2 = st.columns(2)
        with d1:
            st.write("**Raw provenance columns present**")
            st.code("\n".join(present_cols) if present_cols else "None", language="text")
        with d2:
            st.write("**Raw provenance columns missing**")
            st.code("\n".join(missing_cols) if missing_cols else "None", language="text")

    st.sidebar.header("Filters")

    tier_options = sorted(scores_df["confidence_tier_norm"].dropna().astype(str).unique().tolist())
    tier_filter = st.sidebar.multiselect("Confidence tier", options=tier_options, default=tier_options)
    curated_only = st.sidebar.checkbox("Curated liver targets only", value=False)

    available_sort_options = ["Translational score (baseline)"]
    if "gf_geneformer_like_score" in scores_df.columns and scores_df["gf_geneformer_like_score"].notna().any():
        available_sort_options.append("Geneformer-like score")

    sort_label_to_col = {
        "Translational score (baseline)": "translational_score",
        "Geneformer-like score": "gf_geneformer_like_score",
    }
    sort_choice = st.sidebar.selectbox("Sort ncRNA candidates by", options=available_sort_options, index=0)
    sort_col = sort_label_to_col[sort_choice]

    disease_id_opts = sorted(scores_df["disease_id"].dropna().astype(str).unique().tolist())
    if not disease_id_opts:
        st.error("No disease_id values found in target_scores.")
        return

    disease_display_map = {x: display_disease_label(x) for x in disease_id_opts}
    selected_disease_label = st.sidebar.selectbox(
        "Disease context",
        options=[disease_display_map[x] for x in disease_id_opts],
        index=0,
    )
    disease_sel = next(k for k, v in disease_display_map.items() if v == selected_disease_label)

    ctx_id_opts = sorted(scores_df["context_id"].dropna().astype(str).unique().tolist())
    if not ctx_id_opts:
        st.error("No context_id values found in target_scores.")
        return

    context_display_map = {x: display_context_label(x) for x in ctx_id_opts}
    selected_context_label = st.sidebar.selectbox(
        "Tissue/cell context",
        options=[context_display_map[x] for x in ctx_id_opts],
        index=0,
    )
    ctx_sel = next(k for k, v in context_display_map.items() if v == selected_context_label)

    filtered_df = scores_df[
        (scores_df["confidence_tier_norm"].astype(str).isin(tier_filter))
        & (scores_df["disease_id"].astype(str) == disease_sel)
        & (scores_df["context_id"].astype(str) == ctx_sel)
    ].copy()

    if filtered_df.empty:
        st.info("No targets match the current filters.")
        return

    filtered_df["has_curated_evidence"] = (
        filtered_df["top_evidence"].fillna("").astype(str).str.contains("Curated liver evidence", case=False)
    )

    if curated_only:
        filtered_df = filtered_df[filtered_df["has_curated_evidence"]].copy()

    if filtered_df.empty:
        st.info("No targets match the current filters.")
        return

    if "gf_geneformer_like_score" in filtered_df.columns:
        filtered_df["_gf_present"] = filtered_df["gf_geneformer_like_score"].notna().astype(int)
    else:
        filtered_df["_gf_present"] = 0

    filtered_df = (
        filtered_df.sort_values(by=["_gf_present", "translational_score"], ascending=[False, False])
        .drop_duplicates(subset=["ncrna_id", "disease_id", "context_id"], keep="first")
        .copy()
    )

    st.caption(f"Disease context: {selected_disease_label}")
    st.caption(f"Tissue/cell context: {selected_context_label}")
    st.caption(f"Currently ranked by: {sort_choice}")

    filtered_df["tier_icon"] = filtered_df["confidence_tier_norm"].apply(style_confidence_tier)
    filtered_df["Translational score"] = filtered_df["translational_score"].apply(lambda x: round(safe_float(x), 3))
    filtered_df["Relevance"] = filtered_df["relevance_score"].apply(lambda x: round(safe_float(x), 3))
    filtered_df["Mechanism"] = filtered_df["mechanism_score"].apply(lambda x: round(safe_float(x), 3))
    filtered_df["Human evidence"] = filtered_df["human_evidence_score"].apply(lambda x: round(safe_float(x), 3))
    filtered_df["Geneformer-like score"] = filtered_df["gf_geneformer_like_score"].apply(
        lambda x: round(safe_float(x), 3)
    )

    filtered_df["Baseline rank"] = (
        filtered_df["translational_score"].rank(method="dense", ascending=False).astype(int)
    )
    filtered_df["Geneformer rank"] = (
        filtered_df["gf_geneformer_like_score"].rank(method="dense", ascending=False).astype(int)
    )
    filtered_df["Delta rank"] = filtered_df["Baseline rank"] - filtered_df["Geneformer rank"]
    filtered_df["Delta rank display"] = filtered_df["Delta rank"].apply(delta_rank_display)

    if "expr_mean_tcga_pancan" in filtered_df.columns:
        filtered_df["TCGA mean"] = filtered_df["expr_mean_tcga_pancan"].apply(lambda x: round(safe_float(x), 3))
    else:
        filtered_df["TCGA mean"] = 0.0

    ranked_cols = [
        "symbol",
        "tier_icon",
        "confidence_tier_norm",
        "Baseline rank",
        "Geneformer rank",
        "Delta rank display",
        "Translational score",
        "Geneformer-like score",
        "Relevance",
        "Mechanism",
        "Human evidence",
    ]
    if "expr_mean_tcga_pancan" in filtered_df.columns:
        ranked_cols.append("TCGA mean")

    sort_display_col = "Geneformer-like score" if sort_col == "gf_geneformer_like_score" else "Translational score"

    ranked_display = (
        filtered_df[ranked_cols]
        .rename(
            columns={
                "tier_icon": "",
                "confidence_tier_norm": "Confidence tier",
                "Delta rank display": "Δ rank",
            }
        )
        .sort_values(sort_display_col, ascending=False)
        .drop_duplicates(subset=["symbol"], keep="first")
    )

    st.subheader("Ranked ncRNA targets")
    st.dataframe(ranked_display.set_index("symbol"), width="stretch", height=360)

    csv_bytes = ranked_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered ranking as CSV",
        data=csv_bytes,
        file_name="filtered_target_ranking.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("Target dossier")

    target_list = ranked_display["symbol"].dropna().astype(str).tolist()
    if not target_list:
        st.info("No selectable targets available.")
        return

    selected_symbol = st.selectbox("Select target", options=target_list, index=0)
    selected_rows = filtered_df[filtered_df["symbol"].astype(str) == str(selected_symbol)].copy()
    if selected_rows.empty:
        st.warning("Selected target is not available under the current filters.")
        return

    selected_rows = (
        selected_rows.sort_values(by=["_gf_present", sort_col, "translational_score"], ascending=[False, False, False])
        .drop_duplicates(subset=["ncrna_id", "disease_id", "context_id"], keep="first")
        .copy()
    )
    row = selected_rows.iloc[0]

    if not curated_df.empty and "symbol" in curated_df.columns:
        curated_rows = curated_df[curated_df["symbol"].astype(str) == str(selected_symbol)].copy()
    else:
        curated_rows = pd.DataFrame()

    quality = make_quality_summary(row, curated_rows)
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Evidence statements", quality["Evidence statements"])
    q2.metric("Risk flags", quality["Risk flags"])
    q3.metric("Recommended expts", quality["Recommended experiments"])
    q4.metric("Curated records", quality["Curated records"])
    q5.metric("Contradictions", quality["Contradictory records"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"### {selected_symbol}")
        st.write(f"**ncRNA ID:** {row.get('ncrna_id', 'NA')}")
        st.write(f"**Confidence tier:** {row.get('confidence_tier_norm', 'NA')}")
        st.write(f"**Biotype:** {row.get('biotype', 'NA')}")
        st.write(f"**Conservation score:** {safe_float(row.get('conservation_score', 0)):.3f}")
        st.write(f"**Transcript count:** {safe_int(row.get('transcript_count', 0))}")
        st.write(f"**Baseline rank:** {safe_int(row.get('Baseline rank', 0))}")
        st.write(f"**Geneformer rank:** {safe_int(row.get('Geneformer rank', 0))}")
        st.write(f"**Δ rank:** {row.get('Delta rank display', '–')}")

    with c2:
        st.markdown("**Score breakdown**")
        st.write(f"Translational score: {safe_float(row.get('translational_score', 0)):.3f}")
        st.write(f"Geneformer-like score: {safe_float(row.get('gf_geneformer_like_score', 0)):.3f}")
        st.write(f"Relevance: {safe_float(row.get('relevance_score', 0)):.3f}")
        st.write(f"Specificity: {safe_float(row.get('specificity_score', 0)):.3f}")
        st.write(f"Mechanism: {safe_float(row.get('mechanism_score', 0)):.3f}")
        st.write(f"Tractability: {safe_float(row.get('tractability_score', 0)):.3f}")
        st.write(f"Human evidence: {safe_float(row.get('human_evidence_score', 0)):.3f}")
        st.write(f"Risk score: {safe_float(row.get('risk_score', 0)):.3f}")

    with c3:
        st.markdown("**Curated evidence**")
        st.write(f"Curated tier(s): {get_curated_tier_label(curated_rows)}")
        if curated_rows.empty:
            st.caption("No curated liver-disease entry linked.")

    st.markdown("#### Why this rank changed")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Baseline rank", safe_int(row.get("Baseline rank", 0)))
    r2.metric("Geneformer rank", safe_int(row.get("Geneformer rank", 0)))
    r3.metric("Delta rank", safe_int(row.get("Delta rank", 0)))
    r4.metric("GF score", f"{safe_float(row.get('gf_geneformer_like_score', 0)):.4f}")
    st.caption(rank_change_summary(row))

    trace_component_rows = [
        ("Regulatory centrality", safe_float(row.get("gf_regulatory_centrality", 0))),
        ("Perturbation impact", safe_float(row.get("gf_perturbation_impact", 0))),
        ("Disease shift", safe_float(row.get("gf_disease_shift", 0))),
        ("Context support", safe_float(row.get("gf_context_support", 0))),
        ("Risk adjustment", safe_float(row.get("gf_risk_adjustment", 0))),
    ]
    trace_df = pd.DataFrame(trace_component_rows, columns=["Component", "Score"])
    st.dataframe(trace_df.style.format({"Score": "{:.4f}"}), width="stretch", hide_index=True)

    st.markdown("#### Feature evidence behind Geneformer-like components")

    available_global = [c for c in RAW_PROVENANCE_COLS if c in scores_df.columns]
    missing_global = [c for c in RAW_PROVENANCE_COLS if c not in scores_df.columns]

    if missing_global:
        st.warning(
            "Partial provenance only: the current database/view does not contain all engineered raw "
            "feature columns needed for full Geneformer-like explainability."
        )

    s1, s2 = st.columns(2)
    with s1:
        st.write("**Available provenance fields in this app session**")
        st.code("\n".join(available_global) if available_global else "None", language="text")
    with s2:
        st.write("**Missing provenance fields in this app session**")
        st.code("\n".join(missing_global) if missing_global else "None", language="text")

    for component_name, component_features in FEATURE_MAPPING.items():
        available_features = [f for f in component_features if f in row.index]
        missing_features = [f for f in component_features if f not in row.index]

        with st.expander(f"{component_name} — raw contributing features", expanded=False):
            if available_features:
                feature_rows = [
                    {"Feature": feat, "Value": format_feature_value(row.get(feat, pd.NA))}
                    for feat in available_features
                ]
                st.dataframe(pd.DataFrame(feature_rows), width="stretch", hide_index=True)
            else:
                st.caption("No raw feature columns available for this component in the current dataframe.")

            if missing_features:
                st.caption("Missing fields for this component:")
                st.code("\n".join(missing_features), language="text")

    st.markdown("#### TCGA pan-cancer expression")
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("TCGA mean", f"{safe_float(row.get('expr_mean_tcga_pancan', 0)):.3f}")
    t2.metric("TCGA median", f"{safe_float(row.get('expr_median_tcga_pancan', 0)):.3f}")
    t3.metric("TCGA max", f"{safe_float(row.get('expr_max_tcga_pancan', 0)):.3f}")
    t4.metric("TCGA prevalence", f"{safe_float(row.get('expr_prevalence_tcga_pancan', 0)):.3f}")
    t5.metric("In TCGA", safe_int(row.get("in_tcga_pancan_expr", 0)))

    st.markdown("#### Top evidence")
    render_clean_list(row["top_evidence_list"], "No top evidence statements available.")

    st.markdown("#### Risk flags")
    render_clean_list(row["risk_flags_list"], "No major risk flags identified.")

    st.markdown("#### Recommended experiments")
    render_clean_list(row["recommended_experiments_list"], "No specific experiments recommended.")


if __name__ == "__main__":
    main()