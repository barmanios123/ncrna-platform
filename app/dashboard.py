"""
Liver ncRNA Translational Engine — Streamlit Dashboard
v2.0 streamlined dashboard with:
- compact ranking table
- baseline vs Geneformer-like rank traceability
- shortlist comparison
- focused target dossier
- collapsible provenance and evidence sections
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
        "de_consistency",
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
        df = pd.read_sql_query("SELECT * FROM vw_target_scores_enriched", conn)
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

    for col in TCGA_COLS + GF_COLS + RAW_PROVENANCE_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    numeric_cols = [
        "translational_score",
        "relevance_score",
        "specificity_score",
        "mechanism_score",
        "tractability_score",
        "human_evidence_score",
        "curated_score",
        "risk_score",
        "gf_geneformer_like_score",
        "gf_regulatory_centrality",
        "gf_perturbation_impact",
        "gf_disease_shift",
        "gf_context_support",
        "gf_risk_adjustment",
    ]
    for col in numeric_cols:
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


def get_curated_rows_for_symbol(curated_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if curated_df.empty or "symbol" not in curated_df.columns:
        return pd.DataFrame()
    return curated_df[curated_df["symbol"].astype(str) == str(symbol)].copy()


def get_curated_tier_label(curated_rows: pd.DataFrame) -> str:
    if curated_rows.empty or "evidence_tier" not in curated_rows.columns:
        return "None"
    tiers = curated_rows["evidence_tier"].dropna().astype(str).unique().tolist()
    tiers = [normalize_tier_label(t) for t in tiers]
    return ", ".join(sorted(tiers)) if tiers else "None"


def count_contradictions(curated_rows: pd.DataFrame) -> int:
    if curated_rows.empty or "is_contradictory" not in curated_rows.columns:
        return 0
    return int(curated_rows["is_contradictory"].fillna(0).astype(float).sum())


def build_component_table(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Component": "Relevance", "Score": safe_float(row.get("relevance_score", 0))},
            {"Component": "Specificity", "Score": safe_float(row.get("specificity_score", 0))},
            {"Component": "Mechanism", "Score": safe_float(row.get("mechanism_score", 0))},
            {"Component": "Tractability", "Score": safe_float(row.get("tractability_score", 0))},
            {"Component": "Human evidence", "Score": safe_float(row.get("human_evidence_score", 0))},
            {"Component": "Curated", "Score": safe_float(row.get("curated_score", 0))},
            {"Component": "Risk-adjusted", "Score": safe_float(row.get("risk_score", 0))},
            {"Component": "Baseline total", "Score": safe_float(row.get("translational_score", 0))},
            {"Component": "GF regulatory centrality", "Score": safe_float(row.get("gf_regulatory_centrality", 0))},
            {"Component": "GF perturbation impact", "Score": safe_float(row.get("gf_perturbation_impact", 0))},
            {"Component": "GF disease shift", "Score": safe_float(row.get("gf_disease_shift", 0))},
            {"Component": "GF context support", "Score": safe_float(row.get("gf_context_support", 0))},
            {"Component": "GF risk adjustment", "Score": safe_float(row.get("gf_risk_adjustment", 0))},
            {"Component": "GF total", "Score": safe_float(row.get("gf_geneformer_like_score", 0))},
        ]
    )


def build_shortlist_table(compare_df: pd.DataFrame, curated_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in compare_df.iterrows():
        curated_rows = get_curated_rows_for_symbol(curated_df, row["symbol"])
        rows.append(
            {
                "Target": row["symbol"],
                "Tier": row["confidence_tier_norm"],
                "Base rank": safe_int(row["Baseline rank"]),
                "GF rank": safe_int(row["Geneformer rank"]),
                "Δ": row["Delta rank display"],
                "Baseline": round(safe_float(row["translational_score"]), 3),
                "GF": round(safe_float(row["gf_geneformer_like_score"]), 3),
                "Top driver": max(
                    [
                        ("Disease shift", safe_float(row.get("gf_disease_shift", 0))),
                        ("Reg centrality", safe_float(row.get("gf_regulatory_centrality", 0))),
                        ("Perturbation", safe_float(row.get("gf_perturbation_impact", 0))),
                        ("Context", safe_float(row.get("gf_context_support", 0))),
                    ],
                    key=lambda x: x[1],
                )[0],
                "Curated tier": get_curated_tier_label(curated_rows),
                "Contradictions": count_contradictions(curated_rows),
            }
        )
    return pd.DataFrame(rows).sort_values(["GF", "Baseline"], ascending=False).reset_index(drop=True)


def main():
    st.set_page_config(
        page_title="Liver ncRNA Translational Engine",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧬 Liver ncRNA Translational Engine")
    st.caption("Compact ranking + traceability for MASLD/MASH liver ncRNA targets")

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
    sort_choice = st.sidebar.selectbox("Sort by", options=available_sort_options, index=1 if len(available_sort_options) > 1 else 0)
    sort_col = sort_label_to_col[sort_choice]

    disease_id_opts = sorted(scores_df["disease_id"].dropna().astype(str).unique().tolist())
    ctx_id_opts = sorted(scores_df["context_id"].dropna().astype(str).unique().tolist())

    if not disease_id_opts or not ctx_id_opts:
        st.error("Missing disease_id or context_id values in target_scores.")
        return

    disease_display_map = {x: display_disease_label(x) for x in disease_id_opts}
    selected_disease_label = st.sidebar.selectbox(
        "Disease context",
        options=[disease_display_map[x] for x in disease_id_opts],
        index=0,
    )
    disease_sel = next(k for k, v in disease_display_map.items() if v == selected_disease_label)

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

    filtered_df["Baseline rank"] = (
        filtered_df["translational_score"].rank(method="dense", ascending=False).astype(int)
    )
    filtered_df["Geneformer rank"] = (
        filtered_df["gf_geneformer_like_score"]
        .fillna(filtered_df["translational_score"])
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    filtered_df["Delta rank"] = filtered_df["Baseline rank"] - filtered_df["Geneformer rank"]
    filtered_df["Delta rank display"] = filtered_df["Delta rank"].apply(delta_rank_display)
    filtered_df["Tier badge"] = filtered_df["confidence_tier_norm"].apply(style_confidence_tier)

    filtered_df = (
        filtered_df.sort_values([sort_col, "translational_score"], ascending=False)
        .drop_duplicates(subset=["symbol"], keep="first")
        .copy()
    )

    top_row = filtered_df.iloc[0]
    positive_delta_n = int((filtered_df["Delta rank"] > 0).sum())
    curated_n = int(filtered_df["has_curated_evidence"].sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Top target", str(top_row["symbol"]))
    m2.metric("Top GF score", f"{safe_float(top_row['gf_geneformer_like_score']):.3f}")
    m3.metric("Rank improvers", positive_delta_n)
    m4.metric("Curated-supported", curated_n)

    st.caption(f"Disease: {selected_disease_label} | Context: {selected_context_label} | Sorted by: {sort_choice}")

    st.subheader("Ranked targets")
    rank_table = filtered_df[
        [
            "symbol",
            "Tier badge",
            "confidence_tier_norm",
            "Baseline rank",
            "Geneformer rank",
            "Delta rank display",
            "translational_score",
            "gf_geneformer_like_score",
        ]
    ].copy()
    rank_table.columns = [
        "Target",
        "",
        "Confidence tier",
        "Base rank",
        "GF rank",
        "Δ rank",
        "Baseline score",
        "GF score",
    ]
    rank_table["Baseline score"] = rank_table["Baseline score"].apply(lambda x: round(safe_float(x), 3))
    rank_table["GF score"] = rank_table["GF score"].apply(lambda x: round(safe_float(x), 3))
    st.dataframe(rank_table, width="stretch", hide_index=True, height=350)

    ranking_csv = rank_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download ranking CSV",
        data=ranking_csv,
        file_name="filtered_target_ranking.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("Shortlist comparison")

    compare_options = filtered_df["symbol"].astype(str).tolist()
    default_compare = compare_options[:3] if len(compare_options) >= 3 else compare_options
    selected_compare = st.multiselect(
        "Select targets",
        options=compare_options,
        default=default_compare,
    )

    if selected_compare:
        compare_df = filtered_df[filtered_df["symbol"].astype(str).isin(selected_compare)].copy()
        shortlist_table = build_shortlist_table(compare_df, curated_df)
        st.dataframe(shortlist_table, width="stretch", hide_index=True)

        shortlist_csv = shortlist_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download shortlist CSV",
            data=shortlist_csv,
            file_name="shortlist_comparison.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.subheader("Target dossier")

    target_list = filtered_df["symbol"].astype(str).tolist()
    selected_symbol = st.selectbox("Select target", options=target_list, index=0)
    selected_rows = filtered_df[filtered_df["symbol"].astype(str) == selected_symbol].copy()

    if selected_rows.empty:
        st.warning("Selected target is not available under current filters.")
        return

    row = selected_rows.iloc[0]
    curated_rows = get_curated_rows_for_symbol(curated_df, selected_symbol)

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Base rank", safe_int(row["Baseline rank"]))
    a2.metric("GF rank", safe_int(row["Geneformer rank"]))
    a3.metric("Δ rank", row["Delta rank display"])
    a4.metric("Curated contradictions", count_contradictions(curated_rows))

    left, right = st.columns([1.1, 1.4])

    with left:
        st.markdown(f"### {selected_symbol}")
        st.write(f"**ncRNA ID:** {row.get('ncrna_id', 'NA')}")
        st.write(f"**Confidence tier:** {row.get('confidence_tier_norm', 'NA')}")
        st.write(f"**Biotype:** {row.get('biotype', 'NA')}")
        st.write(f"**Curated tier(s):** {get_curated_tier_label(curated_rows)}")
        st.write(f"**Conservation score:** {safe_float(row.get('conservation_score', 0)):.3f}")
        st.write(f"**Transcript count:** {safe_int(row.get('transcript_count', 0))}")

        st.markdown("#### Evidence summary")
        render_clean_list(row["top_evidence_list"], "No top evidence statements available.")

        st.markdown("#### Risks")
        render_clean_list(row["risk_flags_list"], "No major risk flags identified.")

        st.markdown("#### Recommended experiments")
        render_clean_list(row["recommended_experiments_list"], "No specific experiments recommended.")

    with right:
        st.markdown("#### Component score table")
        component_df = build_component_table(row)
        st.dataframe(component_df.style.format({"Score": "{:.4f}"}), width="stretch", hide_index=True, height=520)

    with st.expander("Geneformer-like raw provenance", expanded=False):
        for component_name, component_features in FEATURE_MAPPING.items():
            feature_rows = [
                {"Feature": feat, "Value": "NA" if pd.isna(row.get(feat, pd.NA)) else format(round(float(row.get(feat)), 4), ".4f")}
                if pd.notna(row.get(feat, pd.NA))
                else {"Feature": feat, "Value": "NA"}
                for feat in component_features
            ]
            st.markdown(f"**{component_name}**")
            st.dataframe(pd.DataFrame(feature_rows), width="stretch", hide_index=True)

    with st.expander("Expression context", expanded=False):
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("TCGA mean", f"{safe_float(row.get('expr_mean_tcga_pancan', 0)):.3f}")
        e2.metric("TCGA median", f"{safe_float(row.get('expr_median_tcga_pancan', 0)):.3f}")
        e3.metric("TCGA max", f"{safe_float(row.get('expr_max_tcga_pancan', 0)):.3f}")
        e4.metric("TCGA prevalence", f"{safe_float(row.get('expr_prevalence_tcga_pancan', 0)):.3f}")
        e5.metric("In TCGA", safe_int(row.get("in_tcga_pancan_expr", 0)))

    with st.expander("Methodology and debug", expanded=False):
        st.markdown(
            """
            - Baseline ranking uses translational score.
            - Geneformer-like ranking uses persisted `gf_geneformer_like_score`.
            - Delta rank is baseline rank minus GF rank; positive values mean the target rises under GF scoring.
            - Raw provenance fields are pulled from the enriched view or directly from `target_scores`.
            """
        )
        present_cols = [c for c in RAW_PROVENANCE_COLS if c in scores_df.columns]
        missing_cols = [c for c in RAW_PROVENANCE_COLS if c not in scores_df.columns]

        d1, d2 = st.columns(2)
        with d1:
            st.write("**Present provenance fields**")
            st.code("\n".join(present_cols) if present_cols else "None", language="text")
        with d2:
            st.write("**Missing provenance fields**")
            st.code("\n".join(missing_cols) if missing_cols else "None", language="text")


if __name__ == "__main__":
    main()