"""
Liver ncRNA Translational Engine — Streamlit Dashboard
v0.3 with TCGA display support and human-readable context labels
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
    df = pd.read_sql_query(
        """
        SELECT
            ts.*,
            nm.symbol,
            nm.biotype,
            nm.conservation_score,
            nm.transcript_count
        FROM target_scores ts
        LEFT JOIN ncrna_master nm
            ON ts.ncrna_id = nm.ncrna_id
        """,
        conn,
    )
    conn.close()

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

    for col in TCGA_COLS:
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

    scores_df["confidence_tier_norm"] = scores_df["confidence_tier"].apply(normalize_tier_label)

    st.sidebar.header("Filters")

    tier_options = sorted(scores_df["confidence_tier_norm"].dropna().astype(str).unique().tolist())
    tier_filter = st.sidebar.multiselect(
        "Confidence tier",
        options=tier_options,
        default=tier_options,
    )

    curated_only = st.sidebar.checkbox("Curated liver targets only", value=False)

    disease_id_opts = sorted(scores_df["disease_id"].dropna().astype(str).unique().tolist())
    if not disease_id_opts:
        st.error("No disease_id values found in target_scores.")
        return

    disease_display_map = {x: display_disease_label(x) for x in disease_id_opts}
    disease_display_opts = [disease_display_map[x] for x in disease_id_opts]
    selected_disease_label = st.sidebar.selectbox(
        "Disease context",
        options=disease_display_opts,
        index=0,
    )
    disease_sel = next(k for k, v in disease_display_map.items() if v == selected_disease_label)

    ctx_id_opts = sorted(scores_df["context_id"].dropna().astype(str).unique().tolist())
    if not ctx_id_opts:
        st.error("No context_id values found in target_scores.")
        return

    context_display_map = {x: display_context_label(x) for x in ctx_id_opts}
    context_display_opts = [context_display_map[x] for x in ctx_id_opts]
    selected_context_label = st.sidebar.selectbox(
        "Tissue/cell context",
        options=context_display_opts,
        index=0,
    )
    ctx_sel = next(k for k, v in context_display_map.items() if v == selected_context_label)

    filtered_df = scores_df[
        (scores_df["confidence_tier_norm"].astype(str).isin(tier_filter))
        & (scores_df["disease_id"].astype(str) == disease_sel)
        & (scores_df["context_id"].astype(str) == ctx_sel)
    ].copy()

    filtered_df["has_curated_evidence"] = (
        filtered_df["top_evidence"].fillna("").astype(str).str.contains(
            "Curated liver evidence", case=False
        )
    )

    if curated_only:
        filtered_df = filtered_df[filtered_df["has_curated_evidence"]].copy()

    if filtered_df.empty:
        st.info("No targets match the current filters.")
        return

    st.caption(f"Disease context: {selected_disease_label}")
    st.caption(f"Tissue/cell context: {selected_context_label}")

    filtered_df["tier_icon"] = filtered_df["confidence_tier_norm"].apply(style_confidence_tier)
    filtered_df["Translational score"] = filtered_df["translational_score"].apply(
        lambda x: round(safe_float(x), 3)
    )
    filtered_df["Relevance"] = filtered_df["relevance_score"].apply(
        lambda x: round(safe_float(x), 3)
    )
    filtered_df["Mechanism"] = filtered_df["mechanism_score"].apply(
        lambda x: round(safe_float(x), 3)
    )
    filtered_df["Human evidence"] = filtered_df["human_evidence_score"].apply(
        lambda x: round(safe_float(x), 3)
    )

    if "expr_mean_tcga_pancan" in filtered_df.columns:
        filtered_df["TCGA mean"] = filtered_df["expr_mean_tcga_pancan"].apply(
            lambda x: round(safe_float(x), 3)
        )
    else:
        filtered_df["TCGA mean"] = 0.0

    ranked_cols = [
        "symbol",
        "tier_icon",
        "confidence_tier_norm",
        "Translational score",
        "Relevance",
        "Mechanism",
        "Human evidence",
    ]

    if "expr_mean_tcga_pancan" in filtered_df.columns:
        ranked_cols.append("TCGA mean")

    ranked_display = (
        filtered_df[ranked_cols]
        .rename(columns={"tier_icon": "", "confidence_tier_norm": "Confidence tier"})
        .sort_values("Translational score", ascending=False)
        .drop_duplicates(subset=["symbol"])
    )

    st.subheader("Ranked ncRNA targets")
    st.dataframe(
        ranked_display.set_index("symbol"),
        width="stretch",
        height=360,
    )

    st.subheader("TCGA Pan-Cancer Expression")
    chart_df = filtered_df.copy()
    needed_cols = {"symbol", "expr_mean_tcga_pancan"}

    if needed_cols.issubset(chart_df.columns):
        chart_df["expr_mean_tcga_pancan"] = pd.to_numeric(
            chart_df["expr_mean_tcga_pancan"], errors="coerce"
        )
        chart_df = chart_df.dropna(subset=["expr_mean_tcga_pancan"])
        chart_df = chart_df[chart_df["expr_mean_tcga_pancan"] > 0]
        chart_df = (
            chart_df[["symbol", "expr_mean_tcga_pancan"]]
            .drop_duplicates(subset=["symbol"])
            .sort_values("expr_mean_tcga_pancan", ascending=False)
        )

        if not chart_df.empty:
            st.bar_chart(chart_df.set_index("symbol")["expr_mean_tcga_pancan"])
        else:
            st.info("No non-zero TCGA pan-cancer expression values available to plot.")
    else:
        st.info("TCGA pan-cancer expression data not available for plotting.")

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

    row = selected_rows.sort_values("translational_score", ascending=False).iloc[0]

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

    with c2:
        st.markdown("**Score breakdown**")
        st.write(f"Translational score: {safe_float(row.get('translational_score', 0)):.3f}")
        st.write(f"Relevance: {safe_float(row.get('relevance_score', 0)):.3f}")
        st.write(f"Specificity: {safe_float(row.get('specificity_score', 0)):.3f}")
        st.write(f"Mechanism: {safe_float(row.get('mechanism_score', 0)):.3f}")
        st.write(f"Tractability: {safe_float(row.get('tractability_score', 0)):.3f}")
        st.write(f"Human evidence: {safe_float(row.get('human_evidence_score', 0)):.3f}")
        st.write(f"Risk score: {safe_float(row.get('risk_score', 0)):.3f}")

    with c3:
        st.markdown("**Curated evidence**")
        st.write(f"Curated tier(s): {get_curated_tier_label(curated_rows)}")

        if not curated_rows.empty:
            if "disease_stage" in curated_rows.columns:
                stages = sorted(curated_rows["disease_stage"].dropna().astype(str).unique().tolist())
                st.write("Disease stage(s): " + (", ".join(stages) if stages else "None"))

            if "is_contradictory" in curated_rows.columns:
                contradictions = int(curated_rows["is_contradictory"].fillna(0).astype(float).sum())
                st.write(f"Contradictory flags: {contradictions}")

            if "modality_bias" in curated_rows.columns:
                modality_biases = sorted(curated_rows["modality_bias"].dropna().astype(str).unique().tolist())
                if modality_biases:
                    st.write("Preferred modality: " + ", ".join(modality_biases))
        else:
            st.caption("No curated liver-disease entry linked.")

    st.markdown("#### TCGA pan-cancer expression")
    tcga_present = any(col in selected_rows.columns for col in TCGA_COLS)

    if tcga_present:
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("TCGA mean", f"{safe_float(row.get('expr_mean_tcga_pancan', 0)):.3f}")
        t2.metric("TCGA median", f"{safe_float(row.get('expr_median_tcga_pancan', 0)):.3f}")
        t3.metric("TCGA max", f"{safe_float(row.get('expr_max_tcga_pancan', 0)):.3f}")
        t4.metric("TCGA prevalence", f"{safe_float(row.get('expr_prevalence_tcga_pancan', 0)):.3f}")
        t5.metric("In TCGA", safe_int(row.get("in_tcga_pancan_expr", 0)))
    else:
        st.caption("No TCGA columns available in target_scores yet.")

    st.markdown("#### Top evidence")
    render_clean_list(row["top_evidence_list"], "No top evidence statements available.")

    st.markdown("#### Risk flags")
    render_clean_list(row["risk_flags_list"], "No major risk flags identified.")

    st.markdown("#### Recommended experiments")
    render_clean_list(row["recommended_experiments_list"], "No specific experiments recommended.")

    if not curated_rows.empty:
        available_cols = [
            col for col in [
                "disease_stage",
                "disease_label",
                "evidence_tier",
                "direction",
                "mechanism_note",
                "source_ref",
                "pmid",
                "is_contradictory",
                "curator_note",
            ] if col in curated_rows.columns
        ]

        if available_cols:
            show_df = curated_rows[available_cols].copy()
            if "evidence_tier" in show_df.columns:
                show_df["evidence_tier"] = show_df["evidence_tier"].apply(normalize_tier_label)

            st.markdown("#### Curated literature notes")
            st.dataframe(show_df, width="stretch", height=220)


if __name__ == "__main__":
    main()