"""
ETL: Ingest GSE126848 bulk RNA-seq into ncrna_platform.db

Dataset: GSE126848 - Hepatic transcriptome in NAFLD/NASH vs healthy liver
Groups:
    Disease : NAFL (n=15) + NASH (n=16)
    Healthy : Normal-weight (n=14) + Obese (n=12)

Steps:
    1. Load raw count matrix
    2. Parse sample group labels from series matrix
    3. CPM-normalize counts
    4. Compute per-gene DE stats (log2FC, pvalue, padj, direction)
    5. Map Ensembl IDs to gene symbols
    6. Write to ncrna_master and expression_evidence in SQLite

Usage:
    python etl/ingest_bulk.py
"""

from __future__ import annotations

import gzip
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "ncrna_platform.db"
RAW_DIR = BASE_DIR / "data" / "raw"

COUNTS_PATH = RAW_DIR / "GSE126848_Gene_counts_raw.txt.gz"
SERIES_PATH = RAW_DIR / "GSE126848_series_matrix.txt.gz"
SYMBOL_PATH = RAW_DIR / "ensembl_to_symbol.csv"

DISEASE_ID = "DIS_001"
CONTEXT_ID = "CTX_001"
DATASET_ID = "GSE126848"

DISEASE_PREFIXES = ("NAFL", "NASH")
HEALTHY_PREFIXES = ("Normal-weight", "Obese")

MIN_CPM = 0.5
MIN_SAMPLES_EXPRESSED = 3
PSEUDOCOUNT = 0.5


def load_counts() -> pd.DataFrame:
    print(f"Loading count matrix: {COUNTS_PATH}")
    with gzip.open(COUNTS_PATH, "rt") as f:
        df = pd.read_csv(f, sep="\t", index_col=0)
    print(f"  Raw count matrix shape: {df.shape}")
    return df


def parse_sample_labels(series_path: Path) -> tuple[list, list, dict]:
    titles = []
    accessions = []

    with gzip.open(series_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!Sample_title"):
                parts = line.strip().split("\t")
                titles = [p.strip().strip('"') for p in parts[1:]]
            if line.startswith("!Sample_geo_accession"):
                parts = line.strip().split("\t")
                accessions = [p.strip().strip('"') for p in parts[1:]]

    label_map = {}
    for title in titles:
        matched = False
        for prefix in DISEASE_PREFIXES:
            if title.startswith(prefix):
                label_map[title] = "disease"
                matched = True
                break
        if not matched:
            for prefix in HEALTHY_PREFIXES:
                if title.startswith(prefix):
                    label_map[title] = "healthy"
                    break

    print(f"  Parsed {len(titles)} sample titles")
    print(f"  Disease: {sum(1 for v in label_map.values() if v == 'disease')}")
    print(f"  Healthy: {sum(1 for v in label_map.values() if v == 'healthy')}")

    return titles, accessions, label_map


def align_columns_to_labels(
    counts: pd.DataFrame,
    titles: list,
    accessions: list,
    label_map: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_cols = counts.columns.tolist()
    n_cols = len(count_cols)
    n_titles = len(titles)

    if n_cols != n_titles:
        print(f"  Warning: count cols={n_cols}, titles={n_titles}, truncating to min")
        n = min(n_cols, n_titles)
        count_cols = count_cols[:n]
        titles = titles[:n]
        counts = counts.iloc[:, :n]

    col_to_title = dict(zip(count_cols, titles))
    col_to_group = {
        col: label_map.get(title, "unknown")
        for col, title in col_to_title.items()
    }

    disease_cols = [c for c, g in col_to_group.items() if g == "disease"]
    healthy_cols = [c for c, g in col_to_group.items() if g == "healthy"]

    print(f"  Disease columns: {len(disease_cols)}")
    print(f"  Healthy columns: {len(healthy_cols)}")

    return counts[disease_cols], counts[healthy_cols]


def cpm_normalize(counts: pd.DataFrame) -> pd.DataFrame:
    lib_sizes = counts.sum(axis=0)
    lib_sizes = lib_sizes.replace(0, 1)
    return counts.div(lib_sizes, axis=1) * 1e6


def filter_low_expression(
    disease_cpm: pd.DataFrame,
    healthy_cpm: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expressed_d = (disease_cpm >= MIN_CPM).sum(axis=1)
    expressed_h = (healthy_cpm >= MIN_CPM).sum(axis=1)
    keep = (expressed_d >= MIN_SAMPLES_EXPRESSED) | (expressed_h >= MIN_SAMPLES_EXPRESSED)
    print(f"  Genes before filter: {len(disease_cpm)}")
    disease_cpm = disease_cpm[keep]
    healthy_cpm = healthy_cpm[keep]
    print(f"  Genes after CPM filter: {len(disease_cpm)}")
    return disease_cpm, healthy_cpm


def compute_de_stats(
    disease_cpm: pd.DataFrame,
    healthy_cpm: pd.DataFrame,
) -> pd.DataFrame:
    print("Computing DE statistics...")

    disease_mean = disease_cpm.mean(axis=1) + PSEUDOCOUNT
    healthy_mean = healthy_cpm.mean(axis=1) + PSEUDOCOUNT
    log2fc = np.log2(disease_mean / healthy_mean)

    disease_log = np.log2(disease_cpm + PSEUDOCOUNT)
    healthy_log = np.log2(healthy_cpm + PSEUDOCOUNT)

    pvalues = []
    for gene in disease_cpm.index:
        d_vals = disease_log.loc[gene].values
        h_vals = healthy_log.loc[gene].values
        if len(d_vals) < 2 or len(h_vals) < 2:
            pvalues.append(1.0)
            continue
        try:
            _, p = stats.ttest_ind(d_vals, h_vals, equal_var=False)
            pvalues.append(float(p) if not np.isnan(p) else 1.0)
        except Exception:
            pvalues.append(1.0)

    pvalues_series = pd.Series(pvalues, index=disease_cpm.index)

    try:
        from statsmodels.stats.multitest import multipletests
        _, padj_arr, _, _ = multipletests(pvalues_series.values, method="fdr_bh")
        padj = pd.Series(padj_arr, index=disease_cpm.index)
    except Exception:
        print("  statsmodels not available, using raw pvalues as padj")
        padj = pvalues_series.copy()

    direction = pd.Series(
        np.where(log2fc > 0, "up", "down"),
        index=disease_cpm.index,
    )

    de_df = pd.DataFrame({
        "ensembl_id": disease_cpm.index,
        "log2fc": log2fc.values,
        "pvalue": pvalues_series.values,
        "padj": padj.values,
        "basemean": ((disease_mean + healthy_mean) / 2).values,
        "tpm_disease": disease_cpm.mean(axis=1).values,
        "tpm_healthy": healthy_cpm.mean(axis=1).values,
        "direction": direction.values,
    })

    sig = (de_df["padj"] <= 0.05).sum()
    print(f"  DE computed for {len(de_df)} genes")
    print(f"  Significant (padj<=0.05): {sig}")

    return de_df


def load_symbol_map() -> pd.DataFrame:
    df = pd.read_csv(SYMBOL_PATH)
    df.columns = df.columns.str.strip().str.lower()
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["ensembl_id"] = df["ensembl_id"].astype(str).str.strip()
    print(f"  Symbol map entries: {len(df)}")
    return df


def write_to_db(de_df: pd.DataFrame, symbol_map: pd.DataFrame) -> None:
    merged = de_df.merge(symbol_map, on="ensembl_id", how="left")
    merged_with_symbol = merged[merged["symbol"].notna()].copy()
    print(f"  Genes with symbol: {len(merged_with_symbol)} / {len(de_df)}")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    existing_master = pd.read_sql_query(
        "SELECT ncrna_id, symbol FROM ncrna_master", conn
    )
    existing_symbols = set(
        existing_master["symbol"].astype(str).str.strip().tolist()
    )
    symbol_to_ncrna_id = dict(
        zip(
            existing_master["symbol"].astype(str).str.strip(),
            existing_master["ncrna_id"].astype(str),
        )
    )

    max_id_row = pd.read_sql_query(
        "SELECT MAX(CAST(SUBSTR(ncrna_id, 8) AS INTEGER)) AS max_id FROM ncrna_master",
        conn,
    )
    max_id = int(max_id_row["max_id"].iloc[0]) if max_id_row["max_id"].iloc[0] else 0

    new_master_rows = []
    for _, row in merged_with_symbol.iterrows():
        sym = str(row["symbol"]).strip()
        if sym not in existing_symbols:
            max_id += 1
            ncrna_id = f"LNCRNA_{max_id:04d}"
            new_master_rows.append({
                "ncrna_id": ncrna_id,
                "symbol": sym,
                "biotype": "lncRNA",
                "conservation_score": 0.0,
                "transcript_count": 1,
            })
            symbol_to_ncrna_id[sym] = ncrna_id
            existing_symbols.add(sym)

    if new_master_rows:
        new_master_df = pd.DataFrame(new_master_rows)
        new_master_df.to_sql("ncrna_master", conn, if_exists="append", index=False)
        print(f"  Added {len(new_master_rows)} new ncRNAs to ncrna_master")
    else:
        print("  No new ncRNAs to add to ncrna_master")

    c.execute(
        "DELETE FROM expression_evidence WHERE dataset_id = ?",
        (DATASET_ID,),
    )

    evidence_rows = []
    for _, row in merged_with_symbol.iterrows():
        sym = str(row["symbol"]).strip()
        ncrna_id = symbol_to_ncrna_id.get(sym)
        if ncrna_id is None:
            continue

        evidence_id = f"EXP_{DATASET_ID}_{ncrna_id}_{DISEASE_ID}_{CONTEXT_ID}"
        evidence_rows.append({
            "evidence_id": evidence_id,
            "ncrna_id": ncrna_id,
            "disease_id": DISEASE_ID,
            "context_id": CONTEXT_ID,
            "log2fc": round(float(row["log2fc"]), 6),
            "pvalue": round(float(row["pvalue"]), 6),
            "padj": round(float(row["padj"]), 6),
            "basemean": round(float(row["basemean"]), 6),
            "tpm_disease": round(float(row["tpm_disease"]), 6),
            "tpm_healthy": round(float(row["tpm_healthy"]), 6),
            "specificity_tau": 0.0,
            "direction": str(row["direction"]),
            "dataset_id": DATASET_ID,
        })

    if evidence_rows:
        ev_df = pd.DataFrame(evidence_rows)
        ev_df.to_sql("expression_evidence", conn, if_exists="append", index=False)
        print(f"  Wrote {len(ev_df)} expression_evidence rows")
    else:
        print("  No evidence rows to write")

    conn.commit()
    conn.close()
    print("Done writing to DB.")


def main():
    print("=== GSE126848 Bulk RNA-seq Ingestion ===")
    print(f"Database: {DB_PATH}")

    counts = load_counts()

    print(f"Parsing sample labels: {SERIES_PATH}")
    titles, accessions, label_map = parse_sample_labels(SERIES_PATH)

    disease_counts, healthy_counts = align_columns_to_labels(
        counts, titles, accessions, label_map
    )

    print("CPM normalizing...")
    disease_cpm = cpm_normalize(disease_counts)
    healthy_cpm = cpm_normalize(healthy_counts)

    disease_cpm, healthy_cpm = filter_low_expression(disease_cpm, healthy_cpm)

    de_df = compute_de_stats(disease_cpm, healthy_cpm)

    print(f"Loading symbol map: {SYMBOL_PATH}")
    symbol_map = load_symbol_map()

    write_to_db(de_df, symbol_map)

    print("\n=== Summary ===")
    conn = sqlite3.connect(DB_PATH)
    n_master = pd.read_sql_query(
        "SELECT COUNT(*) AS n FROM ncrna_master", conn
    )["n"].iloc[0]
    n_evidence = pd.read_sql_query(
        "SELECT COUNT(*) AS n FROM expression_evidence WHERE dataset_id = ?",
        conn,
        params=(DATASET_ID,),
    )["n"].iloc[0]
    conn.close()

    print(f"  ncrna_master total rows : {n_master}")
    print(f"  expression_evidence rows for {DATASET_ID}: {n_evidence}")
    print("Done.")


if __name__ == "__main__":
    main()