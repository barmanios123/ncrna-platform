import gzip, sqlite3, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "ncrna_platform.db"
DATA_DIR = BASE_DIR / "data"
COUNTS_PATH = DATA_DIR / "GSE130970_counts.csv.gz"
SERIES_PATH = DATA_DIR / "GSE130970_series_matrix.txt.gz"
SYMBOL_PATH = DATA_DIR / "raw" / "ensembl_to_symbol.csv"
DATASET_ID = "GSE130970"
DISEASE_ID = "DIS_001"
CONTEXT_ID = "CTX_001"
MIN_CPM, MIN_SAMPLES, PSEUDO = 0.5, 3, 0.5

def main():
    print("=== GSE130970 ingestion ===")
    with gzip.open(COUNTS_PATH, "rt") as f:
        counts = pd.read_csv(f, index_col=0)
    print(f"Counts: {counts.shape}")
    n = len(counts.columns)
    mid = n // 2
    d_cols = list(counts.columns[:mid])
    h_cols = list(counts.columns[mid:])
    print(f"Split: {len(d_cols)} disease / {len(h_cols)} healthy")
    dc = counts[d_cols]
    hc = counts[h_cols]
    def cpm(df):
        return df.divide(df.sum(axis=0), axis=1) * 1e6
    dc_cpm = cpm(dc)
    hc_cpm = cpm(hc)
    all_cpm = pd.concat([dc_cpm, hc_cpm], axis=1)
    keep = (all_cpm >= MIN_CPM).sum(axis=1) >= MIN_SAMPLES
    dc_cpm = dc_cpm.loc[keep]
    hc_cpm = hc_cpm.loc[keep]
    print(f"Genes after filter: {keep.sum()}")
    print("Computing DE (vectorized)...")
    d_log = np.log2(dc_cpm.values + PSEUDO)
    h_log = np.log2(hc_cpm.values + PSEUDO)
    log2fc = d_log.mean(axis=1) - h_log.mean(axis=1)
    _, pvals = stats.ttest_ind(d_log, h_log, axis=1)
    pvals = np.nan_to_num(pvals, nan=1.0)
    genes = dc_cpm.index.tolist()
    de = pd.DataFrame({"gene": genes, "log2fc": log2fc, "pvalue": pvals})
    de = de.sort_values("pvalue")
    try:
        from statsmodels.stats.multitest import multipletests
        _, padj, _, _ = multipletests(de["pvalue"].values, method="fdr_bh")
        de["padj"] = padj
    except Exception:
        de["padj"] = de["pvalue"]
    de["direction"] = de["log2fc"].apply(lambda x: "up" if x > 0 else "down")
    print(f"DE done: {len(de)} genes")
    if SYMBOL_PATH.exists():
        sm = pd.read_csv(SYMBOL_PATH)
        if "ensembl_gene_id" in sm.columns:
            smap = dict(zip(sm["ensembl_gene_id"], sm.get("hgnc_symbol", sm.iloc[:,1])))
        else:
            smap = {}
    else:
        smap = {}
    print(f"Writing to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM expression_evidence WHERE dataset_id = ?", (DATASET_ID,))
    rows = []
    for _, r in de.iterrows():
        eid = str(r["gene"])
        sym = smap.get(eid, eid)
        rows.append((
            eid, DISEASE_ID, CONTEXT_ID, float(r["log2fc"]),
            float(r["pvalue"]), float(r["padj"]), 0.0, 0.0, 0.0,
            0.0, str(r["direction"]), DATASET_ID
        ))
    conn.executemany("""
        INSERT INTO expression_evidence
            (ncrna_id, disease_id, context_id, log2fc, pvalue, padj,
             basemean, tpm_disease, tpm_healthy, specificity_tau, direction, dataset_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    n_ev = conn.execute("SELECT COUNT(*) FROM expression_evidence WHERE dataset_id=?", (DATASET_ID,)).fetchone()[0]
    conn.close()
    print(f"Wrote {n_ev} rows to expression_evidence.")
    print("Done.")

if __name__ == "__main__":
    main()
