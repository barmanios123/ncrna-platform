"""
ETL: Ingest GSE135251 (Ann Daly MASLD cohort) into ncrna_platform.db

Dataset: GSE135251 - MASLD/NASH spectrum vs healthy liver biopsies
Format: 216 individual HTSeq-count files in GSE135251_RAW.tar
Series matrix: GSE135251_series_matrix.txt.gz

Biology (plain language):
  This is a human liver biopsy RNA-seq study. Patients span the full MASLD
  spectrum: healthy controls, simple steatosis, NASH without fibrosis, and
  advanced NASH/fibrosis. We compare MASLD (disease) vs healthy (control)
  to find ncRNAs that change across disease severity.

Pipeline:
  1. Read series matrix to get sample labels (disease vs healthy)
  2. Stream each per-sample count file from the tar (no disk extraction)
  3. Merge all samples into a genes x samples count matrix
  4. CPM normalize, filter low-expression genes
  5. Vectorized DE stats (log2FC + t-test with FDR)
  6. Write to expression_evidence in SQLite
"""
import gzip, io, sqlite3, tarfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH   = BASE_DIR / "ncrna_platform.db"
DATA_DIR  = BASE_DIR / "data"
TAR_PATH  = DATA_DIR / "GSE135251_RAW.tar"
SERIES    = DATA_DIR / "GSE135251_series_matrix.txt.gz"
SYMBOLS   = DATA_DIR / "raw" / "ensembl_to_symbol.csv"

DATASET_ID = "GSE135251"
DISEASE_ID = "DIS_001"   # MASLD / liver disease
CONTEXT_ID = "CTX_001"   # liver
MIN_CPM, MIN_SAMPLES, PSEUDO = 0.5, 3, 0.5

# HTSeq special rows to skip
HTSEQ_SKIP = {"__no_feature", "__ambiguous", "__too_low_aQual",
              "__not_aligned", "__alignment_not_unique"}

def parse_series_matrix():
    """Extract GSM->label mapping from the series matrix.
    Healthy = NAS score 0 or 'normal' or 'healthy'.
    Disease = any MASLD / steatosis / NASH sample.
    """
    accessions, label_map = [], {}
    try:
        with gzip.open(SERIES, "rt", encoding="latin-1") as fh:
            for line in fh:
                if line.startswith("!Sample_geo_accession"):
                    accessions = [p.strip().strip('"') for p in line.strip().split("\t")[1:]]
                elif line.startswith("!Sample_characteristics_ch1"):
                    parts = [p.strip().strip('"').lower() for p in line.strip().split("\t")[1:]]
                    for i, val in enumerate(parts):
                        if i >= len(accessions):
                            break
                        acc = accessions[i]
                        # NAS score: 0 = healthy baseline, >0 = disease
                        if "nas score: 0" in val or "nas:0" in val:
                            label_map.setdefault(acc, "healthy")
                        elif "nas score:" in val or "nas:" in val:
                            label_map.setdefault(acc, "disease")
                        # Text-based fallback
                        if "healthy" in val or "normal" in val or "control" in val:
                            label_map.setdefault(acc, "healthy")
                        elif any(k in val for k in ["nash", "nafld", "masld", "steatosis",
                                                    "fibrosis", "cirrhosis"]):
                            label_map.setdefault(acc, "disease")
    except Exception as e:
        print(f"  Warning parsing series matrix: {e}")
    print(f"  Parsed {len(accessions)} accessions, labeled {len(label_map)}")
    return accessions, label_map

def load_counts_from_tar():
    """Stream all per-sample HTSeq count files from tar.
    Returns DataFrame (genes x samples) and list of sample names.
    """
    print(f"Streaming count files from {TAR_PATH.name}...")
    counts = {}   # sample_name -> Series(gene->count)
    with tarfile.open(TAR_PATH, "r") as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".counts.txt.gz")]
        print(f"  Found {len(members)} count files")
        for m in members:
            sample_name = m.name.split("_")[0]  # e.g. GSM3998167
            fobj = tf.extractfile(m)
            if fobj is None:
                continue
            try:
                with gzip.open(fobj, "rt") as gz:
                    rows = {}
                    for line in gz:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split("\t")
                        gene = parts[0]
                        if gene in HTSEQ_SKIP:
                            continue
                        try:
                            rows[gene] = int(parts[1])
                        except (IndexError, ValueError):
                            pass
                counts[sample_name] = pd.Series(rows, dtype=float)
            except Exception as e:
                print(f"  Warning reading {m.name}: {e}")
    df = pd.DataFrame(counts).fillna(0)
    print(f"  Count matrix assembled: {df.shape} (genes x samples)")
    return df

def assign_labels(df, accessions, label_map):
    """Split count matrix columns into disease and healthy."""
    disease_cols, healthy_cols = [], []
    for col in df.columns:
        lbl = label_map.get(col)
        if lbl == "disease":
            disease_cols.append(col)
        elif lbl == "healthy":
            healthy_cols.append(col)
    # Fallback: split 50/50 by sample number
    if not disease_cols or not healthy_cols:
        print("  WARNING: label parse failed — using 50/50 split")
        mid = len(df.columns) // 2
        disease_cols = list(df.columns[:mid])
        healthy_cols = list(df.columns[mid:])
    print(f"  Disease: {len(disease_cols)} samples | Healthy: {len(healthy_cols)} samples")
    return df[disease_cols], df[healthy_cols]

def cpm(df):
    totals = df.sum(axis=0)
    totals = totals.replace(0, 1)   # avoid div-by-zero
    return df.divide(totals, axis=1) * 1e6

def filter_lowexpr(dc, hc):
    all_c = pd.concat([dc, hc], axis=1)
    keep  = (all_c >= MIN_CPM).sum(axis=1) >= MIN_SAMPLES
    print(f"  Genes after CPM filter: {keep.sum()} / {len(keep)}")
    return dc.loc[keep], hc.loc[keep]

def run_de(dc, hc):
    print("  Vectorized DE (ttest_ind axis=1)...")
    dl = np.log2(dc.values + PSEUDO)
    hl = np.log2(hc.values + PSEUDO)
    log2fc = dl.mean(axis=1) - hl.mean(axis=1)
    _, pvals = stats.ttest_ind(dl, hl, axis=1)
    pvals = np.nan_to_num(pvals, nan=1.0)
    de = pd.DataFrame({"gene": dc.index, "log2fc": log2fc, "pvalue": pvals})
    de = de.sort_values("pvalue")
    try:
        from statsmodels.stats.multitest import multipletests
        _, padj, _, _ = multipletests(de["pvalue"].values, method="fdr_bh")
        de["padj"] = padj
    except Exception:
        de["padj"] = de["pvalue"]
    de["direction"] = de["log2fc"].apply(lambda x: "up" if x > 0 else "down")
    print(f"  DE done: {len(de)} genes")
    return de

def write_db(de):
    print(f"Writing {len(de)} rows to DB...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM expression_evidence WHERE dataset_id = ?", (DATASET_ID,))
    rows = [
        (str(r.gene), DISEASE_ID, CONTEXT_ID,
         float(r.log2fc), float(r.pvalue), float(r.padj),
         0.0, 0.0, 0.0, 0.0,
         str(r.direction), DATASET_ID)
        for r in de.itertuples(index=False)
    ]
    conn.executemany("""
        INSERT INTO expression_evidence
            (ncrna_id, disease_id, context_id, log2fc, pvalue, padj,
             basemean, tpm_disease, tpm_healthy, specificity_tau,
             direction, dataset_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM expression_evidence WHERE dataset_id=?",
                     (DATASET_ID,)).fetchone()[0]
    conn.close()
    print(f"  Confirmed {n} rows in expression_evidence for {DATASET_ID}")

def main():
    print("=" * 50)
    print(f"GSE135251 Ingestion — MASLD/NASH Cohort")
    print(f"Dataset: {DATASET_ID} | DB: {DB_PATH.name}")
    print("=" * 50)

    print("\n[1] Parsing sample labels from series matrix...")
    accessions, label_map = parse_series_matrix()

    print("\n[2] Loading counts from tar archive...")
    df = load_counts_from_tar()

    print("\n[3] Assigning disease/healthy labels...")
    dc, hc = assign_labels(df, accessions, label_map)

    print("\n[4] CPM normalisation + low-expression filter...")
    dc_cpm = cpm(dc)
    hc_cpm = cpm(hc)
    dc_cpm, hc_cpm = filter_lowexpr(dc_cpm, hc_cpm)

    print("\n[5] Differential expression...")
    de = run_de(dc_cpm, hc_cpm)

    print("\n[6] Writing to SQLite...")
    write_db(de)

    print("\n=== Summary ===")
    print(f"  Genes ingested   : {len(de)}")
    print(f"  Significant (padj<0.05, |log2FC|>1): "
          f"{((de.padj < 0.05) & (de.log2fc.abs() > 1)).sum()}")
    print(f"  Up-regulated     : {(de.direction == 'up').sum()}")
    print(f"  Down-regulated   : {(de.direction == 'down').sum()}")
    print("Done.")

if __name__ == "__main__":
    main()
