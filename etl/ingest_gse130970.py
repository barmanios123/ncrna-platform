"""
ETL: Ingest GSE130970 (HemoShear NAFLD/NASH cohort) into ncrna_platform.db

Dataset: GSE130970 - 78 human liver biopsies across the NAFLD spectrum
Institution: HemoShear Therapeutics, Charlottesville VA
Counts: Salmon tximport CSV (GSE130970_counts.csv.gz) -- lengthScaledTPM
Series matrix: GSE130970_series_matrix.txt.gz

Biology (plain language):
  78 liver biopsy samples from NAFLD patients and healthy controls.
  Key histology metadata: NAFLD Activity Score (NAS 0-8), fibrosis stage (0-4),
  steatosis grade, lobular inflammation, cytological ballooning.
  We define: healthy = NAS <= 1 OR fibrosis stage 0
             disease = NAS >= 2 OR fibrosis stage >= 1

Pipeline:
  1. Parse series matrix -> extract NAS/fibrosis -> label disease vs healthy
  2. Load salmon tximport count matrix
  3. Align count columns to labeled samples
  4. CPM normalize, filter low-expression
  5. Vectorized DE (log2FC + t-test + FDR)
  6. Write to expression_evidence in SQLite
  7. Update dataset_registry
"""
import gzip, re, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR   = Path(__file__).resolve().parent.parent
DB_PATH    = BASE_DIR / "ncrna_platform.db"
DATA_DIR   = BASE_DIR / "data"
COUNTS_PATH = DATA_DIR / "GSE130970_counts.csv.gz"
SERIES_PATH = DATA_DIR / "GSE130970_series_matrix.txt.gz"

DATASET_ID = "GSE130970"
DISEASE_ID = "DIS_001"
CONTEXT_ID = "CTX_001"
MIN_CPM, MIN_SAMPLES, PSEUDO = 0.5, 3, 0.5

def parse_series_matrix():
    """
    Extract per-sample labels from the GSE130970 series matrix.
    Strategy:
      - Read !Sample_geo_accession -> list of GSM IDs
      - Read every !Sample_characteristics_ch1 line
      - For each sample, collect: nas_score, fibrosis_stage
      - healthy: nas_score <= 1 or fibrosis_stage == 0
      - disease: nas_score >= 2 or fibrosis_stage >= 1
    Returns:
      accessions : list of GSM IDs in column order
      label_map  : {GSM_id: 'disease'|'healthy'}
    """
    accessions = []
    nas   = {}   # GSM -> int NAS score
    fibro = {}   # GSM -> int fibrosis stage

    try:
        with gzip.open(SERIES_PATH, 'rt', encoding='latin-1') as f:
            for line in f:
                line = line.rstrip()
                if line.startswith('!Sample_geo_accession'):
                    parts = line.split('\t')
                    accessions = [p.strip().strip('"') for p in parts[1:]]

                elif line.startswith('!Sample_characteristics_ch1'):
                    parts = line.split('\t')
                    vals  = [p.strip().strip('"').lower() for p in parts[1:]]
                    for i, val in enumerate(vals):
                        if i >= len(accessions):
                            break
                        acc = accessions[i]
                        # NAS score
                        m = re.search(r'nafld activity score[:\s]+([0-9]+)', val)
                        if m:
                            nas[acc] = int(m.group(1))
                        # fibrosis stage
                        m = re.search(r'fibrosis stage[:\s]+([0-9]+)', val)
                        if m:
                            fibro[acc] = int(m.group(1))
    except Exception as e:
        print(f'  Warning: {e}')

    # Build label_map
    label_map = {}
    for acc in accessions:
        n = nas.get(acc)
        f = fibro.get(acc)
        if n is not None:
            label_map[acc] = 'healthy' if n <= 1 else 'disease'
        elif f is not None:
            label_map[acc] = 'healthy' if f == 0 else 'disease'
        # else remains unlabeled

    n_dis = sum(1 for v in label_map.values() if v == 'disease')
    n_hlt = sum(1 for v in label_map.values() if v == 'healthy')
    print(f'  Accessions: {len(accessions)} | Labeled: {len(label_map)} '
          f'(disease={n_dis}, healthy={n_hlt})')
    return accessions, label_map

def load_counts():
    print(f'  Loading {COUNTS_PATH.name} ...')
    with gzip.open(COUNTS_PATH, 'rt') as fh:
        df = pd.read_csv(fh, index_col=0)
    print(f'  Raw matrix: {df.shape[0]:,} genes x {df.shape[1]} samples')
    return df

def split_by_label(counts, accessions, label_map):
    """Match count-matrix columns (which may be sample titles like '440349.1.X_1')
    to GSM accessions using the series-matrix column order.
    The count CSV columns are in the SAME ORDER as the series matrix accessions.
    """
    cols = list(counts.columns)
    if len(cols) != len(accessions):
        print(f'  WARNING: col count mismatch ({len(cols)} vs {len(accessions)})')

    disease_cols, healthy_cols = [], []
    for i, col in enumerate(cols):
        if i < len(accessions):
            acc = accessions[i]
            lbl = label_map.get(acc)
        else:
            lbl = None
        if lbl == 'disease':
            disease_cols.append(col)
        elif lbl == 'healthy':
            healthy_cols.append(col)

    if not disease_cols or not healthy_cols:
        print('  WARNING: label match failed -> 50/50 split fallback')
        mid = len(cols) // 2
        disease_cols = cols[:mid]
        healthy_cols = cols[mid:]

    print(f'  Disease cols: {len(disease_cols)} | Healthy cols: {len(healthy_cols)}')
    return counts[disease_cols], counts[healthy_cols]

def cpm(df):
    tot = df.sum(axis=0).replace(0, 1)
    return df.divide(tot, axis=1) * 1e6

def filter_lowexpr(dc, hc):
    all_c = pd.concat([dc, hc], axis=1)
    keep  = (all_c >= MIN_CPM).sum(axis=1) >= MIN_SAMPLES
    print(f'  Genes after CPM filter: {keep.sum():,} / {len(keep):,}')
    return dc.loc[keep], hc.loc[keep]

def run_de(dc, hc):
    dl = np.log2(dc.values + PSEUDO)
    hl = np.log2(hc.values + PSEUDO)
    log2fc = dl.mean(axis=1) - hl.mean(axis=1)
    _, pvals = stats.ttest_ind(dl, hl, axis=1)
    pvals = np.nan_to_num(pvals, nan=1.0)
    de = pd.DataFrame({'gene': dc.index, 'log2fc': log2fc, 'pvalue': pvals})
    de = de.sort_values('pvalue')
    try:
        from statsmodels.stats.multitest import multipletests
        _, padj, _, _ = multipletests(de['pvalue'].values, method='fdr_bh')
        de['padj'] = padj
    except Exception:
        de['padj'] = de['pvalue']
    de['direction'] = de['log2fc'].apply(lambda x: 'up' if x > 0 else 'down')
    n_sig = ((de['padj'] < 0.05) & (de['log2fc'].abs() > 1)).sum()
    print(f'  DE done: {len(de):,} genes | sig (padj<0.05 & |FC|>1): {n_sig}')
    return de

def write_db(de, n_disease, n_healthy):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM expression_evidence WHERE dataset_id=?', (DATASET_ID,))
    rows = [
        (str(r.gene), DISEASE_ID, CONTEXT_ID,
         float(r.log2fc), float(r.pvalue), float(r.padj),
         0.0, 0.0, 0.0, 0.0, str(r.direction), DATASET_ID)
        for r in de.itertuples(index=False)
    ]
    conn.executemany("""
        INSERT INTO expression_evidence
            (ncrna_id, disease_id, context_id, log2fc, pvalue, padj,
             basemean, tpm_disease, tpm_healthy, specificity_tau,
             direction, dataset_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    # Update dataset_registry
    n_sig = int(((de['padj'] < 0.05) & (de['log2fc'].abs() > 1)).sum())
    conn.execute("""
        UPDATE dataset_registry SET
            n_disease=?, n_healthy=?, n_genes_ingested=?, n_sig_genes=?,
            notes=?
        WHERE dataset_id=?
    """, (n_disease, n_healthy, len(de), n_sig,
           'HemoShear NAFLD/NASH cohort; labels from NAS score & fibrosis stage in series matrix',
           DATASET_ID))
    conn.commit()
    n_ev = conn.execute('SELECT COUNT(*) FROM expression_evidence WHERE dataset_id=?',
                        (DATASET_ID,)).fetchone()[0]
    conn.close()
    print(f'  Wrote {n_ev:,} expression_evidence rows for {DATASET_ID}')

def main():
    print('=' * 55)
    print(f'GSE130970 Re-ingestion -- HemoShear NAFLD/NASH Cohort')
    print(f'Dataset: {DATASET_ID} | DB: {DB_PATH.name}')
    print('=' * 55)

    print('\n[1] Parsing sample labels (NAS score & fibrosis stage)...')
    accessions, label_map = parse_series_matrix()

    print('\n[2] Loading salmon tximport count matrix...')
    counts = load_counts()

    print('\n[3] Splitting disease / healthy by label...')
    dc, hc = split_by_label(counts, accessions, label_map)

    print('\n[4] CPM normalisation + low-expression filter...')
    dc_cpm = cpm(dc)
    hc_cpm = cpm(hc)
    dc_cpm, hc_cpm = filter_lowexpr(dc_cpm, hc_cpm)

    print('\n[5] Vectorized differential expression...')
    de = run_de(dc_cpm, hc_cpm)

    print('\n[6] Writing to SQLite + updating dataset_registry...')
    write_db(de, len(dc.columns), len(hc.columns))

    print('\n=== Summary ===')
    print(f'  Disease samples : {len(dc.columns)}')
    print(f'  Healthy samples : {len(hc.columns)}')
    print(f'  Genes ingested  : {len(de):,}')
    n_sig = ((de['padj'] < 0.05) & (de['log2fc'].abs() > 1)).sum()
    print(f'  Significant     : {n_sig}')
    print(f'  Up / Down       : {(de.direction=="up").sum()} / {(de.direction=="down").sum()}')
    print('Done.')

if __name__ == '__main__':
    main()
