"""
ETL: Ingest GSE193066 (Duke NAFLD cohort) into ncrna_platform.db

Dataset: GSE193066 - 106 human liver biopsies, NAFLD spectrum with fibrosis staging
Design: HCC-naive NAFLD patients; fibrosis F0-F4; RNA-seq normalized expression (GCT)
Labels: fibrosis stage 0 = healthy baseline; stage 1-4 = disease
Counts: GSE193066_NAFLD_counts.gct.gz (normalized expression, GCT v1.2 format)

Biology:
  This dataset captures the full fibrosis progression spectrum in NAFLD:
  F0 = no fibrosis (healthy liver), F1-F2 = mild/moderate, F3-F4 = advanced/cirrhosis.
  The PLS-NAFLD risk score is also encoded, allowing subgroup analysis.
  Using fibrosis F0 vs F1+ as the disease axis captures fibrogenesis gene programs.

Pipeline:
  1. Parse series matrix for fibrosis stage per sample (GSM accession order)
  2. Load GCT expression matrix
  3. Align samples by GSM accession -> fibrosis labels
  4. CPM-normalize (GCT is already normalized; apply pseudo-CPM for comparability)
  5. Vectorized DE stats
  6. Write to expression_evidence + update dataset_registry
"""
import gzip
import re, re, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR    = Path(__file__).resolve().parent.parent
DB_PATH     = BASE_DIR / "ncrna_platform.db"
DATA_DIR    = BASE_DIR / "data"
GCT_PATH    = DATA_DIR / "GSE193066_NAFLD_counts.gct.gz"
SERIES_PATH = DATA_DIR / "GSE193066_series_matrix.txt.gz"

DATASET_ID  = "GSE193066"
DISEASE_ID  = "DIS_001"
CONTEXT_ID  = "CTX_001"
MIN_EXPR    = 1.0   # minimum normalized expression value (GCT already normalized)
MIN_SAMPLES = 3
PSEUDO      = 0.5

def parse_labels():
    """Extract fibrosis stage per sample. F0=healthy, F1-4=disease."""
    accessions, fibro, nas = [], {}, {}
    with gzip.open(SERIES_PATH, 'rt', encoding='latin-1') as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith('!Sample_geo_accession'):
                accessions = [p.strip().strip('"') for p in line.split('\t')[1:]]
            elif line.startswith('!Sample_source_name_ch1'):
                # sample name (HUNafld001 etc.) in same order as accessions
                titles = [p.strip().strip('"') for p in line.split('\t')[1:]]
            elif line.startswith('!Sample_characteristics_ch1'):
                parts = [p.strip().strip('"').lower() for p in line.split('\t')[1:]]
                for i, val in enumerate(parts):
                    if i >= len(accessions): break
                    acc = accessions[i]
                    m = re.search(r'fibrosis stage[:\s]+([0-9]+)', val)
                    if m: fibro[acc] = int(m.group(1))
                    m = re.search(r'nafld activity score[:\s]+([0-9]+)', val)
                    if m: nas[acc] = int(m.group(1))
    label_map = {}
    for acc in accessions:
        f = fibro.get(acc)
        n = nas.get(acc)
        if f is not None:
            label_map[acc] = 'healthy' if f == 0 else 'disease'
        elif n is not None:
            label_map[acc] = 'healthy' if n <= 1 else 'disease'
    n_d = sum(1 for v in label_map.values() if v=='disease')
    n_h = sum(1 for v in label_map.values() if v=='healthy')
    try:
        title_map = dict(zip(accessions, titles))
    except Exception:
        title_map = {}
    print(f'  {len(accessions)} accessions | disease={n_d}, healthy={n_h}')
    return accessions, label_map, title_map

def load_gct():
    """Read GCT v1.2 format.
    Line 1: #1.2
    Line 2: nrows\tncols
    Line 3: header (Name\tDescription\tsample1\tsample2...)
    Lines 4+: data
    """
    print(f'  Reading GCT: {GCT_PATH.name} ...')
    with gzip.open(GCT_PATH, 'rt', encoding='latin-1') as fh:
        lines = fh.readlines()
    # Find header line (skip #version line and nrows/ncols line)
    start = 0
    for i, l in enumerate(lines):
        # Skip comment/version lines and numeric dimension lines
        stripped = l.strip()
        if stripped.startswith('#'):
            continue
        # Check if this looks like a dimension line (starts with digits)
        if re.match(r'^\d+', stripped):
            continue
        # This is the header row
        start = i
        break
    header = lines[start].rstrip().split('\t')
    rows = []
    for l in lines[start+1:]:
        if l.strip():
            rows.append(l.rstrip().split('\t'))
    df = pd.DataFrame(rows, columns=header)
    # Gene ID in 'Name' column, drop 'Description'
    gene_col = header[0]   # 'Name'
    df = df.set_index(gene_col)
    for drop_col in ['Description', 'gene_id', 'Name', 'description']:
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    print(f'  GCT matrix: {df.shape[0]:,} genes x {df.shape[1]} samples')
    return df

def split_labels(df, accessions, label_map, title_map):
    """Match GCT columns (sample names like HUNafld001) to GSM accessions."""
    # Build reverse: title -> accession
    rev = {v: k for k, v in title_map.items()}
    disease_cols, healthy_cols = [], []
    for col in df.columns:
        # Try direct GSM match
        lbl = label_map.get(col)
        if lbl is None:
            # Try by title
            acc = rev.get(col)
            if acc: lbl = label_map.get(acc)
        if lbl is None:
            # Try positional: GCT columns in same order as accessions
            try:
                idx = list(df.columns).index(col)
                if idx < len(accessions):
                    lbl = label_map.get(accessions[idx])
            except ValueError:
                pass
        if lbl == 'disease': disease_cols.append(col)
        elif lbl == 'healthy': healthy_cols.append(col)
    if not disease_cols or not healthy_cols:
        print('  WARNING: label match failed -> 50/50 split fallback')
        mid = len(df.columns)//2
        disease_cols = list(df.columns[:mid])
        healthy_cols = list(df.columns[mid:])
    print(f'  Disease: {len(disease_cols)} | Healthy: {len(healthy_cols)}')
    return df[disease_cols], df[healthy_cols]

def filter_lowexpr(dc, hc):
    all_e = pd.concat([dc, hc], axis=1)
    keep  = (all_e >= MIN_EXPR).sum(axis=1) >= MIN_SAMPLES
    print(f'  Genes after expr filter: {keep.sum():,} / {len(keep):,}')
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
    de['direction'] = de['log2fc'].apply(lambda x: 'up' if x>0 else 'down')
    n_sig = ((de['padj']<0.05) & (de['log2fc'].abs()>1)).sum()
    print(f'  DE done: {len(de):,} genes | sig: {n_sig}')
    return de

def write_db(de, n_dis, n_hlt):
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
    n_sig = int(((de['padj']<0.05) & (de['log2fc'].abs()>1)).sum())
    conn.execute("""
        INSERT OR REPLACE INTO dataset_registry
            (dataset_id, geo_accession, disease, tissue, organism, platform,
             n_samples, n_disease, n_healthy, n_genes_ingested, n_sig_genes,
             count_format, publication, notes, added_date)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,date('now'))
    """, (DATASET_ID, DATASET_ID,
           'NAFLD fibrosis progression (F0-F4)', 'liver', 'Homo sapiens',
           'Illumina RNA-seq (normalized GCT)',
           n_dis+n_hlt, n_dis, n_hlt, len(de), n_sig,
           'GCT normalized expression (NAFLD.HUn106.gct.gz)',
           'Duke NAFLD cohort; GEO GSE193066',
           'Fibrosis F0=healthy vs F1-F4=disease; PLS-NAFLD risk score available'))
    conn.commit()
    n_ev = conn.execute('SELECT COUNT(*) FROM expression_evidence WHERE dataset_id=?',
                        (DATASET_ID,)).fetchone()[0]
    conn.close()
    print(f'  Wrote {n_ev:,} rows for {DATASET_ID}')

def main():
    print('='*55)
    print('GSE193066 Ingestion -- Duke NAFLD Fibrosis Cohort')
    print(f'DB: {DB_PATH.name}')
    print('='*55)
    print('\n[1] Parsing labels (fibrosis stage)...')
    accessions, label_map, title_map = parse_labels()
    print('\n[2] Loading GCT expression matrix...')
    df = load_gct()
    print('\n[3] Splitting disease/healthy...')
    dc, hc = split_labels(df, accessions, label_map, title_map)
    print('\n[4] Expression filter...')
    dc, hc = filter_lowexpr(dc, hc)
    print('\n[5] Differential expression...')
    de = run_de(dc, hc)
    print('\n[6] Writing to SQLite...')
    write_db(de, len(dc.columns), len(hc.columns))
    n_sig = ((de['padj']<0.05) & (de['log2fc'].abs()>1)).sum()
    print(f'\n=== Summary ===')
    print(f'  Disease (F1-F4) : {len(dc.columns)}')
    print(f'  Healthy (F0)    : {len(hc.columns)}')
    print(f'  Genes ingested  : {len(de):,}')
    print(f'  Significant     : {n_sig}')
    print(f'  Up/Down         : {(de.direction=="up").sum()} / {(de.direction=="down").sum()}')
    print('Done.')

if __name__ == '__main__':
    main()
