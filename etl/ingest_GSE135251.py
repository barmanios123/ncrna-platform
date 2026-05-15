from __future__ import annotations
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

BASE_DIR = Path.home() / 'ncrna_platform'
DB_PATH = BASE_DIR / 'ncrna_platform.db'
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
SYMBOL_PATH = RAW_DIR / 'ensembl_to_symbol.csv'
DATASET_ID = 'GSE135251'
DISEASE_ID = 'DIS_001'
CONTEXT_ID = 'CTX_001'
MIN_CPM = 0.5
MIN_SAMPLES_EXPRESSED = 3
PSEUDOCOUNT = 0.5

def load_counts():
    p = RAW_DIR / 'GSE135251_counts_matrix.csv.gz'
    print(f'Loading {p}')
    df = pd.read_csv(p, index_col=0, compression='gzip')
    print(f'  shape: {df.shape}')
    return df

def load_metadata():
    p = PROCESSED_DIR / 'GSE135251_metadata.csv'
    meta = pd.read_csv(p, index_col=0)
    return meta

def label_samples(meta, counts):
    disease_groups = {'NASH_F0-F1','NASH_F2','NASH_F3','NASH_F4','NAFL'}
    dc, hc = [], []
    for gsm in counts.columns:
        if gsm in meta.index:
            g = meta.loc[gsm, 'group_in_paper']
            if g in disease_groups: dc.append(gsm)
            elif g == 'control': hc.append(gsm)
    print(f'  Disease={len(dc)}, Healthy={len(hc)}')
    return counts[dc], counts[hc]

def cpm_normalize(c):
    lib = c.sum(axis=0).replace(0,1)
    return c.div(lib, axis=1) * 1e6

def filter_low_expression(d, h):
    all_cpm = pd.concat([d,h], axis=1)
    keep = (all_cpm >= MIN_CPM).sum(axis=1) >= MIN_SAMPLES_EXPRESSED
    print(f'  Genes kept: {keep.sum()}/{len(keep)}')
    return d.loc[keep], h.loc[keep]

def compute_de_stats(d, h):
    print('Computing DE stats...')
    dis = np.log2(d + PSEUDOCOUNT)
    hlt = np.log2(h + PSEUDOCOUNT)
    log2fc = dis.mean(axis=1) - hlt.mean(axis=1)
    pvals = [stats.ttest_ind(dis.loc[g], hlt.loc[g], equal_var=False)[1] for g in dis.index]
    de = pd.DataFrame({'log2fc': log2fc, 'pvalue': pvals,
                        'mean_dis': dis.mean(axis=1), 'mean_hlt': hlt.mean(axis=1)})
    try:
        from scipy.stats import false_discovery_control
        de['padj'] = false_discovery_control(de['pvalue'].fillna(1))
    except Exception:
        de['padj'] = de['pvalue']
    print(f'  DE shape: {de.shape}')
    return de

def write_to_db(de_df):
    conn = sqlite3.connect(DB_PATH)
    master = pd.read_sql('SELECT ncrna_id, ensembl_id FROM ncrna_master', conn)
    e2n = dict(zip(master['ensembl_id'], master['ncrna_id']))
    conn.execute('DELETE FROM expression_evidence WHERE dataset_id=?', (DATASET_ID,))
    now = datetime.utcnow().isoformat()
    rows = []
    for gene_id, row in de_df.iterrows():
        nid = e2n.get(gene_id)
        if nid is None: continue
        rows.append({'ncrna_id': nid, 'disease_id': DISEASE_ID,
                     'context_id': CONTEXT_ID, 'dataset_id': DATASET_ID,
                     'log2fc': row['log2fc'], 'pvalue': row['pvalue'],
                     'padj': row['padj'],
                     'basemean': (row['mean_dis']+row['mean_hlt'])/2,
                     'tpm_disease': row['mean_dis'],
                     'tpm_healthy': row['mean_hlt'],
                     'direction': 'up' if row['log2fc']>0 else 'down',
                     'added_date': now})
    if rows:
        ev = pd.DataFrame(rows)
        ev.to_sql('expression_evidence', conn, if_exists='append', index=False)
        print(f'  Wrote {len(ev)} evidence rows')
    else:
        print('  No matching rows')
    conn.commit(); conn.close()

def main():
    print('=== GSE135251 Ingestion ===')
    counts = load_counts()
    meta = load_metadata()
    d, h = label_samples(meta, counts)
    print('CPM normalizing...')
    d_cpm, h_cpm = cpm_normalize(d), cpm_normalize(h)
    d_cpm, h_cpm = filter_low_expression(d_cpm, h_cpm)
    de_df = compute_de_stats(d_cpm, h_cpm)
    write_to_db(de_df)
    conn = sqlite3.connect(DB_PATH)
    nm = pd.read_sql('SELECT COUNT(*) AS n FROM ncrna_master', conn)['n'].iloc[0]
    ne = pd.read_sql('SELECT COUNT(*) AS n FROM expression_evidence WHERE dataset_id=?',
                     conn, params=(DATASET_ID,))['n'].iloc[0]
    conn.close()
    print(f'Done. ncrna_master={nm}, {DATASET_ID} evidence={ne}')

if __name__ == '__main__':
    main()
