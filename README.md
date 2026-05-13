# Liver ncRNA Translational Engine

A target prioritization platform for non-coding RNA (ncRNA) drug discovery in MASLD/MASH → fibrosis/HCC.

## What It Does
- Integrates GEO and TCGA expression data for a curated 12-ncRNA liver panel
- Computes 72 biological features per target (expression stats, TCGA pancan metrics, curated evidence)
- Scores and ranks ncRNA targets using a translational scoring model
- Displays results in a live Streamlit dashboard with confidence tiers and target dossiers

## Tech Stack
Python · pandas · SQLite · Streamlit · scikit-learn · JupyterLab

## How to Run
```bash
pip install -r requirements.txt
python run_pipeline.py
streamlit run app/dashboard.py --server.port 8502
```

## Data Sources
- GEO (Gene Expression Omnibus) — GSE126848 liver disease dataset
- TCGA Pan-Cancer RNA-seq (EBPlusPlusAdjustPANCANIlluminaHiSeqV2)
- Hand-curated knowledge graph (liver ncRNA evidence, mechanisms, clinical flags)

## Author
Beshoy Armanios, PhD — Pharmacology & RNA Therapeutics
