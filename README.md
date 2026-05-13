# Liver ncRNA Translational Engine

A target prioritization platform for non-coding RNA (ncRNA) drug discovery in MASLD/MASH → fibrosis/HCC.

## What It Does
- Integrates GEO and TCGA expression data for a curated 12-ncRNA liver panel
- Computes 72 biological features per target, including expression statistics, TCGA pan-cancer metrics, and curated evidence
- Scores and ranks ncRNA targets using a translational scoring model
- Displays results in a Streamlit dashboard with confidence tiers and target dossiers

## Tech Stack
Python · pandas · SQLite · Streamlit · scikit-learn · JupyterLab

## How to Run
```bash
pip install -r requirements.txt
python health_check.py
python misc/run_pipeline.py
streamlit run app/dashboard.py
```

## Project Structure
- `app/dashboard.py` — Streamlit dashboard
- `misc/run_pipeline.py` — pipeline launcher
- `health_check.py` — project and database validation checks
- `ncrna_platform.db` — local SQLite database
- `outputs/` — generated outputs and result files

## Data Sources
- GEO (Gene Expression Omnibus) — GSE126848 liver disease dataset
- TCGA Pan-Cancer RNA-seq (EBPlusPlusAdjustPANCANIlluminaHiSeqV2)
- Hand-curated knowledge graph with liver ncRNA evidence, mechanisms, and clinical flags

## Author
Beshoy Armanios, PhD MPH — Pharmacology & RNA Therapeutics