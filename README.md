# Liver ncRNA Translational Engine

A target prioritization platform for non-coding RNA (ncRNA) drug discovery in MASLD/MASH → fibrosis/HCC.

## What It Does
- Integrates GEO and TCGA expression data for a curated 12-ncRNA liver panel
- Computes 60+ biological features per target, including GEO expression statistics, TCGA pan-cancer metrics, curated liver-disease evidence, and model-derived risk flags
- Scores and ranks ncRNA targets using a translational scoring model with confidence tiers
- Computes a Geneformer-like composite score per target from regulatory, perturbation, disease-shift, context, and risk components
- Persists raw provenance features (pathways, perturbation evidence, clinical links, TCGA metrics, and risk flags) in SQLite for transparent rank traceability
- Displays results in a Streamlit dashboard with confidence tiers, target dossiers, and mechanistic context

## Tech Stack
Python · pandas · SQLite · Streamlit · scikit-learn · JupyterLab

## How to Run

From the project root:

```bash
pip install -r requirements.txt
python health_check.py
python misc/run_pipeline.py
streamlit run app/dashboard.py
```

Alternatively, you can run the core modules directly:

```bash
python -m models.features
python -m models.scoring
streamlit run app/dashboard.py
```

## Project Structure
- `app/dashboard.py` — Streamlit dashboard
- `misc/run_pipeline.py` — pipeline launcher
- `health_check.py` — project and database validation checks
- `models/features.py` — feature engineering and risk flag computation
- `models/scoring.py` — translational scoring, Geneformer-like scoring, and SQLite persistence
- `ncrna_platform.db` — local SQLite database
- `outputs/` — generated outputs and result files

## Data Sources
- GEO (Gene Expression Omnibus) — GSE126848 liver disease dataset
- TCGA Pan-Cancer RNA-seq — EBPlusPlusAdjustPANCANIlluminaHiSeqRNASeqV2 (11,069 samples)
- Structured knowledge graph — seeded with liver ncRNA evidence, mechanisms, and clinical flags for 12 ncRNA targets

## Author
Beshoy Armanios, PhD MPH — Pharmacology & RNA Therapeutics