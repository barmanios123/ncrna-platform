from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPR_PATH = PROJECT_ROOT / "data" / "processed" / "gse126848_expression.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "gse126848_metadata_template.csv"


def main():
    df = pd.read_csv(EXPR_PATH)
    sample_cols = [c for c in df.columns if c != "symbol"]

    meta = pd.DataFrame({
        "sample_id": sample_cols,
        "disease_label": ["TODO"] * len(sample_cols),
        "disease_stage": ["TODO"] * len(sample_cols),
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(OUT_PATH, index=False)
    print(f"Wrote template with {len(meta)} samples to {OUT_PATH}")


if __name__ == "__main__":
    main()