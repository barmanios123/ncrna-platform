import gzip
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "GSE126848_Gene_counts_raw.txt.gz"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "gse126848_expression.csv"


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_PATH}")

    with gzip.open(RAW_PATH, "rt") as f:
        df = pd.read_csv(f, sep="\t")

    # Standardize gene column name to 'symbol'
    if "GeneSymbol" in df.columns:
        df = df.rename(columns={"GeneSymbol": "symbol"})
    elif "gene" in df.columns:
        df = df.rename(columns={"gene": "symbol"})
    elif df.columns[0].lower().startswith("gene"):
        df = df.rename(columns={df.columns[0]: "symbol"})
    else:
        df = df.rename(columns={df.columns[0]: "symbol"})

    df = df[df["symbol"].notna()]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} with shape {df.shape}")


if __name__ == "__main__":
    main()