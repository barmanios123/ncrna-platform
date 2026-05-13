import gzip
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERIES_PATH = PROJECT_ROOT / "data" / "raw" / "GSE126848_series_matrix.txt.gz"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "gse126848_metadata.csv"


def main():
    if not SERIES_PATH.exists():
        raise FileNotFoundError(f"Series matrix not found: {SERIES_PATH}")

    with gzip.open(SERIES_PATH, "rt") as f:
        lines = [line.rstrip("\n") for line in f]

    sample_meta = {}
    sample_ids = []

    for line in lines:
        if line.startswith("!Sample_title"):
            parts = line.split("\t")
            sample_ids = [p.replace('"', "") for p in parts[1:]]
            sample_meta["sample_id"] = sample_ids

        elif line.startswith("!Sample_characteristics_ch1"):
            parts = line.split("\t")
            values = [p.replace('"', "") for p in parts[1:]]
            # Some GSEs put everything under one generic header; keep as one column
            sample_meta["characteristics"] = values

    if not sample_ids:
        raise RuntimeError("Could not find !Sample_title in series matrix")
    if "characteristics" not in sample_meta:
        raise RuntimeError("Could not find !Sample_characteristics_ch1 in series matrix")

    meta_df = pd.DataFrame(sample_meta)

    def map_group_from_char(value: str):
        v = str(value).lower()
        # Adjust patterns if needed after inspection
        if "nash" in v:
            return "NASH", "NASH"
        if "nafl" in v:
            return "NAFL", "NAFL"
        if "obese" in v:
            return "Obese", "Control"
        if "healthy" in v or "normal weight" in v or "control" in v:
            return "Healthy", "Control"
        return "REVIEW", "REVIEW"

    labels = meta_df["characteristics"].apply(map_group_from_char)
    meta_df["disease_label"] = labels.apply(lambda x: x[0])
    meta_df["disease_stage"] = labels.apply(lambda x: x[1])

    meta_df = meta_df[["sample_id", "disease_label", "disease_stage", "characteristics"]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(OUT_PATH, index=False)

    print(f"Wrote metadata to {OUT_PATH}")
    print("\nLabel counts:")
    print(meta_df["disease_label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()