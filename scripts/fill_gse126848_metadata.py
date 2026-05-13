from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_PATH = PROJECT_ROOT / "data" / "processed" / "gse126848_metadata_template.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "gse126848_metadata.csv"


def infer_label(sample_id: str) -> tuple[str, str]:
    s = str(sample_id).strip().upper()

    if "NASH" in s:
        return "NASH", "NASH"
    if "NAFL" in s:
        return "NAFL", "NAFL"
    if "OBESE" in s or s.startswith("OB_") or s.startswith("OBESE_"):
        return "Obese", "Control"
    if "HEALTHY" in s or "CTRL" in s or "CONTROL" in s or s.startswith("H_"):
        return "Healthy", "Control"

    return "REVIEW", "REVIEW"


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Template not found: {IN_PATH}")

    df = pd.read_csv(IN_PATH)

    if "sample_id" not in df.columns:
        raise ValueError("Template must contain a sample_id column")

    labels = df["sample_id"].apply(infer_label)
    df["disease_label"] = labels.apply(lambda x: x[0])
    df["disease_stage"] = labels.apply(lambda x: x[1])

    df.to_csv(OUT_PATH, index=False)

    print(f"Wrote auto-filled metadata to {OUT_PATH}")
    print("\nLabel counts:")
    print(df["disease_label"].value_counts(dropna=False).to_string())

    review_df = df[df["disease_label"] == "REVIEW"]
    if not review_df.empty:
        print("\nSamples requiring manual review:")
        print(review_df["sample_id"].to_string(index=False))
    else:
        print("\nAll samples were auto-labeled.")
        

if __name__ == "__main__":
    main()