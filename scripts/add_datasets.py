from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

DATASETS = [
    {
        "dataset_id": "GSE126848",
        "counts_path": DATA_DIR / "GSE126848_Gene_counts_raw.txt.gz",
        "metadata_path": DATA_DIR / "GSE126848_series_matrix.txt.gz",
        "symbol_map_path": DATA_DIR / "ensembl_to_symbol.csv",
        "platform": "bulk_rnaseq",
        "disease_id": "DIS_001",
        "context_id": "CTX_001",
    },
    {
        "dataset_id": "GSE_NEW_001",
        "counts_path": DATA_DIR / "GSE_NEW_001_counts.txt.gz",
        "metadata_path": DATA_DIR / "GSE_NEW_001_series_matrix.txt.gz",
        "symbol_map_path": DATA_DIR / "ensembl_to_symbol.csv",
        "platform": "bulk_rnaseq",
        "disease_id": "DIS_001",
        "context_id": "CTX_001",
    },
    {
        "dataset_id": "GSE_NEW_002",
        "counts_path": DATA_DIR / "GSE_NEW_002_counts.txt.gz",
        "metadata_path": DATA_DIR / "GSE_NEW_002_series_matrix.txt.gz",
        "symbol_map_path": DATA_DIR / "ensembl_to_symbol.csv",
        "platform": "bulk_rnaseq",
        "disease_id": "DIS_001",
        "context_id": "CTX_001",
    },
]


def validate_dataset(ds: dict) -> None:
    required = ["dataset_id", "counts_path", "metadata_path", "symbol_map_path"]
    for key in required:
        if key not in ds:
            raise ValueError(f"Missing required key: {key}")

    for path_key in ["counts_path", "metadata_path", "symbol_map_path"]:
        path = Path(ds[path_key])
        if not path.exists():
            raise FileNotFoundError(f"{path_key} not found for {ds['dataset_id']}: {path}")


def run_ingestion(ds: dict) -> None:
    cmd = [
        sys.executable,
        "-m",
        "etl.ingest_bulk",
        "--dataset-id",
        ds["dataset_id"],
        "--counts",
        str(ds["counts_path"]),
        "--metadata",
        str(ds["metadata_path"]),
        "--symbol-map",
        str(ds["symbol_map_path"]),
        "--platform",
        ds.get("platform", "bulk_rnaseq"),
        "--disease-id",
        ds.get("disease_id", "DIS_001"),
        "--context-id",
        ds.get("context_id", "CTX_001"),
    ]

    print(f"\n=== Ingesting {ds['dataset_id']} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    for ds in DATASETS:
        validate_dataset(ds)
        run_ingestion(ds)

    print("\nAll datasets ingested successfully.")
    print("Next run:")
    print("  python -m scripts.build_expression_features")
    print("  python -m models.scoring")
    print("  python -m models.train")


if __name__ == "__main__":
    main()