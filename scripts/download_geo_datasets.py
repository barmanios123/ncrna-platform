"""
Download GEO datasets for the ncRNA platform using curl (fast, native on Mac).
Run from repo root: python scripts/download_geo_datasets.py
"""

import subprocess
import sys
from pathlib import Path

DATA_DIR = Path.home() / "ncrna_platform" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADS = [
    {
        "name": "GSE130970 series matrix",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE130nnn/GSE130970/matrix/GSE130970_series_matrix.txt.gz",
        "dest": DATA_DIR / "GSE130970_series_matrix.txt.gz",
    },
    {
        "name": "GSE135251 series matrix",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE135nnn/GSE135251/matrix/GSE135251_series_matrix.txt.gz",
        "dest": DATA_DIR / "GSE135251_series_matrix.txt.gz",
    },
    {
        "name": "GSE130970 supplementary counts",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE130nnn/GSE130970/suppl/GSE130970_RAW.tar",
        "dest": DATA_DIR / "GSE130970_RAW.tar",
    },
    {
        "name": "GSE135251 supplementary counts",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE135nnn/GSE135251/suppl/GSE135251_RAW.tar",
        "dest": DATA_DIR / "GSE135251_RAW.tar",
    },
]


def download_with_curl(url: str, dest: Path, name: str) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  Already exists: {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return True

    print(f"\nDownloading {name}...")
    print(f"  URL: {url}")
    print(f"  Dest: {dest}")

    cmd = [
        "curl",
        "--progress-bar",
        "--retry", "3",
        "--retry-delay", "5",
        "--connect-timeout", "30",
        "-L",
        "-o", str(dest),
        url,
    ]

    result = subprocess.run(cmd)
    if result.returncode == 0 and dest.exists():
        print(f"  Done: {dest.stat().st_size / 1024 / 1024:.1f} MB")
        return True
    else:
        print(f"  FAILED (exit code {result.returncode})")
        if dest.exists():
            dest.unlink()
        return False


def main():
    print(f"Saving files to: {DATA_DIR}\n")
    results = []
    for d in DOWNLOADS:
        ok = download_with_curl(d["url"], d["dest"], d["name"])
        results.append((d["name"], ok))

    print("\n=== Download summary ===")
    for name, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"  {status}: {name}")

    failed = [n for n, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)} downloads failed. Re-run the script to retry.")
        sys.exit(1)
    else:
        print("\nAll downloads complete.")
        print("\nNext steps:")
        print("  python -m scripts.build_expression_features")
        print("  python -m models.scoring")
        print("  python -m models.train")


if __name__ == "__main__":
    main()