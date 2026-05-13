import sqlite3
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "ncrna_platform.db"


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    nm = pd.read_sql_query("SELECT * FROM ncrna_master", conn)
    ts = pd.read_sql_query("SELECT * FROM target_scores", conn)
    ct = pd.read_sql_query("SELECT * FROM curated_targets", conn)

    if "symbol" not in nm.columns:
        raise ValueError("ncrna_master must contain a symbol column")
    if "symbol" not in ct.columns:
        raise ValueError("curated_targets must contain a symbol column")

    nm["symbol_norm"] = nm["symbol"].astype(str).str.strip().str.upper()
    ct["symbol_norm"] = ct["symbol"].astype(str).str.strip().str.upper()

    ts_join = ts.merge(
        nm[["ncrna_id", "symbol", "symbol_norm"]],
        on="ncrna_id",
        how="left"
    )

    ts_symbols = set(ts_join["symbol_norm"].dropna().tolist())
    curated_symbols = set(ct["symbol_norm"].dropna().tolist())

    unmatched = sorted(curated_symbols - ts_symbols)

    print("\n=== CURATED SYMBOLS NOT IN SCORED UNIVERSE ===")
    if unmatched:
        for sym in unmatched:
            print(sym)
    else:
        print("None")

    print("\n=== MATCH CANDIDATES IN NCRNA_MASTER ===")
    for sym in unmatched:
        candidates = nm[
            (nm["symbol_norm"] == sym) |
            (nm["symbol"].astype(str).str.contains(sym, case=False, na=False))
        ].copy()

        print(f"\n[{sym}]")
        if candidates.empty:
            print("  No candidates found in ncrna_master")
        else:
            cols = [c for c in ["ncrna_id", "symbol", "biotype"] if c in candidates.columns]
            print(candidates[cols].drop_duplicates().to_string(index=False))

    print("\n=== CHECK WHETHER SYMBOL EXISTS IN NCRNA_MASTER BUT WAS NOT SCORED ===")
    all_master_symbols = set(nm["symbol_norm"].dropna().tolist())

    for sym in unmatched:
        if sym in all_master_symbols:
            matched_rows = nm[nm["symbol_norm"] == sym]
            print(f"{sym}: PRESENT in ncrna_master but absent from target_scores")
            print(matched_rows[[c for c in ['ncrna_id', 'symbol'] if c in matched_rows.columns]].to_string(index=False))
        else:
            print(f"{sym}: NOT present in ncrna_master")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()