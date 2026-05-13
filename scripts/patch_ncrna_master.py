import sqlite3
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "ncrna_platform.db"


MISSING_ROWS = [
    {
        "ncrna_id": "LNCRNA_011",
        "symbol": "H19",
        "biotype": "lncRNA",
        "ensembl_id": None,
        "aliases": "H19 imprinted maternally expressed transcript",
        "species": "human",
        "chromosome": None,
        "start_pos": None,
        "end_pos": None,
        "strand": None,
        "conservation_score": 0.70,
        "transcript_count": 1,
    },
    {
        "ncrna_id": "LNCRNA_012",
        "symbol": "MEG3",
        "biotype": "lncRNA",
        "ensembl_id": None,
        "aliases": "maternally expressed 3",
        "species": "human",
        "chromosome": None,
        "start_pos": None,
        "end_pos": None,
        "strand": None,
        "conservation_score": 0.72,
        "transcript_count": 1,
    },
]


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    existing = pd.read_sql_query("SELECT * FROM ncrna_master", conn)

    existing_cols = existing.columns.tolist()
    required_min = {"ncrna_id", "symbol"}
    missing_min = required_min - set(existing_cols)
    if missing_min:
        raise ValueError(f"ncrna_master is missing required minimum columns: {missing_min}")

    print("ncrna_master columns:")
    print(existing_cols)

    for row in MISSING_ROWS:
        symbol = row["symbol"]
        found = existing["symbol"].astype(str).str.upper().eq(symbol.upper()).any()

        if found:
            print(f"Skipping {symbol}: already present")
            continue

        filtered_row = {k: v for k, v in row.items() if k in existing_cols}

        insert_df = pd.DataFrame([filtered_row])

        for col in existing_cols:
            if col not in insert_df.columns:
                insert_df[col] = None

        insert_df = insert_df[existing_cols]
        insert_df.to_sql("ncrna_master", conn, if_exists="append", index=False)

        print(f"Inserted {symbol}")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()