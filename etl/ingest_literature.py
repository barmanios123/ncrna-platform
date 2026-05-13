"""
Step 1: ingest curated liver ncRNA literature evidence
Run:
    python etl/ingest_literature.py
"""

import os
import json
import sqlite3
import pandas as pd

DB_PATH = "ncrna_platform.db"
CURATED_CSV = "data/external/curated_targets.csv"
OUT_JSON = "db/literature_seed.json"
OUT_SUMMARY = "outputs/tables/curated_target_summary.csv"


def ensure_dirs():
    os.makedirs("db", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)


def connect_db(db_path=DB_PATH):
    return sqlite3.connect(db_path)


def ensure_tables(conn):
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS curated_targets (
        curated_id           TEXT PRIMARY KEY,
        symbol               TEXT NOT NULL,
        aliases              TEXT,
        disease_stage        TEXT,
        disease_label        TEXT,
        evidence_tier        TEXT,
        direction            TEXT,
        mechanism_note       TEXT,
        modality_bias        TEXT,
        cell_context         TEXT,
        source_type          TEXT,
        source_ref           TEXT,
        pmid                 TEXT,
        is_contradictory     INTEGER DEFAULT 0,
        curator_note         TEXT,
        added_date           TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS literature_evidence_v2 (
        lit_id               TEXT PRIMARY KEY,
        symbol               TEXT NOT NULL,
        disease_label        TEXT,
        disease_stage        TEXT,
        evidence_tier        TEXT,
        direction            TEXT,
        mechanism_note       TEXT,
        source_type          TEXT,
        source_ref           TEXT,
        pmid                 TEXT,
        is_contradictory     INTEGER DEFAULT 0,
        curator_note         TEXT,
        added_date           TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()


def load_curated_targets(csv_path=CURATED_CSV):
    df = pd.read_csv(csv_path).fillna("")
    expected_cols = [
        "symbol", "aliases", "disease_stage", "disease_label", "evidence_tier",
        "direction", "mechanism_note", "modality_bias", "cell_context",
        "source_type", "source_ref", "pmid", "is_contradictory", "curator_note"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in curated_targets.csv: {missing}")
    return df


def write_curated_tables(conn, df):
    c = conn.cursor()

    curated_rows = []
    lit_rows = []

    for i, row in df.iterrows():
        curated_id = f"CUR_{i+1:03d}"
        lit_id = f"LIT2_{i+1:03d}"

        curated_rows.append((
            curated_id,
            row["symbol"],
            row["aliases"],
            row["disease_stage"],
            row["disease_label"],
            row["evidence_tier"],
            row["direction"],
            row["mechanism_note"],
            row["modality_bias"],
            row["cell_context"],
            row["source_type"],
            row["source_ref"],
            str(row["pmid"]),
            int(row["is_contradictory"]),
            row["curator_note"],
        ))

        lit_rows.append((
            lit_id,
            row["symbol"],
            row["disease_label"],
            row["disease_stage"],
            row["evidence_tier"],
            row["direction"],
            row["mechanism_note"],
            row["source_type"],
            row["source_ref"],
            str(row["pmid"]),
            int(row["is_contradictory"]),
            row["curator_note"],
        ))

    c.executemany("""
        INSERT OR REPLACE INTO curated_targets
        (curated_id, symbol, aliases, disease_stage, disease_label, evidence_tier,
         direction, mechanism_note, modality_bias, cell_context, source_type,
         source_ref, pmid, is_contradictory, curator_note)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, curated_rows)

    c.executemany("""
        INSERT OR REPLACE INTO literature_evidence_v2
        (lit_id, symbol, disease_label, disease_stage, evidence_tier,
         direction, mechanism_note, source_type, source_ref, pmid,
         is_contradictory, curator_note)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, lit_rows)

    conn.commit()


def export_outputs(df):
    ensure_dirs()

    df.to_csv(OUT_SUMMARY, index=False)

    records = df.to_dict(orient="records")
    with open(OUT_JSON, "w") as f:
        json.dump(records, f, indent=2)

    print(f"✅ Wrote summary CSV: {OUT_SUMMARY}")
    print(f"✅ Wrote JSON seed: {OUT_JSON}")


def print_summary(df):
    print("\n── Curated Target Summary ──")
    print(df[[
        "symbol", "disease_stage", "disease_label",
        "evidence_tier", "direction", "is_contradictory"
    ]].to_string(index=False))

    print("\nCounts by evidence tier:")
    print(df["evidence_tier"].value_counts(dropna=False).to_string())

    print("\nCounts by disease stage:")
    print(df["disease_stage"].value_counts(dropna=False).to_string())


def main():
    ensure_dirs()
    df = load_curated_targets(CURATED_CSV)

    conn = connect_db(DB_PATH)
    ensure_tables(conn)
    write_curated_tables(conn, df)
    conn.close()

    export_outputs(df)
    print_summary(df)
    print(f"\n✅ Loaded {len(df)} curated liver ncRNA evidence records into {DB_PATH}")


if __name__ == "__main__":
    main()