import sqlite3
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "ncrna_platform.db"

REQUIRED_TABLES = [
    "ncrna_master",
    "target_scores",
    "curated_targets",
]

def check_table_exists(conn, table_name):
    q = """
    SELECT name
    FROM sqlite_master
    WHERE type='table' AND name=?
    """
    return pd.read_sql_query(q, conn, params=[table_name]).shape[0] > 0

def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    print("\n=== TABLE PRESENCE ===")
    for table in REQUIRED_TABLES:
        exists = check_table_exists(conn, table)
        print(f"{table}: {'OK' if exists else 'MISSING'}")

    print("\n=== target_scores summary ===")
    ts = pd.read_sql_query("SELECT * FROM target_scores", conn)
    print(f"rows={len(ts)} cols={len(ts.columns)}")
    print(ts[[
        "ncrna_id", "disease_id", "context_id", "confidence_tier",
        "translational_score", "relevance_score", "mechanism_score",
        "human_evidence_score", "risk_score"
    ]].head(10).to_string(index=False))

    print("\nMissing values by column:")
    print(ts.isna().sum().sort_values(ascending=False).head(20).to_string())

    print("\nDuplicate row counts by key:")
    dup_n = ts.duplicated(subset=["ncrna_id", "disease_id", "context_id"]).sum()
    print(f"duplicate target score keys = {dup_n}")

    print("\nScore range checks:")
    score_cols = [
        "translational_score", "relevance_score", "specificity_score",
        "mechanism_score", "tractability_score", "human_evidence_score", "risk_score"
    ]
    for col in score_cols:
        if col in ts.columns:
            vals = pd.to_numeric(ts[col], errors="coerce")
            print(f"{col}: min={vals.min()} max={vals.max()} nulls={vals.isna().sum()}")

    print("\n=== curated_targets summary ===")
    try:
        ct = pd.read_sql_query("SELECT * FROM curated_targets", conn)
        print(f"rows={len(ct)} cols={len(ct.columns)}")
        if "symbol" in ct.columns:
            print("top symbols in curated_targets:")
            print(ct["symbol"].value_counts().head(10).to_string())
        if "is_contradictory" in ct.columns:
            print(f"contradictory rows = {ct['is_contradictory'].fillna(0).astype(float).sum()}")
    except Exception as e:
        print(f"Could not read curated_targets: {e}")
        ct = pd.DataFrame()

    print("\n=== join integrity ===")
    join_df = pd.read_sql_query(
        """
        SELECT
            ts.ncrna_id,
            ts.disease_id,
            ts.context_id,
            nm.symbol
        FROM target_scores ts
        LEFT JOIN ncrna_master nm
            ON ts.ncrna_id = nm.ncrna_id
        """,
        conn,
    )
    missing_symbols = join_df["symbol"].isna().sum()
    print(f"target_scores rows missing ncrna_master symbol after join = {missing_symbols}")

    if not ct.empty and "symbol" in ct.columns:
        ts_symbols = set(join_df["symbol"].dropna().astype(str).unique())
        ct_symbols = set(ct["symbol"].dropna().astype(str).unique())
        unmatched_curated = sorted(ct_symbols - ts_symbols)
        print(f"curated symbols not present in target_scores join = {len(unmatched_curated)}")
        print(unmatched_curated[:20])

    conn.close()
    print("\nValidation complete.")

if __name__ == "__main__":
    main()