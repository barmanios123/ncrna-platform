import sqlite3
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "ncrna_platform.db"


def precision_recall_at_k(df, score_col, label_col, ks=(3, 5, 10)):
    rows = []
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    positives = int(df[label_col].sum())
    positives = max(positives, 1)

    for k in ks:
        top = ranked.head(k)
        precision = float(top[label_col].mean()) if len(top) else 0.0
        recall = float(top[label_col].sum()) / positives
        rows.append({
            "model": score_col,
            "k": k,
            "precision_at_k": round(precision, 3),
            "recall_at_k": round(recall, 3),
        })
    return pd.DataFrame(rows)


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    ts = pd.read_sql_query("SELECT * FROM target_scores", conn)
    nm = pd.read_sql_query("SELECT ncrna_id, symbol FROM ncrna_master", conn)
    ct = pd.read_sql_query("SELECT DISTINCT symbol FROM curated_targets", conn)
    conn.close()

    df = ts.merge(nm, on="ncrna_id", how="left")
    df["is_curated_positive"] = df["symbol"].isin(ct["symbol"]).astype(int)

    if "literature_count" not in df.columns:
        df["literature_count"] = 0

    frames = [
        precision_recall_at_k(df, "translational_score", "is_curated_positive"),
        precision_recall_at_k(df, "relevance_score", "is_curated_positive"),
        precision_recall_at_k(df, "literature_count", "is_curated_positive"),
    ]
    out = pd.concat(frames, ignore_index=True)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()