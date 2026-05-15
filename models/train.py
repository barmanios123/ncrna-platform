"""
Training pipeline for ncRNA prioritization.

Builds a labeled training set by merging user labels from data/training_targets.csv
with platform-derived features from the SQLite database, benchmarks a few models,
writes CV predictions and feature importances, and saves a fitted pipeline.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DB_PATH = Path("ncrna_platform.db")
LABELS_PATH = Path("data/training_targets.csv")
OUTPUT_DIR = Path("output")
MODEL_PATH = OUTPUT_DIR / "training_pipeline.joblib"
METRICS_PATH = OUTPUT_DIR / "training_metrics.json"
BENCHMARK_PATH = OUTPUT_DIR / "model_benchmark.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "training_predictions.csv"
IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
TRAINING_SET_PATH = OUTPUT_DIR / "training_merged_dataset.csv"
METADATA_PATH = OUTPUT_DIR / "training_metadata.json"

DEFAULT_MERGE_KEYS = ["ncrna_id"]
OPTIONAL_GROUP_COLS = ["dataset_id", "group_id", "study_id", "fold_group"]
TARGET_COL = "label"

EXCLUDE_COLS = {
    TARGET_COL,
    "prediction",
    "prediction_cv",
    "residual",
    "residual_cv",
    "rank",
    "split",
    "fold",
    "set",
    "notes",
    "confidence_tier",
}

NON_FEATURE_HINTS = {
    "score_id",
    "top_evidence",
    "risk_flags",
    "recommended_experiments",
    "model_version",
    "gf_model_version",
    "scored_date",
}

CATEGORICAL_CANDIDATES = {
    "symbol",
    "disease_id",
    "context_id",
    "label_source",
    "source",
    "modality",
    "target_class",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_labels(path: Path = LABELS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing labels file: {path}")

    labels = pd.read_csv(path)
    if TARGET_COL not in labels.columns:
        raise ValueError(f"Labels file must contain '{TARGET_COL}' column")

    labels[TARGET_COL] = pd.to_numeric(labels[TARGET_COL], errors="coerce")
    labels = labels.dropna(subset=[TARGET_COL]).copy()

    if labels.empty:
        raise ValueError("No valid labeled rows found in training_targets.csv")

    return labels


def load_target_scores(db_path: Path = DB_PATH) -> pd.DataFrame:
    if not db_path.exists():
        raise FileNotFoundError(f"Missing database: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        ts = pd.read_sql_query("SELECT * FROM target_scores", conn)
        nm = pd.read_sql_query("SELECT ncrna_id, symbol FROM ncrna_master", conn)
    finally:
        conn.close()

    if "symbol" not in ts.columns and "ncrna_id" in ts.columns:
        ts = ts.merge(nm, on="ncrna_id", how="left")

    return ts


def infer_merge_keys(labels: pd.DataFrame, features: pd.DataFrame) -> list[str]:
    keys = [k for k in DEFAULT_MERGE_KEYS if k in labels.columns and k in features.columns]
    if not keys and "symbol" in labels.columns and "symbol" in features.columns:
        keys = ["symbol"]
    if not keys:
        raise ValueError("Could not infer merge keys between labels and target_scores")
    return keys


def merge_training_data(labels: pd.DataFrame, features: pd.DataFrame, merge_keys: list[str]) -> pd.DataFrame:
    feat = features.copy()

    drop_dup_cols = [c for c in labels.columns if c in feat.columns and c not in merge_keys]
    if drop_dup_cols:
        feat = feat.drop(columns=drop_dup_cols)

    merged = labels.merge(feat, on=merge_keys, how="left", validate="one_to_one")

    missing_feature_rows = merged.drop(columns=labels.columns, errors="ignore").isna().all(axis=1).sum()
    if missing_feature_rows > 0:
        print(f"⚠️ Warning: {missing_feature_rows} labeled rows had no matching feature row after merge")

    merged = merged.dropna(subset=[TARGET_COL]).copy()
    return merged


def infer_task_type(train_df: pd.DataFrame) -> str:
    y = pd.to_numeric(train_df[TARGET_COL], errors="coerce").dropna()
    unique_vals = y.nunique()
    if unique_vals <= 10 and set(np.unique(y)).issubset({0, 1}):
        return "binary_classification"
    return "regression"


def get_group_series(train_df: pd.DataFrame) -> pd.Series | None:
    for col in OPTIONAL_GROUP_COLS:
        if col in train_df.columns:
            s = train_df[col].astype(str)
            if s.nunique() >= 2:
                return s
    return None


def choose_feature_columns(train_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    excluded_cols_present = []
    numeric_cols = []
    categorical_cols = []

    for col in train_df.columns:
        if col in EXCLUDE_COLS or col in NON_FEATURE_HINTS:
            excluded_cols_present.append(col)
            continue

        if col in DEFAULT_MERGE_KEYS or col in OPTIONAL_GROUP_COLS:
            excluded_cols_present.append(col)
            continue

        if col.endswith("_id") and col not in {"disease_id", "context_id"}:
            excluded_cols_present.append(col)
            continue

        if col == TARGET_COL:
            excluded_cols_present.append(col)
            continue

        series = train_df[col]

        if col in CATEGORICAL_CANDIDATES:
            categorical_cols.append(col)
        elif pd.api.types.is_bool_dtype(series):
            numeric_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    numeric_cols = sorted(set(numeric_cols))
    categorical_cols = sorted(set(categorical_cols))
    excluded_cols_present = sorted(set(excluded_cols_present))

    return numeric_cols, categorical_cols, excluded_cols_present


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def get_cv(task_type: str, y: np.ndarray, groups: pd.Series | None):
    n = len(y)
    if groups is not None and groups.nunique() >= 2:
        n_splits = min(5, int(groups.nunique()))
        if n_splits >= 2:
            return GroupKFold(n_splits=n_splits), None, "GroupKFold"
    n_splits = min(5, n)
    if n_splits < 2:
        raise ValueError("Need at least 2 rows for cross-validation")
    return KFold(n_splits=n_splits, shuffle=True, random_state=42), None, "KFold"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": corr,
    }


def benchmark_models(
    X: pd.DataFrame,
    y: np.ndarray,
    task_type: str,
    groups: pd.Series | None,
    numeric_cols: list[str],
    categorical_cols: list[str],
):
    if task_type != "regression":
        raise NotImplementedError("This training script currently supports regression labels")

    model_specs = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=1,
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    cv, _, cv_name = get_cv(task_type, y, groups)

    rows = []
    fitted = {}

    for name, model in model_specs.items():
        pre = build_preprocessor(numeric_cols, categorical_cols)
        pipe = Pipeline(
            steps=[
                ("preprocessor", pre),
                ("model", model),
            ]
        )

        cv_out = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            groups=groups if cv_name == "GroupKFold" else None,
            scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"),
            return_train_score=False,
            n_jobs=1,
        )

        mean_r2 = float(np.mean(cv_out["test_r2"]))
        mean_mae = float(-np.mean(cv_out["test_neg_mean_absolute_error"]))
        mean_rmse = float(-np.mean(cv_out["test_neg_root_mean_squared_error"]))

        rows.append(
            {
                "model_name": name,
                "cv_name": cv_name,
                "mean_r2": mean_r2,
                "mean_mae": mean_mae,
                "mean_rmse": mean_rmse,
            }
        )
        fitted[name] = pipe

    benchmark_df = pd.DataFrame(rows).sort_values(
        by=["mean_r2", "mean_mae"], ascending=[False, True]
    ).reset_index(drop=True)

    best_name = benchmark_df.iloc[0]["model_name"]
    best_pipeline = fitted[best_name]

    best_preds = cross_val_predict(
        best_pipeline,
        X,
        y,
        cv=cv,
        groups=groups if cv_name == "GroupKFold" else None,
        method="predict",
        n_jobs=1,
    )
    metrics = regression_metrics(y, best_preds)

    best_pipeline.fit(X, y)

    return best_name, best_pipeline, metrics, benchmark_df, best_preds


def build_prediction_df(train_df: pd.DataFrame, preds: np.ndarray, task_type: str) -> pd.DataFrame:
    out = train_df.copy()
    if task_type != "regression":
        raise NotImplementedError("This training script currently supports regression labels")
    out["prediction_cv"] = preds.astype(float)
    out["residual_cv"] = out[TARGET_COL].astype(float) - out["prediction_cv"]
    sort_cols = [c for c in ["prediction_cv", TARGET_COL] if c in out.columns]
    out = out.sort_values(sort_cols, ascending=[False, False][: len(sort_cols)]).reset_index(drop=True)
    return out


def extract_feature_names(preprocessor: ColumnTransformer, numeric_cols: list[str], categorical_cols: list[str]) -> list[str]:
    feature_names = []

    if numeric_cols:
        feature_names.extend(numeric_cols)

    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(cat_names)

    return feature_names


def extract_feature_importance(
    best_pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    pre = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]

    feature_names = extract_feature_names(pre, numeric_cols, categorical_cols)

    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": vals})
    else:
        result = permutation_importance(
            best_pipeline,
            X,
            y,
            n_repeats=20,
            random_state=42,
            n_jobs=1,
        )
        imp_df = pd.DataFrame(
            {"feature": X.columns.tolist(), "importance": result.importances_mean}
        )

    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df


def save_outputs(
    pipeline: Pipeline,
    metrics: dict,
    benchmark_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    train_df: pd.DataFrame,
    metadata: dict,
) -> None:
    ensure_output_dir()

    joblib.dump(pipeline, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    benchmark_df.to_csv(BENCHMARK_PATH, index=False)
    pred_df.to_csv(PREDICTIONS_PATH, index=False)
    imp_df.to_csv(IMPORTANCE_PATH, index=False)
    train_df.to_csv(TRAINING_SET_PATH, index=False)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main():
    labels = load_labels(LABELS_PATH)
    features = load_target_scores(DB_PATH)

    merge_keys = infer_merge_keys(labels, features)
    print(f"Using merge keys: {merge_keys}")

    train_df = merge_training_data(labels, features, merge_keys)
    print(f"Merged training set shape: {train_df.shape}")

    task_type = infer_task_type(train_df)
    print(f"Inferred task type: {task_type}")

    groups = get_group_series(train_df)
    if groups is not None:
        print(f"Using grouped CV based on column with {groups.nunique()} unique groups")
    else:
        print("No grouping column detected; using standard CV")

    numeric_cols, categorical_cols, excluded_cols_present = choose_feature_columns(train_df)

    print(f"Numeric feature columns: {len(numeric_cols)}")
    print(f"Categorical feature columns: {len(categorical_cols)}")
    print(f"Excluded columns present: {len(excluded_cols_present)}")

    X = train_df[numeric_cols + categorical_cols].copy()
    y = train_df[TARGET_COL].astype(float).values

    print("Benchmarking candidate models...")
    best_name, best_pipeline, metrics, benchmark_df, best_preds = benchmark_models(
        X=X,
        y=y,
        task_type=task_type,
        groups=groups,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    print("\nModel benchmark:")
    print(benchmark_df.to_string(index=False))

    print(f"\nSelected best model: {best_name}")

    pred_df = build_prediction_df(train_df, best_preds, task_type)
    imp_df = extract_feature_importance(best_pipeline, X, y, numeric_cols, categorical_cols)

    if not imp_df.empty:
        print("\nTop feature importances:")
        print(imp_df.head(20).to_string(index=False))

    metadata = {
        "timestamp_utc": utc_now_iso(),
        "db_path": str(DB_PATH),
        "merge_keys": merge_keys,
        "task_type": task_type,
        "n_rows": int(train_df.shape[0]),
        "n_columns": int(train_df.shape[1]),
        "numeric_feature_cols": numeric_cols,
        "categorical_feature_cols": categorical_cols,
        "excluded_cols_present": excluded_cols_present,
        "label_columns_present": labels.columns.tolist(),
        "best_model_name": best_name,
        "best_model_type": type(best_pipeline.named_steps["model"]).__name__,
        "confidence_tier_excluded": True,
    }

    save_outputs(
        pipeline=best_pipeline,
        metrics=metrics,
        benchmark_df=benchmark_df,
        pred_df=pred_df,
        imp_df=imp_df,
        train_df=train_df,
        metadata=metadata,
    )

    print("\nFinal metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()