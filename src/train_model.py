"""Train and evaluate classification models for credit-card fraud detection."""
from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_PATH = ROOT / "models" / "fraud_pipeline.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
TARGET_CANDIDATES = ["Class", "class", "Fraud", "fraud", "is_fraud", "target"]

def load_dataset(path):
    df = pd.read_csv(path)
    target = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target is None:
        raise ValueError(f"No target column found. Expected one of: {TARGET_CANDIDATES}")
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.isin([0, 1])
    df, y = df.loc[valid].copy(), y.loc[valid].astype(int)
    X = df.drop(columns=[target])
    # Remove obvious identifier columns that would not generalize.
    drop = [c for c in X.columns if c.lower() in {"id", "transaction_id", "trans_id"}]
    X = X.drop(columns=drop)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found.")
    if y.nunique() != 2:
        raise ValueError("Target must contain both classes 0 and 1.")
    return X, y, target

def build_models(seed=42):
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=250, class_weight="balanced_subsample",
                random_state=seed, n_jobs=-1, min_samples_leaf=2
            ))
        ]),
    }

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)] if hasattr(model, "predict_proba") else None
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if proba is not None else None,
        "average_precision": float(average_precision_score(y_test, proba)) if proba is not None else None,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0,1]).tolist(),
        "classification_report": classification_report(y_test, pred, labels=[0,1], target_names=["Legitimate","Fraud"], output_dict=True, zero_division=0)
    }

def train(data_path, test_size=0.2, seed=42):
    X, y, target = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    results = {}
    fitted = {}
    for name, model in build_models(seed).items():
        print(f"Training {name}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        fitted[name] = model
        print(f"  precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
              f"f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f}")
    # Select by F1; review recall/precision trade-off before operational use.
    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = fitted[best_name]
    bundle = {
        "model_name": best_name,
        "pipeline": best_model,
        "feature_names": list(X.columns),
        "feature_defaults": {c: float(X_train[c].median()) if X_train[c].notna().any() else 0.0 for c in X.columns},
        "threshold": 0.5,
        "target_column": target,
        "classes": [int(v) for v in best_model.classes_],
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    report = {
        "model_name": best_name,
        "dataset": Path(data_path).name,
        "target_column": target,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "class_counts": {str(int(k)): int(v) for k,v in y.value_counts().sort_index().items()},
        "models": results,
        **results[best_name],
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nBest model by F1: {best_name}")
    print(f"Saved pipeline: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to input CSV")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not Path(args.data).exists():
        raise SystemExit(
            f"Dataset not found: {args.data}\n"
            "Download a credit-card fraud CSV and place it at data/creditcard.csv, "
            "or pass --data path/to/file.csv."
        )
    train(args.data, args.test_size, args.seed)

if __name__ == "__main__":
    main()
