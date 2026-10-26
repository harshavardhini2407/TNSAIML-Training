import sys
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train_model import load_dataset, evaluate

def test_load_dataset_accepts_class_target(tmp_path):
    csv = tmp_path / "toy.csv"
    pd.DataFrame({
        "Amount": [1, 2, 3, 4, 5, 6, 7, 8],
        "V1": [0.1, 0.2, 0.3, 0.4, 1, 1.1, 1.2, 1.3],
        "Class": [0, 0, 0, 0, 1, 1, 1, 1],
    }).to_csv(csv, index=False)
    X, y, target = load_dataset(csv)
    assert target == "Class"
    assert list(X.columns) == ["Amount", "V1"]
    assert y.sum() == 4

def test_evaluate_returns_fraud_metrics():
    X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5, 6, 7]})
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ])
    model.fit(X, y)
    metrics = evaluate(model, X, y)
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert len(metrics["confusion_matrix"]) == 2
