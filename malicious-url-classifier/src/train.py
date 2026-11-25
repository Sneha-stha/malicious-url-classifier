import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import build_feature_matrix


def train_and_evaluate(data_path: Path, model_out: Path, test_size: float = 0.25) -> None:
    df = pd.read_csv(data_path).dropna(subset=["url", "label"])
    X = build_feature_matrix(df["url"])
    y = df["label"].astype(int)

    stratify_target = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_target
    )

    candidates = {
        "LogisticRegression (Baseline)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=150, max_depth=8, random_state=42
        ),
    }

    best_name = ""
    best_model = None
    best_f1 = -1.0

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)

        print(f"\n=== {name} ===")
        print("Confusion Matrix (rows=actual, cols=predicted):")
        print(confusion_matrix(y_test, preds))
        print(classification_report(y_test, preds, digits=3, zero_division=0))

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_name = name
            best_model = model

    if hasattr(best_model, "named_steps") and "clf" in best_model.named_steps:
        estimator = best_model.named_steps["clf"]
    else:
        estimator = best_model

    if hasattr(estimator, "feature_importances_"):
        importances = pd.Series(estimator.feature_importances_, index=X.columns).sort_values(
            ascending=False
        )
        print("Top Feature Importances:")
        print(importances.head(6).to_string())

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "feature_names": list(X.columns)}, model_out)
    print(f"\nSaved best model ({best_name}, macro F1={best_f1:.3f}) to {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Malicious URL Classifier.")
    parser.add_argument("--data", type=Path, default=Path("data/sample_urls.csv"))
    parser.add_argument("--out", type=Path, default=Path("models/url_classifier.joblib"))
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()
    train_and_evaluate(args.data, args.out, args.test_size)