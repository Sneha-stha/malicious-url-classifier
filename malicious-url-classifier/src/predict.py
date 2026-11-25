import argparse
from pathlib import Path

import joblib

from features import build_feature_matrix, extract_url_features


def predict_url(url: str, model_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Run 'python src/train.py' first."
        )

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    X = build_feature_matrix([url])[feature_names]
    pred = int(model.predict(X)[0])
    label_text = "MALICIOUS / SUSPICIOUS" if pred == 1 else "BENIGN"

    confidence_str = ""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence_str = f" (confidence: {proba[pred]:.1%})"

    raw_feats = extract_url_features(url)
    print(f"URL        : {url}")
    print(f"Prediction : {label_text}{confidence_str}")
    print(
        "Key Signals: "
        f"length={int(raw_feats['url_length'])}, "
        f"dots={int(raw_feats['dot_count'])}, "
        f"hyphens={int(raw_feats['hyphen_count'])}, "
        f"https={int(raw_feats['uses_https'])}, "
        f"keywords={int(raw_feats['suspicious_keyword_count'])}, "
        f"entropy={raw_feats['url_entropy']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a URL as benign or malicious.")
    parser.add_argument("url", type=str, help="Target URL to inspect")
    parser.add_argument("--model", type=Path, default=Path("models/url_classifier.joblib"))
    args = parser.parse_args()
    predict_url(args.url, args.model)