import math
import re
from urllib.parse import urlparse

import pandas as pd

SUSPICIOUS_KEYWORDS = {
    "login",
    "verify",
    "update",
    "secure",
    "account",
    "free",
    "bonus",
    "claim",
    "wallet",
    "suspend",
    "confirm",
    "password",
    "signin",
    "urgent",
}

IP_PATTERN = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    probabilities = [text.count(ch) / len(text) for ch in set(text)]
    return -sum(p * math.log2(p) for p in probabilities)


def extract_url_features(url: str) -> dict[str, float]:
    clean_url = (url or "").strip()
    parsed = urlparse(clean_url if "://" in clean_url else f"http://{clean_url}")
    hostname = parsed.hostname or ""
    path_and_query = f"{parsed.path}?{parsed.query}" if parsed.query else parsed.path
    lower_url = clean_url.lower()

    keyword_hits = sum(1 for word in SUSPICIOUS_KEYWORDS if word in lower_url)
    has_ip = 1.0 if IP_PATTERN.match(hostname) else 0.0
    uses_https = 1.0 if parsed.scheme.lower() == "https" else 0.0

    return {
        "url_length": float(len(clean_url)),
        "hostname_length": float(len(hostname)),
        "path_length": float(len(path_and_query)),
        "dot_count": float(clean_url.count(".")),
        "hyphen_count": float(clean_url.count("-")),
        "digit_count": float(sum(ch.isdigit() for ch in clean_url)),
        "at_symbol_count": float(clean_url.count("@")),
        "slash_count": float(clean_url.count("/")),
        "has_ip_address": has_ip,
        "uses_https": uses_https,
        "suspicious_keyword_count": float(keyword_hits),
        "url_entropy": round(_shannon_entropy(clean_url), 4),
    }


def build_feature_matrix(urls: list[str] | pd.Series) -> pd.DataFrame:
    rows = [extract_url_features(u) for u in urls]
    return pd.DataFrame(rows)