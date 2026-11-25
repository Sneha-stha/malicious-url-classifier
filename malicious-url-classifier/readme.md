# Malicious URL Classifier

A modular Python and `scikit-learn` project that classifies URLs as **benign** or **malicious/phishing** using lexical, structural, and entropy-based URL features.

## Project Overview
- **Goal:** Detect suspicious or phishing URLs from structural URL attributes without needing to visit the live website first.
- **Approach:** Extract domain/path/character features in `src/features.py`, train and compare baseline (`LogisticRegression`) vs. ensemble (`RandomForestClassifier`) models in `src/train.py`, and run single-URL inference in `src/predict.py`.

## Setup

```bash
git clone <your-repo-url>
cd malicious-url-classifier
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt