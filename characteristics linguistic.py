
#!/usr/bin/env python3
"""
analyze_core_features.py

Compute core linguistic features that distinguish discriminative (label 1) from non-discriminative (label 0) sentences in the probing_records_TEXTONLY.csv dataset.

Features extracted per sentence
[1] spatial_terms        – count of locative words (left, right, etc.)
[2] article_count        – total definite + indefinite articles (the, a, an)
[3] colour_words         – count of colour adjectives
[4] token_count          – sentence length in tokens
[5] ttr                  – type-token ratio (richness)

The script outputs a CSV with mean values per label and percentage difference.

USAGE
-----
python3 analyze_core_features.py --input probing_records_TEXTONLY.csv --output core_features_summary.csv
"""
import argparse
import pandas as pd
import re

SPATIALS = [
    "left", "right", "top", "bottom", "front", "back", "center", "middle",
    "near", "above", "below", "behind", "next", "between", "under", "over"
]
COLOURS = [
    "red", "blue", "green", "yellow", "pink", "black", "white", "grey", "gray",
    "brown", "orange", "purple", "gold", "silver", "violet", "cyan"
]
ARTICLES = ["the", "a", "an"]

def extract_features(text: str) -> dict[str, float]:
    """Return core linguistic feature counts for a single sentence."""
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    token_set = set(tokens)
    return {
        "spatial_terms": sum(tokens.count(w) for w in SPATIALS),
        "article_count": sum(tokens.count(a) for a in ARTICLES),
        "colour_words": sum(tokens.count(c) for c in COLOURS),
        "token_count": len(tokens),
        "ttr": len(token_set) / len(tokens) if tokens else 0.0,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute core linguistic features summary.")
    parser.add_argument("--input", "-i", default="probing_records_TEXTONLY.csv", help="Input CSV with columns y, txt")
    parser.add_argument("--output", "-o", default="core_features_summary.csv", help="Output CSV path for feature summary")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    features_df = df["txt"].apply(extract_features).apply(pd.Series)
    df_feat = pd.concat([df[["y"]], features_df], axis=1)

    cols = ["spatial_terms", "ttr", "article_count", "colour_words", "token_count"]
    summary = df_feat.groupby("y")[cols].mean().T
    summary.columns = ["label0_mean", "label1_mean"]
    summary["percent_difference"] = (
        (summary["label1_mean"] - summary["label0_mean"]) / summary["label0_mean"] * 100
    )
    summary.reset_index(inplace=True)
    summary.rename(columns={"index": "feature"}, inplace=True)

    summary.to_csv(args.output, index=False)
    print(f"[✓] Core feature summary written to {args.output}")


if __name__ == "__main__":
    main()
