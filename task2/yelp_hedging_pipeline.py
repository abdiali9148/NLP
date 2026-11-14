#!/usr/bin/env python3
"""Reproducible Yelp hedging/certainty feature extraction.

Dependencies: pandas >=2.0, pyarrow >=14.0.
Actual Yelp data and vocabulary lists must be provided by the user.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import pandas as pd

SPLITS = ("train", "dev", "test")
EXPECTED_ROWS = {"train": 398_664, "dev": 51_897, "test": 47_184}
EXCLUDED = {"would", "could"}
APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'"})


def normalize_text(text: object) -> str:
    """Apply exactly the stipulated lowercase and curly-apostrophe rules."""
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return ""
    if not isinstance(text, str):
        raise TypeError(f"Review text must be a string or null, got {type(text)!r}")
    return text.lower().translate(APOSTROPHES)


def load_terms(path: Path) -> list[str]:
    """Read one word or phrase per line; ignore blank and comment lines."""
    if not path.is_file():
        raise FileNotFoundError(f"Word list not found: {path}")
    items: set[str] = set()
    for original in path.read_text(encoding="utf-8-sig").splitlines():
        line = normalize_text(original.strip())
        if line and not line.startswith("#") and line not in EXCLUDED:
            items.add(line)
    if not items:
        raise ValueError(f"No usable terms in {path}")
    # Longest first resolves overlapping single-token and multi-token phrases.
    return sorted(items, key=lambda term: (-len(term), term))


def compile_matcher(terms: list[str]) -> re.Pattern[str]:
    normalized = sorted(
        {normalize_text(t.strip()) for t in terms if t and normalize_text(t.strip()) not in EXCLUDED},
        key=lambda term: (-len(term), term),
    )
    if not normalized:
        raise ValueError("Matching requires at least one nonexcluded term")
    # \b around re.escape(term) is the exact boundary contract.
    return re.compile(r"\b(?:" + "|".join(re.escape(term) for term in normalized) + r")\b")


def extract_features(
    text_column: pd.Series,
    hedge_terms: list[str],
    certainty_terms: list[str],
) -> tuple[pd.DataFrame, Counter[str], Counter[str], list[list[str]], list[list[str]]]:
    """Count nonoverlapping longest-first term matches within each vocabulary.

    Returns count dataframe (same row order), term counters, and per-row hit lists.
    Lists are retained for reproducible manual checking and term summaries.
    """
    hedges_re = compile_matcher(hedge_terms)
    certainty_re = compile_matcher(certainty_terms)
    hedge_total: Counter[str] = Counter()
    certainty_total: Counter[str] = Counter()
    hedge_hits: list[list[str]] = []
    certainty_hits: list[list[str]] = []
    hedge_counts: list[int] = []
    certainty_counts: list[int] = []
    for raw_text in text_column:
        text = normalize_text(raw_text)
        h = [match.group(0) for match in hedges_re.finditer(text)]
        c = [match.group(0) for match in certainty_re.finditer(text)]
        hedge_hits.append(h)
        certainty_hits.append(c)
        hedge_counts.append(len(h))
        certainty_counts.append(len(c))
        hedge_total.update(h)
        certainty_total.update(c)
    counts = pd.DataFrame(
        {"hedge_count": hedge_counts, "certainty_count": certainty_counts},
        index=text_column.index,
    )
    return counts, hedge_total, certainty_total, hedge_hits, certainty_hits


def load_reviews(path: Path, text_col: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Review text file not found: {path}")
    columns = ["review_id", text_col]
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix in (".jsonl", ".json"):
        return pd.read_json(path, lines=True)[columns]
    if path.suffix == ".csv":
        return pd.read_csv(path, usecols=columns)
    raise ValueError(f"Unsupported text input: {path}. Use .parquet, .jsonl, or .csv")


def load_merged(features_path: Path, reviews_path: Path, text_col: str, split: str) -> pd.DataFrame:
    if not features_path.is_file():
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    features = pd.read_parquet(features_path, columns=["review_id", "n_sents"])
    reviews = load_reviews(reviews_path, text_col)
    if features.review_id.isna().any() or reviews.review_id.isna().any():
        raise ValueError(f"{split}: null review_id detected")
    if features.review_id.duplicated().any() or reviews.review_id.duplicated().any():
        raise ValueError(f"{split}: duplicate review_id detected")
    if len(features) != EXPECTED_ROWS[split]:
        raise ValueError(f"{split}: expected {EXPECTED_ROWS[split]:,} rows, got {len(features):,}")
    frame = features.merge(reviews, on="review_id", how="left", sort=False,
                           validate="one_to_one", indicator=True)
    if (frame["_merge"] != "both").any():
        missing = frame.loc[frame["_merge"] != "both", "review_id"].head(5).tolist()
        raise ValueError(f"{split}: missing review texts for IDs: {missing}")
    frame = frame.drop(columns="_merge")
    if not pd.api.types.is_numeric_dtype(frame["n_sents"]):
        raise ValueError(f"{split}: n_sents is not numeric")
    if frame["n_sents"].isna().any() or (frame["n_sents"] <= 0).any():
        bad = int(((frame["n_sents"] <= 0) | frame["n_sents"].isna()).sum())
        raise ValueError(f"{split}: {bad} rows have null/zero/negative n_sents; cannot divide by 0")
    return frame


def spotcheck_file(
    frame: pd.DataFrame,
    hedge_hits: list[list[str]],
    certainty_hits: list[list[str]],
    path: Path,
    n: int,
    seed: int,
    text_col: str,
) -> None:
    check = frame[["review_id", "n_sents", text_col]].copy()
    check["hedge_matches"] = [", ".join(h) for h in hedge_hits]
    check["certainty_matches"] = [", ".join(c) for c in certainty_hits]
    check["hedge_count"] = [len(h) for h in hedge_hits]
    check["certainty_count"] = [len(c) for c in certainty_hits]
    positive = check.loc[(check.hedge_count > 0) | (check.certainty_count > 0)]
    negative = check.loc[(check.hedge_count == 0) & (check.certainty_count == 0)]
    # Deliberately oversample positive cases to make manual inspection efficient.
    positive_n = min(len(positive), round(0.75 * n))
    negative_n = min(len(negative), n - positive_n)
    sample = pd.concat([
        positive.sample(n=positive_n, random_state=seed) if positive_n else positive.iloc[:0],
        negative.sample(n=negative_n, random_state=seed) if negative_n else negative.iloc[:0],
    ], ignore_index=True)
    if len(sample) < n:
        rest = check.loc[~check.review_id.isin(sample.review_id)]
        sample = pd.concat([sample, rest.sample(n=min(n - len(sample), len(rest)), random_state=seed)], ignore_index=True)
    sample.to_csv(path, index=False, encoding="utf-8")


def summary(frame: pd.DataFrame, hedge_terms: Counter[str], certainty_terms: Counter[str]) -> dict:
    rows = len(frame)
    all_terms = hedge_terms + certainty_terms
    return {
        "rows": rows,
        "hedge_rate_mean": float(frame.hedge_rate.mean()),
        "hedge_rate_median": float(frame.hedge_rate.median()),
        "certainty_rate_mean": float(frame.certainty_rate.mean()),
        "certainty_rate_median": float(frame.certainty_rate.median()),
        "zero_hedge_pct": 100 * float((frame.hedge_count == 0).sum()) / rows,
        "zero_certainty_pct": 100 * float((frame.certainty_count == 0).sum()) / rows,
        "zero_both_pct": 100 * float(((frame.hedge_count + frame.certainty_count) == 0).sum()) / rows,
        "top_10_hedges": hedge_terms.most_common(10),
        "top_10_certainty": certainty_terms.most_common(10),
        "top_10_overall": all_terms.most_common(10),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--reviews-dir", type=Path, help="Default: features-dir")
    parser.add_argument("--out-dir", type=Path, help="Default: features-dir")
    parser.add_argument("--hedges", type=Path, required=True)
    parser.add_argument("--certainty", type=Path, required=True)
    parser.add_argument("--features-pattern", default="features_{split}.parquet")
    parser.add_argument("--reviews-pattern", default="reviews_{split}.parquet")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--pilot-only", action="store_true", help="Analyze only a 300-row pilot per split; no final parquet output")
    parser.add_argument("--pilot-size", type=int, default=300)
    parser.add_argument("--audit-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.pilot_size < 1 or args.audit_size < 1:
        parser.error("--pilot-size and --audit-size must be positive")
    out_dir = args.out_dir or args.features_dir
    reviews_dir = args.reviews_dir or args.features_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    hedges = load_terms(args.hedges)
    certainty = load_terms(args.certainty)
    print(f"Vocabulary: {len(hedges)} hedges, {len(certainty)} certainty terms; would/could excluded")
    overlapping = sorted(set(hedges).intersection(certainty))
    if overlapping:
        print(f"WARNING: terms appear in BOTH lists: {overlapping}")
    report: dict[str, dict] = {}
    for split in SPLITS:
        features_path = args.features_dir / args.features_pattern.format(split=split)
        reviews_path = reviews_dir / args.reviews_pattern.format(split=split)
        frame = load_merged(features_path, reviews_path, args.text_col, split)
        if args.pilot_only:
            frame = frame.sample(n=min(args.pilot_size, len(frame)), random_state=args.seed).reset_index(drop=True)
        counts, h_counter, c_counter, h_hits, c_hits = extract_features(frame[args.text_col], hedges, certainty)
        frame["hedge_count"] = counts["hedge_count"].to_numpy()
        frame["certainty_count"] = counts["certainty_count"].to_numpy()
        frame["hedge_rate"] = frame["hedge_count"] / frame["n_sents"]
        frame["certainty_rate"] = frame["certainty_count"] / frame["n_sents"]
        report[split] = summary(frame, h_counter, c_counter)
        prefix = "pilot" if args.pilot_only else "spotcheck"
        audit_path = out_dir / f"{prefix}_matches_{split}.csv"
        spotcheck_file(frame, h_hits, c_hits, audit_path, min(args.audit_size, len(frame)), args.seed, args.text_col)
        if not args.pilot_only:
            final = frame[["review_id", "hedge_count", "certainty_count", "hedge_rate", "certainty_rate"]]
            assert len(final) == EXPECTED_ROWS[split]
            assert final.review_id.equals(frame.review_id)
            out_path = out_dir / f"hedging_features_{split}.parquet"
            final.to_parquet(out_path, index=False, engine="pyarrow")
            reread = pd.read_parquet(out_path, columns=["review_id"])
            if not reread.review_id.equals(frame.review_id):
                raise AssertionError(f"{split}: output review IDs changed during parquet write")
            print(f"Wrote {out_path} ({len(final):,} rows)")
        print(f"{split}: {json.dumps(report[split])}")
        print(f"Manual audit sample (NOT yet manually verified): {audit_path}")
    report_name = "hedging_pilot_summary.json" if args.pilot_only else "hedging_summary.json"
    (out_dir / report_name).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Summary: {out_dir / report_name}")
    print("IMPORTANT: Review spotcheck/pilot CSV rows against source text manually before sharing final deliverables.")


if __name__ == "__main__":
    main()
