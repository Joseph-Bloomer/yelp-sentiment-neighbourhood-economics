"""Stage 4a — VADER lexicon sentiment, one compound score per review.

Runs on the laptop: VADER is rule-based and trivial per review, so CPU handles
the full corpus. I keep the compound score ([-1, 1]); Stage 5 averages it per
tract. The US corpus is ~6.88M rows (~3 GB of text), so a full run streams:
read in batches, score, write straight to disk, keeping only compact
(compound, stars) arrays for validation. --smoke runs the same scoring on ~12
reviews in memory and writes scores_vader_smoke.parquet, never the real file.

Output: data/processed/scores_vader.parquet  (review_id, vader_compound)

Run:        .venv\\Scripts\\python.exe -m pipeline.stage4a_vader
Smoke test: add --smoke
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config
from pipeline.stage4_common import (
    CORPUS_PARQUET,
    FULL_CORPUS_PARQUET,
    _window_mask,
    load_scoring_corpus,
    out_path,
)

OUT_STEM = "scores_vader"
FIG_PATH = config.FIGURES_DIR / "stage4a_vader_validation.png"

# Streaming batch size: one batch of text sits in a few hundred MB.
BATCH_SIZE = 200_000

_OUT_SCHEMA = pa.schema([("review_id", pa.string()), ("vader_compound", pa.float64())])


# --- 1. Score -----------------------------------------------------------------
def score_vader(texts: pd.Series) -> pd.Series:
    """VADER compound score per review text (in-memory; used by --smoke)."""
    analyser = SentimentIntensityAnalyzer()
    scores = [analyser.polarity_scores(t)["compound"] for t in texts]
    return pd.Series(scores, index=texts.index, name="vader_compound")


def score_vader_streaming(corpus_path, window, out_file, batch_size=BATCH_SIZE):
    """Chunked scoring for a full run: read -> score -> write, batch by batch.

    Memory stays flat at ~one batch of text; only the compact compound/stars
    arrays (8 bytes/review) are kept, for validation.
    Returns (compound Series, stars Series, n_scored).
    """
    analyser = SentimentIntensityAnalyzer()
    pf = pq.ParquetFile(corpus_path)
    n_total = pf.metadata.num_rows
    print(f"[1] Score with VADER (streaming; {n_total:,} corpus rows, window {window!r})")

    writer = None
    comp_parts: list[np.ndarray] = []
    star_parts: list[np.ndarray] = []
    n_done = 0
    t0 = time.perf_counter()
    for batch in pf.iter_batches(
        batch_size=batch_size, columns=["review_id", "stars", "date", "text"]
    ):
        d = batch.to_pandas()
        d = d[_window_mask(d["date"].dt.year, window)]
        if not len(d):
            continue
        comp = np.fromiter(
            (analyser.polarity_scores(t)["compound"] for t in d["text"]),
            dtype="float64", count=len(d),
        )
        table = pa.table({
            "review_id": pa.array(d["review_id"].to_numpy(), type=pa.string()),
            "vader_compound": pa.array(comp, type=pa.float64()),
        })
        if writer is None:
            writer = pq.ParquetWriter(out_file, _OUT_SCHEMA)
        writer.write_table(table)

        comp_parts.append(comp)
        star_parts.append(d["stars"].to_numpy(dtype="float64"))
        n_done += len(d)
        rate = n_done / (time.perf_counter() - t0)
        print(f"    scored {n_done:>9,} / {n_total:,}  ({rate:,.0f} reviews/s)")

    if writer is not None:
        writer.close()
    compound = pd.Series(
        np.concatenate(comp_parts) if comp_parts else np.array([]), name="vader_compound"
    )
    stars = pd.Series(
        np.concatenate(star_parts) if star_parts else np.array([]), name="stars"
    )
    return compound, stars, n_done


# --- 2. Validate ----------------------------------------------------------------
def validate(stars: pd.Series, compound: pd.Series, smoke: bool) -> None:
    """Distribution, stars correlation, and (full runs) a diagnostic figure.

    `stars` and `compound` are positionally aligned (same order, RangeIndex)."""
    print("[2] Validation")
    print("    compound distribution:")
    print(compound.describe().to_string().replace("\n", "\n      "))

    # Stars are ordinal, so report Spearman alongside Pearson.
    pearson = compound.corr(stars)
    spearman = compound.corr(stars, method="spearman")
    print(f"\n    VADER vs stars: Pearson r = {pearson:.3f}, Spearman rho = {spearman:.3f}")
    print("    (high correlation feeds the 'text adds little over stars' risk;")
    print("     low correlation questions whether VADER captures the rating signal)")

    print("\n    mean compound by star rating:")
    by_stars = compound.groupby(stars).agg(["mean", "count"])
    print(by_stars.to_string().replace("\n", "\n      "))

    if smoke:
        print("    (smoke run: figures skipped)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(compound, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("VADER compound — distribution")
    axes[0].set_xlabel("compound score")
    axes[0].set_ylabel("reviews")
    axes[1].bar(by_stars.index, by_stars["mean"], width=0.6, color="steelblue")
    axes[1].set_title("Mean compound by star rating")
    axes[1].set_xlabel("stars")
    axes[1].set_ylabel("mean compound")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"    figure -> {FIG_PATH}")


# --- Orchestration ---------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke", action="store_true",
        help="plumbing check: score a handful of reviews, write *_smoke.parquet",
    )
    parser.add_argument(
        "--full-corpus", action="store_true",
        help="score the nationwide corpus (reviews_corpus_full.parquet, all "
             "metros incl. non-US) instead of the US study corpus",
    )
    args = parser.parse_args()

    config.ensure_dirs()
    t0 = time.perf_counter()

    corpus_path = FULL_CORPUS_PARQUET if args.full_corpus else CORPUS_PARQUET
    window = "all" if config.SCORE_ALL_REVIEWS else "union"
    out = out_path(OUT_STEM, smoke=args.smoke)

    if args.smoke:
        # Same per-review scoring as a full run, just in memory on ~12 rows.
        corpus = load_scoring_corpus(
            window=window, smoke=True, corpus_path=corpus_path
        ).reset_index(drop=True)
        print("[1] Score with VADER (smoke, in-memory)")
        compound = score_vader(corpus["text"])
        validate(corpus["stars"], compound, smoke=True)
        pd.DataFrame(
            {"review_id": corpus["review_id"], "vader_compound": compound}
        ).to_parquet(out, index=False)
        n = len(compound)
    else:
        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus not found: {corpus_path}\n"
                "Run Stage 2 first:  .venv\\Scripts\\python.exe -m pipeline.stage2_corpus"
            )
        compound, stars, n = score_vader_streaming(corpus_path, window, out)
        validate(stars, compound, smoke=False)

    runtime = time.perf_counter() - t0
    print(f"\nSaved {n:,} scores -> {out}")
    print(f"Runtime: {runtime:.1f} s ({runtime / 60:.1f} min)")


if __name__ == "__main__":
    main()
