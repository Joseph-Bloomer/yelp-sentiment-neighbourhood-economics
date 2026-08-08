"""Stage 5 — aggregate per-review scores to tract-level features.

Laptop, CPU — joins and groupbys, seconds. Single vintage: ACS 5-year ending
2018, reviews 2014-2018. Attach tracts, window the reviews, join the three
score files on review_id, aggregate per tract. Stage 4 scores the whole corpus,
so a different window is a cheap re-run of this stage alone.

Decisions:
  * Tracts below MIN_REVIEWS_PER_TRACT are flagged (`meets_min_reviews`), not
    dropped — the threshold is a robustness knob for Stage 6.
  * Every method carries `{method}_n_scored`, every aspect
    `{aspect}_n_mentions` — rates without denominators are uninterpretable.
  * Absent != neutral survives aggregation: `{aspect}_polarity_mean` averages
    over mentioning reviews only (absent is NaN at review level, so the
    NaN-skipping mean is exactly the estimator I want);
    `{aspect}_mention_rate` is the salience feature.
  * SiEBERT headline is `siebert_share_pos` (share with P(positive) > 0.5) —
    the natural read of a binary model; `siebert_mean_prob` kept as secondary.
  * Means are review-level, so heavily-reviewed businesses weigh more.

Writes data/processed/tract_features_{acs_year}.parquet — one row per tract
with >= 1 windowed review. --smoke aggregates the *_smoke score files (tiny,
disjoint samples) to prove joins and schema.
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from pipeline.stage4_common import ASPECTS, CORPUS_PARQUET

# Per-review score files; the smoke variants are whatever the Stage 4 smoke
# runs produced (the LLM one is the Qwen3-4B Q4 laptop-iGPU run).
SCORE_FILES = {
    False: {
        "vader": "scores_vader.parquet",
        "siebert": "scores_siebert.parquet",
        "llm": "scores_llm.parquet",
    },
    True: {
        "vader": "scores_vader_smoke.parquet",
        "siebert": "scores_siebert_smoke.parquet",
        "llm": "scores_llm_q4_smoke.parquet",
    },
}

# An aspect is flagged too sparse when fewer than this share of retained tracts
# have at least MIN_MENTIONS_PER_TRACT mentions of it.
MIN_MENTIONS_PER_TRACT = 5
SPARSE_COVERAGE_FLAG = 0.5


def out_parquet(acs_year: int, smoke: bool):
    suffix = "_smoke" if smoke else ""
    return config.PROCESSED_DIR / f"tract_features_{acs_year}{suffix}.parquet"


def fig_path(acs_year: int):
    return config.FIGURES_DIR / f"stage5_features_{acs_year}.png"


# --- 1. Load inputs ----------------------------------------------------------
def load_scores(smoke: bool) -> dict[str, pd.DataFrame | None]:
    """Load whichever per-review score files exist; report what is missing."""
    print(f"[1] Per-review score files ({'smoke' if smoke else 'full'} set)")
    scores: dict[str, pd.DataFrame | None] = {}
    for method, filename in SCORE_FILES[smoke].items():
        path = config.PROCESSED_DIR / filename
        if path.exists():
            df = pd.read_parquet(path)
            assert df["review_id"].is_unique, f"duplicate review_ids in {filename}"
            # Drop llm_raw before merging — ~0.7 GB across 6.1M rows that
            # aggregation never reads.
            df = df.drop(columns=[c for c in ("llm_raw",) if c in df.columns])
            scores[method] = df
            print(f"    {method:8s}: {len(df):>9,} scored reviews  ({filename})")
        else:
            scores[method] = None
            print(f"    {method:8s}: MISSING ({filename}) — its features will be NaN")
    return scores


# --- 2. Aggregate ------------------------------------------------------------
def aggregate(
    acs_year: int, corpus: pd.DataFrame, scores: dict[str, pd.DataFrame | None]
) -> pd.DataFrame:
    """Tract-level features for the single ACS vintage."""
    start, end = config.REVIEW_WINDOWS[acs_year]
    print(f"\n[2] Aggregate (ACS {acs_year}, reviews {start}-{end})")

    biz_path = config.INTERIM_DIR / f"businesses_with_tract_{acs_year}.parquet"
    biz = pd.read_parquet(biz_path, columns=["business_id", "tract_geoid"])

    # Inner join = the geographic filter: a review survives only if its
    # business has a tract assignment in this vintage.
    df = corpus.merge(biz, on="business_id", how="inner", validate="m:1")
    print(f"    reviews matched to a tract            : {len(df):,} "
          f"(dropped {len(corpus) - len(df):,} with no {acs_year} tract)")

    df = df[df["date"].dt.year.between(start, end)]
    print(f"    reviews inside the window             : {len(df):,}")

    for method, sdf in scores.items():
        if sdf is not None:
            df = df.merge(sdf, on="review_id", how="left", validate="1:1")
    n_scored = {
        m: int(df[c].notna().sum())
        for m, c in [("vader", "vader_compound"), ("siebert", "siebert_pos_prob"),
                     ("llm", f"{ASPECTS[0]}_mentioned")]
        if c in df.columns
    }
    print(f"    scored coverage in window             : {n_scored}")

    # Prepared columns so the whole aggregation is one .agg call. Missing
    # methods become all-NaN columns (features exist, carry no data).
    if "vader_compound" not in df.columns:
        df["vader_compound"] = np.nan
    if "siebert_pos_prob" not in df.columns:
        df["siebert_pos_prob"] = np.nan
    df["siebert_pos"] = (df["siebert_pos_prob"] > 0.5).astype(float).mask(
        df["siebert_pos_prob"].isna()
    )
    for aspect in ASPECTS:
        m_col, p_col = f"{aspect}_mentioned", f"{aspect}_polarity"
        if m_col not in df.columns:
            df[m_col], df[p_col] = np.nan, np.nan
        df[f"{aspect}_mentioned_f"] = df[m_col].astype(float)

    spec = {
        "n_reviews": ("review_id", "size"),
        "n_businesses": ("business_id", "nunique"),
        "stars_mean": ("stars", "mean"),
        "vader_n_scored": ("vader_compound", "count"),
        "vader_mean": ("vader_compound", "mean"),
        "siebert_n_scored": ("siebert_pos_prob", "count"),
        "siebert_share_pos": ("siebert_pos", "mean"),
        "siebert_mean_prob": ("siebert_pos_prob", "mean"),
        "llm_n_scored": (f"{ASPECTS[0]}_mentioned_f", "count"),
    }
    for aspect in ASPECTS:
        spec[f"{aspect}_mention_rate"] = (f"{aspect}_mentioned_f", "mean")
        spec[f"{aspect}_n_mentions"] = (f"{aspect}_mentioned_f", "sum")
        spec[f"{aspect}_polarity_mean"] = (f"{aspect}_polarity", "mean")

    features = df.groupby("tract_geoid").agg(**spec).reset_index()
    features.insert(1, "acs_year", acs_year)
    features["meets_min_reviews"] = features["n_reviews"] >= config.MIN_REVIEWS_PER_TRACT
    return features.sort_values("tract_geoid").reset_index(drop=True)


# --- 3. Validate -------------------------------------------------------------
def validate(features: pd.DataFrame, acs_year: int, smoke: bool) -> None:
    print(f"[3] Validation (ACS {acs_year})")
    acs_path = config.PROCESSED_DIR / f"acs_tracts_{acs_year}.parquet"
    n_acs = len(pd.read_parquet(acs_path)) if acs_path.exists() else None

    n_any = len(features)
    n_kept = int(features["meets_min_reviews"].sum())
    denom = f" of {n_acs} ACS tracts" if n_acs else ""
    print(f"    tracts with >= 1 windowed review      : {n_any}{denom}")
    print(f"    meeting MIN_REVIEWS_PER_TRACT ({config.MIN_REVIEWS_PER_TRACT:>2})   : "
          f"{n_kept} ({n_kept / max(n_any, 1):.0%} of reviewed tracts)")

    kept = features[features["meets_min_reviews"]] if n_kept else features
    print("\n    feature distributions (tracts meeting the minimum):")
    cols = ["n_reviews", "stars_mean", "vader_mean", "siebert_share_pos",
            "location_mention_rate", "location_polarity_mean"]
    print(kept[cols].describe().round(3).to_string().replace("\n", "\n      "))

    print("\n    per-aspect coverage (tracts meeting the minimum):")
    for aspect in ASPECTS:
        n_m = kept[f"{aspect}_n_mentions"]
        ok_share = (n_m >= MIN_MENTIONS_PER_TRACT).mean() if len(kept) else 0.0
        sparse = " <-- TOO SPARSE?" if ok_share < SPARSE_COVERAGE_FLAG else ""
        target = "  (TARGET SIGNAL)" if aspect == "location" else ""
        print(f"      {aspect:9s}: median mentions/tract {n_m.median():6.1f}; "
              f"{ok_share:5.1%} of tracts have >= {MIN_MENTIONS_PER_TRACT}"
              f"{target}{sparse}")

    if smoke:
        print("    (smoke run: disjoint tiny samples — judge joins/schema, "
              "not coverage; figures skipped)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(features["n_reviews"], bins=np.geomspace(1, max(features["n_reviews"].max(), 10), 30),
                 color="steelblue", edgecolor="white")
    axes[0].axvline(config.MIN_REVIEWS_PER_TRACT, color="firebrick", linestyle="--",
                    label=f"min = {config.MIN_REVIEWS_PER_TRACT}")
    axes[0].set_xscale("log")
    axes[0].set_title(f"Reviews per tract — {acs_year} window")
    axes[0].set_xlabel("windowed reviews (log)")
    axes[0].set_ylabel("tracts")
    axes[0].legend()
    shares = [(kept[f"{a}_n_mentions"] >= MIN_MENTIONS_PER_TRACT).mean() for a in ASPECTS]
    colours = ["firebrick" if a == "location" else "steelblue" for a in ASPECTS]
    axes[1].bar(ASPECTS, shares, color=colours)
    axes[1].axhline(SPARSE_COVERAGE_FLAG, color="grey", linestyle=":")
    axes[1].set_title(f"Tracts with >= {MIN_MENTIONS_PER_TRACT} mentions — {acs_year}")
    axes[1].set_ylabel("share of retained tracts")
    fig.tight_layout()
    fig.savefig(fig_path(acs_year), dpi=150)
    plt.close(fig)
    print(f"    figure -> {fig_path(acs_year)}")


# --- Orchestration ---------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke", action="store_true",
        help="aggregate the *_smoke score files to prove joins/schema; "
             "writes tract_features_*_smoke.parquet",
    )
    args = parser.parse_args()

    config.ensure_dirs()
    t0 = time.perf_counter()

    # Guard against a stray extra vintage.
    assert len(config.ACS_YEARS) == 1, (
        f"Stage 5 is single-vintage; config.ACS_YEARS = {config.ACS_YEARS}"
    )
    acs_year = config.ACS_YEARS[0]

    corpus = pd.read_parquet(
        CORPUS_PARQUET, columns=["review_id", "business_id", "stars", "date"]
    )
    assert corpus["review_id"].is_unique, "duplicate review_ids in the corpus"
    scores = load_scores(smoke=args.smoke)

    features = aggregate(acs_year, corpus, scores)
    validate(features, acs_year, smoke=args.smoke)
    out = out_parquet(acs_year, smoke=args.smoke)
    features.to_parquet(out, index=False)
    print(f"    saved {len(features):,} tracts -> {out}")

    runtime = time.perf_counter() - t0
    print(f"\nRuntime: {runtime:.1f} s")


if __name__ == "__main__":
    main()
