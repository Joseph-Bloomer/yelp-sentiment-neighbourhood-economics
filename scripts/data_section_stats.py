"""Descriptives + selection diagnostics for the Data section. Reads pipeline
outputs read-only; prints a summary, writes tables to data/processed/descriptives/
and figures to figures/. Not part of the pipeline.

Five sections, runnable separately via --sections:
  A corpus overview, B geography & coverage, C selection diagnostics,
  D ACS indicator distributions, E char-cap / SiEBERT 512-token truncation.

The 2.4 GB `text` column is streamed in batches; the length arrays and the
seeded 100k token sample are cached in descriptives/_cache/ (--refresh rebuilds).

Run: .venv\\Scripts\\python.exe scripts/data_section_stats.py [--sections A,E]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import mannwhitneyu

# Project root on the path so `import config` works from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# --- Constants ---------------------------------------------------------------
ACS_YEAR = config.ACS_YEARS[0]
WINDOW = config.REVIEW_WINDOWS[ACS_YEAR]                 # (2014, 2018)
CORPUS_YEARS = list(range(*config.CORPUS_DATE_WINDOW)) + [config.CORPUS_DATE_WINDOW[1]]
PCTS = [1, 5, 25, 50, 75, 90, 95, 99]                   # reported percentiles

CORPUS_PATH = config.PROCESSED_DIR / "reviews_corpus.parquet"
CORPUS_FULL_PATH = config.PROCESSED_DIR / "reviews_corpus_full.parquet"
BUSINESS_PATH = config.INTERIM_DIR / f"businesses_with_tract_{ACS_YEAR}.parquet"
BUSINESS_FULL_PATH = config.PROCESSED_DIR / "businesses_full.parquet"
ACS_PATH = config.PROCESSED_DIR / f"acs_tracts_{ACS_YEAR}.parquet"
TRACT_FEATURES_PATH = config.PROCESSED_DIR / f"tract_features_{ACS_YEAR}.parquet"

# Stage 6 per-target analysis n (>=20-review tracts minus missing outcomes).
# Self-check target for the spread-ratio "analysis" group.
STAGE6_ANALYSIS_N = {
    "median_household_income": 3_418,
    "median_gross_rent": 3_358,
    "poverty_rate": 3_425,
    "unemployment_rate": 3_425,
}

# Stage 1-2 funnel anchor; used only if businesses_full.parquet is absent.
RAW_BUSINESSES_INCL_CANADA = 150_346

ACS_INDICATORS = [
    "median_household_income",
    "median_gross_rent",
    "poverty_rate",
    "unemployment_rate",
]
ACS_LABELS = {
    "median_household_income": "Median household income (USD)",
    "median_gross_rent": "Median gross rent (USD)",
    "poverty_rate": "Poverty rate (proportion 0-1)",
    "unemployment_rate": "Unemployment rate (proportion 0-1)",
}

# Food test: any category token containing 'Restaurant', or membership of this set.
FOOD_CATEGORIES = {
    "Food", "Cafes", "Coffee & Tea", "Bakeries", "Bars", "Breweries",
    "Desserts", "Food Trucks", "Specialty Food", "Fast Food", "Sandwiches",
    "Pizza", "Wine Bars", "Ice Cream & Frozen Yogurt",
}

# Section E — processing artefacts.
CHAR_CAP_OLD = 4_000          # the earlier, buggy LLM cap
CHAR_CAP_NEW = 5_000          # the corrected Yelp source-level cap
SIEBERT_MODEL = "siebert/sentiment-roberta-large-english"
SIEBERT_TOKEN_LIMIT = 512     # RoBERTa max sequence length (incl. 2 special tokens)
WORDS_PER_TOKEN = 1.3         # rough words->tokens inflation for the full-corpus proxy
TOKEN_SAMPLE_N = 100_000
TOKEN_SAMPLE_SEED = 42

DESC_DIR = config.PROCESSED_DIR / "descriptives"
CACHE_DIR = DESC_DIR / "_cache"


# --- Small shared helpers ----------------------------------------------------
def banner(letter: str, title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  SECTION {letter}. {title}")
    print("=" * 78)


def show(obj, n: int | None = None) -> None:
    """Print a DataFrame/Series indented by four spaces."""
    text = (obj.head(n) if n is not None else obj).to_string()
    print("    " + text.replace("\n", "\n    "))


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> None:
    path = DESC_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    print(f"    table  -> {path.relative_to(config.PROJECT_ROOT)}")


def save_fig(fig, name: str) -> None:
    path = config.FIGURES_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    figure -> {path.relative_to(config.PROJECT_ROOT)}")


def summarise(values, pcts=PCTS) -> pd.Series:
    """count/mean/median/sd/min/max + percentiles, NaNs dropped."""
    s = pd.Series(np.asarray(values, dtype="float64")).dropna()
    out = {
        "count": float(s.size),
        "mean": s.mean(),
        "median": s.median(),
        "sd": s.std(ddof=1),
        "min": s.min(),
        "max": s.max(),
    }
    for p in pcts:
        out[f"p{p}"] = s.quantile(p / 100.0)
    return pd.Series(out)


# --- Loading -----------------------------------------------------------------
def load_business_table() -> pd.DataFrame:
    """US-tract-matched businesses with state FIPS from the tract GEOID."""
    biz = pd.read_parquet(
        BUSINESS_PATH, columns=["business_id", "tract_geoid", "name", "categories"]
    )
    # First two GEOID digits = state FIPS — not the Yelp `state` column (mislabels).
    biz["state_fips"] = biz["tract_geoid"].str[:2]
    return biz


def load_review_meta(biz: pd.DataFrame) -> pd.DataFrame:
    """Non-text corpus columns with year/tract/state attached. Id columns are
    cast to `category` to keep the ~6.1M-row frame near 1 GB."""
    print("[load] reading corpus metadata (no text) ...")
    t0 = time.perf_counter()
    meta = pd.read_parquet(
        CORPUS_PATH, columns=["business_id", "user_id", "stars", "date"]
    )
    meta["year"] = meta["date"].dt.year.astype("int16")

    # Business->tract is m:1.
    biz_tract = biz.set_index("business_id")["tract_geoid"]
    biz_state = biz.set_index("business_id")["state_fips"]
    meta["tract_geoid"] = meta["business_id"].map(biz_tract).astype("category")
    meta["state_fips"] = meta["business_id"].map(biz_state).astype("category")
    n_unmapped = int(meta["tract_geoid"].isna().sum())

    meta["business_id"] = meta["business_id"].astype("category")
    meta["user_id"] = meta["user_id"].astype("category")
    print(f"       {len(meta):,} reviews loaded in {time.perf_counter() - t0:.0f} s "
          f"(~{meta.memory_usage(deep=True).sum() / 1e9:.1f} GB); "
          f"{n_unmapped:,} reviews without a tract")
    return meta


# --- Streamed text pass (lengths + seeded token sample) ----------------------
def _stream_text_features(corpus_path: Path, sample_n: int, seed: int):
    """One streamed pass over `text`: char/word lengths plus a `sample_n` reservoir
    of raw texts for Section E. Keeping the smallest uniform keys gives an exact
    uniform sample, reproducible for a fixed seed and read order."""
    pf = pq.ParquetFile(corpus_path)
    n_total = pf.metadata.num_rows
    print(f"[text] streaming {n_total:,} review texts "
          f"(char/word lengths + {sample_n:,}-row token sample) ...")
    rng = np.random.default_rng(seed)
    char_parts, word_parts = [], []
    kept_keys = np.empty(0, dtype="float64")
    kept_texts = np.empty(0, dtype=object)
    n_done, t0 = 0, time.perf_counter()
    for batch in pf.iter_batches(batch_size=200_000, columns=["text"]):
        ser = batch.column("text").to_pandas().fillna("")
        char_parts.append(ser.str.len().to_numpy(dtype="int32"))
        word_parts.append(ser.str.split().str.len().to_numpy(dtype="int32"))

        keys = np.concatenate([kept_keys, rng.random(len(ser))])
        texts = np.concatenate([kept_texts, ser.to_numpy()])
        if keys.size > sample_n:
            idx = np.argpartition(keys, sample_n)[:sample_n]
            kept_keys, kept_texts = keys[idx], texts[idx]
        else:
            kept_keys, kept_texts = keys, texts

        n_done += len(ser)
        print(f"       {n_done:>9,}/{n_total:,} "
              f"({n_done / (time.perf_counter() - t0):,.0f}/s)", end="\r")
    print()
    return (np.concatenate(char_parts), np.concatenate(word_parts), kept_texts)


def get_text_features(refresh: bool, sample_n: int):
    """Cached wrapper around the streamed text pass."""
    char_p = CACHE_DIR / "char_lengths.npy"
    word_p = CACHE_DIR / "word_lengths.npy"
    samp_p = CACHE_DIR / "token_sample.parquet"
    if not refresh and char_p.exists() and word_p.exists() and samp_p.exists():
        print("[text] using cached length arrays + token sample "
              "(--refresh to rebuild)")
        sample = pd.read_parquet(samp_p)["text"].to_numpy()
        return np.load(char_p), np.load(word_p), sample

    char_lens, word_lens, sample = _stream_text_features(
        CORPUS_PATH, sample_n, TOKEN_SAMPLE_SEED
    )
    np.save(char_p, char_lens)
    np.save(word_p, word_lens)
    pd.DataFrame({"text": sample}).to_parquet(samp_p, index=False)
    print(f"[text] cached -> {CACHE_DIR.relative_to(config.PROJECT_ROOT)}")
    return char_lens, word_lens, sample


# =============================================================================
# A. CORPUS OVERVIEW
# =============================================================================
def section_a(meta: pd.DataFrame, biz: pd.DataFrame, char_lens, word_lens) -> None:
    banner("A", "Corpus overview")

    # --- headline counts + dates ---
    n_reviews = len(meta)
    n_biz = meta["business_id"].nunique()
    n_users = meta["user_id"].nunique()
    d_min, d_max = meta["date"].min(), meta["date"].max()
    overview = pd.Series({
        "n_reviews": n_reviews,
        "n_distinct_businesses": n_biz,
        "n_distinct_users": n_users,
        "date_min": d_min.date().isoformat(),
        "date_max": d_max.date().isoformat(),
    })
    print("  Headline counts:")
    show(overview)
    print(f"  (note: {d_max.date().isoformat()} — 2022 is a partial year, the dump "
          f"ends mid-January)")

    # --- reviews per calendar year ---
    per_year = (meta["year"].value_counts()
                .reindex(CORPUS_YEARS, fill_value=0).sort_index())
    per_year_df = pd.DataFrame({
        "n_reviews": per_year,
        "share_pct": (per_year / n_reviews * 100).round(2),
    })
    per_year_df.index.name = "year"
    print("\n  Reviews per calendar year:")
    show(per_year_df)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(per_year.index.astype(str), per_year.values, color="steelblue")
    ax.set_title("Reviews per calendar year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of reviews")
    save_fig(fig, "data_reviews_per_year")

    # --- star distribution ---
    stars_int = meta["stars"].round().astype(int)
    counts = stars_int.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0).sort_index()
    star_df = pd.DataFrame({
        "count": counts,
        "share_pct": (counts / counts.sum() * 100).round(2),
    })
    star_df.index.name = "stars"
    moments = pd.Series({
        "mean": meta["stars"].mean(),
        "median": meta["stars"].median(),
        "sd": meta["stars"].std(ddof=1),
        "five_star_share_pct": star_df.loc[5, "share_pct"],
    })
    print("\n  Star rating distribution:")
    show(star_df)
    print(f"  mean={moments['mean']:.3f}  median={moments['median']:.1f}  "
          f"sd={moments['sd']:.3f}  |  5-star share = {moments['five_star_share_pct']:.1f}%")

    # --- review length (characters & words) ---
    length_df = pd.DataFrame({
        "characters": summarise(char_lens),
        "words": summarise(word_lens),
    })
    print("\n  Review length (characters and whitespace-split words):")
    show(length_df.round(1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(np.clip(char_lens, 0, 5000), bins=60, color="steelblue",
                 edgecolor="white")
    axes[0].set_title("Review length — characters")
    axes[0].set_xlabel("Characters (clipped at 5,000)")
    axes[0].set_ylabel("Number of reviews")
    axes[1].hist(np.clip(word_lens, 0, 1000), bins=60, color="seagreen",
                 edgecolor="white")
    axes[1].set_title("Review length — words")
    axes[1].set_xlabel("Words (clipped at 1,000)")
    axes[1].set_ylabel("Number of reviews")
    save_fig(fig, "data_review_length")

    # --- reviews per business & per user (heavily skewed) ---
    rpb = meta.groupby("business_id", observed=True).size()
    rpu = meta.groupby("user_id", observed=True).size()
    per_entity = pd.DataFrame({
        "reviews_per_business": summarise(rpb),
        "reviews_per_user": summarise(rpu),
    })
    one_review_share = (rpu == 1).mean() * 100
    print("\n  Reviews per business and per user (expect strong right skew):")
    show(per_entity.round(2))
    print(f"  users with exactly one review: {(rpu == 1).sum():,} "
          f"({one_review_share:.1f}% of users)")

    # top 10 businesses by corpus review count, with names
    names = biz.set_index("business_id")["name"]
    top_biz = (rpb.sort_values(ascending=False).head(10).rename("n_reviews")
               .reset_index())
    top_biz["name"] = top_biz["business_id"].map(names)
    top_biz = top_biz[["business_id", "name", "n_reviews"]]
    print("\n  Top 10 businesses by review count:")
    show(top_biz, n=10)

    # --- skew figure (log-scaled histograms) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, colour, label in (
        (axes[0], rpb, "steelblue", "business"),
        (axes[1], rpu, "indianred", "user"),
    ):
        bins = np.geomspace(1, max(data.max(), 10), 40)
        ax.hist(data, bins=bins, color=colour, edgecolor="white")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Reviews per {label}")
        ax.set_xlabel("Reviews (log)")
        ax.set_ylabel(f"{label.capitalize()}s (log)")
    save_fig(fig, "data_reviews_per_entity")

    # --- persist ---
    save_table(overview.to_frame("value"), "a_corpus_overview")
    save_table(per_year_df, "a_reviews_per_year")
    save_table(star_df, "a_star_distribution")
    save_table(moments.to_frame("value"), "a_star_moments")
    save_table(length_df, "a_review_length")
    save_table(per_entity, "a_reviews_per_entity")
    save_table(pd.Series({"users_with_one_review": int((rpu == 1).sum()),
                          "share_pct": one_review_share}).to_frame("value"),
               "a_single_review_users")
    save_table(top_biz, "a_top_businesses", index=False)


# =============================================================================
# B. GEOGRAPHY & COVERAGE
# =============================================================================
def section_b(meta: pd.DataFrame, biz: pd.DataFrame, acs: pd.DataFrame) -> None:
    banner("B", "Geography & coverage")

    biz_tracts = set(biz["tract_geoid"].unique())
    acs_tracts = set(acs["tract_geoid"].unique())

    # --- per-state counts (state from tract GEOID FIPS, not the Yelp column) ---
    biz_per_state = biz.groupby("state_fips").size().rename("n_businesses")
    rev_per_state = meta.groupby("state_fips", observed=True).size().rename("n_reviews")
    per_state = pd.concat([biz_per_state, rev_per_state], axis=1).fillna(0).astype(int)
    per_state["label"] = [config.STATE_LABELS.get(f, "?") for f in per_state.index]
    per_state = per_state.sort_values("n_reviews", ascending=False)
    per_state = per_state[["label", "n_businesses", "n_reviews"]]
    per_state.index.name = "state_fips"

    n_states = biz["state_fips"].nunique()
    win = meta[meta["year"].between(*WINDOW)]
    rev_per_tract_win = win.groupby("tract_geoid", observed=True).size()
    n_tracts_ge_min = int((rev_per_tract_win >= config.MIN_REVIEWS_PER_TRACT).sum())

    print(f"  states (FIPS prefixes)                         : {n_states}")
    print(f"  tracts with >=1 business                       : {len(biz_tracts):,}")
    print(f"  tracts with >={config.MIN_REVIEWS_PER_TRACT} reviews in {WINDOW[0]}-{WINDOW[1]} "
          f"window : {n_tracts_ge_min:,}")
    print("\n  Businesses and reviews per state:")
    show(per_state)

    # --- per-tract concentration ---
    biz_per_tract = biz.groupby("tract_geoid").size()
    rev_per_tract = meta.groupby("tract_geoid", observed=True).size()
    per_tract = pd.DataFrame({
        "businesses_per_tract": summarise(biz_per_tract),
        "reviews_per_tract": summarise(rev_per_tract),
    })
    print("\n  Per-tract concentration (tracts carrying >=1 business; "
          "reviews over the full 2013-2022 corpus):")
    show(per_tract.round(2))

    # --- coverage of the ACS tract universe ---
    covered = acs_tracts & biz_tracts
    n_acs = len(acs_tracts)
    cov = pd.Series({
        "acs_tracts_total": n_acs,
        "covered_tracts": len(covered),
        "covered_pct": round(len(covered) / n_acs * 100, 2),
        "business_tracts_not_in_acs": len(biz_tracts - acs_tracts),
    })
    print("\n  Coverage of the ACS tract universe:")
    show(cov)

    # --- funnels ---
    n_corpus_biz = meta["business_id"].nunique()
    biz_funnel = pd.DataFrame({
        "stage": ["raw (incl. Canada)", "US-tract-matched", "in final corpus"],
        "n_businesses": [
            len(pd.read_parquet(BUSINESS_FULL_PATH, columns=["business_id"]))
            if BUSINESS_FULL_PATH.exists() else RAW_BUSINESSES_INCL_CANADA,
            len(biz),
            n_corpus_biz,
        ],
    })
    print("\n  Business funnel:")
    show(biz_funnel, n=3)

    rev_rows = []
    if CORPUS_FULL_PATH.exists():
        n_full = pq.ParquetFile(CORPUS_FULL_PATH).metadata.num_rows
        rev_rows.append(("nationwide corpus (all metros, 2013-2022)", n_full))
    rev_rows.append(("US study corpus (final, 2013-2022)", len(meta)))
    rev_funnel = pd.DataFrame(rev_rows, columns=["stage", "n_reviews"])
    print("\n  Review funnel (file-reconstructable stages; the pre-2013 and "
          "non-US drops are itemised in the Stage 2 log):")
    show(rev_funnel, n=len(rev_funnel))

    # --- categories ---
    rpb = meta.groupby("business_id", observed=True).size().rename("n_reviews")
    cat = biz[["business_id", "categories"]].dropna(subset=["categories"]).copy()
    cat["category"] = cat["categories"].str.split(",")
    cat = cat.explode("category")
    cat["category"] = cat["category"].str.strip()
    cat = cat[cat["category"] != ""]
    cat = cat.merge(rpb, on="business_id", how="left")
    cat["n_reviews"] = cat["n_reviews"].fillna(0)

    by_biz = (cat.groupby("category")["business_id"].nunique()
              .sort_values(ascending=False).head(20).rename("n_businesses"))
    by_rev = (cat.groupby("category")["n_reviews"].sum()
              .sort_values(ascending=False).head(20).astype(int).rename("n_reviews"))
    print("\n  Top 20 categories by business count:")
    show(by_biz.to_frame())
    print("\n  Top 20 categories by review count:")
    show(by_rev.to_frame())

    # restaurant/food share (business-level flag, then weighted by reviews)
    def is_food(cat_string: str) -> bool:
        toks = [c.strip() for c in str(cat_string).split(",")]
        return any("Restaurant" in t for t in toks) or any(t in FOOD_CATEGORIES for t in toks)

    biz_food = biz.assign(
        is_food=biz["categories"].fillna("").map(is_food)
    ).set_index("business_id")
    food_business_pct = biz_food["is_food"].mean() * 100
    rpb_aligned = rpb.reindex(biz_food.index).fillna(0)
    food_review_pct = rpb_aligned[biz_food["is_food"]].sum() / rpb_aligned.sum() * 100
    food = pd.Series({
        "food_related_businesses_pct": round(food_business_pct, 2),
        "food_related_reviews_pct": round(food_review_pct, 2),
        "businesses_without_categories": int(biz["categories"].isna().sum()),
    })
    print("\n  Restaurant/food-related share "
          "(any 'Restaurant' category + a small food set):")
    show(food)

    # --- persist ---
    save_table(per_state, "b_per_state")
    save_table(per_tract, "b_per_tract")
    save_table(cov.to_frame("value"), "b_coverage")
    save_table(biz_funnel, "b_business_funnel", index=False)
    save_table(rev_funnel, "b_review_funnel", index=False)
    save_table(by_biz.to_frame(), "b_top_categories_by_business")
    save_table(by_rev.to_frame(), "b_top_categories_by_review")
    save_table(food.to_frame("value"), "b_food_share")


# =============================================================================
# C. SELECTION DIAGNOSTICS
# =============================================================================
def spread_ratios(acs: pd.DataFrame) -> None:
    """Range restriction: outcome spread in evaluated tracts vs the universe.
    Ratio < 1 = the slice is more homogeneous, attenuating absolute R² levels.

    Three nested groups: universe (all study-state tracts), covered (>=1 windowed
    review, in the Stage 5 feature table), analysis (also clears the >=20-review
    floor — the Stage 6 set). Descriptive only: at n in the thousands any
    difference is "significant"; the question is magnitude.
    """
    feats = pd.read_parquet(
        TRACT_FEATURES_PATH, columns=["tract_geoid", "meets_min_reviews"]
    )
    covered_tracts = set(feats["tract_geoid"])
    analysis_tracts = set(feats.loc[feats["meets_min_reviews"], "tract_geoid"])
    groups = {
        "universe": acs,
        "covered": acs[acs["tract_geoid"].isin(covered_tracts)],
        "analysis": acs[acs["tract_geoid"].isin(analysis_tracts)],
    }
    print(f"\n  Spread ratios — range restriction on the four ACS outcomes "
          f"(tracts: universe {len(groups['universe']):,}, "
          f"covered {len(groups['covered']):,}, "
          f"analysis {len(groups['analysis']):,}):")

    rows = []
    for ind in ACS_INDICATORS:
        row = {"indicator": ind}
        for name, frame in groups.items():
            s = frame[ind].dropna()
            row[f"n_{name}"] = int(s.size)
            row[f"sd_{name}"] = s.std(ddof=1)
            row[f"iqr_{name}"] = s.quantile(0.75) - s.quantile(0.25)
        for name in ("analysis", "covered"):
            row[f"{name}_sd_ratio"] = row[f"sd_{name}"] / row["sd_universe"]
            row[f"{name}_iqr_ratio"] = row[f"iqr_{name}"] / row["iqr_universe"]
        rows.append(row)
    spread = pd.DataFrame(rows).set_index("indicator")

    printed = ["n_universe", "n_covered", "n_analysis",
               "analysis_sd_ratio", "analysis_iqr_ratio",
               "covered_sd_ratio", "covered_iqr_ratio"]
    show(spread[printed].round(3))
    print("  (analysis/universe is the primary pair; covered/universe says "
          "whether any restriction bites at platform coverage or at the "
          "20-review floor)")

    # The analysis group must be the set Stage 6 actually fits on.
    mismatches = {
        ind: (int(spread.loc[ind, "n_analysis"]), STAGE6_ANALYSIS_N[ind])
        for ind in ACS_INDICATORS
        if int(spread.loc[ind, "n_analysis"]) != STAGE6_ANALYSIS_N[ind]
    }
    if mismatches:
        print("\n  " + "!" * 70)
        print("  WARNING: analysis-group n does NOT match the Stage 6 per-target "
              "counts.")
        for ind, (got, want) in mismatches.items():
            print(f"    {ind}: got {got:,}, expected {want:,}")
        print("  The spread ratios below are computed on a different tract set "
              "than the\n  ladder — do not quote them until this is reconciled.")
        print("  " + "!" * 70)
    else:
        print("  sanity check OK: analysis n per outcome matches Stage 6 "
              "(income 3,418 · poverty 3,425 · unemployment 3,425 · rent 3,358)")

    save_table(spread, "c_spread_ratios")


def section_c(meta: pd.DataFrame, biz: pd.DataFrame, acs: pd.DataFrame) -> None:
    banner("C", "Selection diagnostics")

    # --- covered vs uncovered ACS tracts ---
    biz_tracts = set(biz["tract_geoid"].unique())
    acs = acs.copy()
    acs["covered"] = acs["tract_geoid"].isin(biz_tracts)
    n_cov = int(acs["covered"].sum())
    n_unc = int((~acs["covered"]).sum())
    print(f"  ACS tracts: {n_cov:,} covered (>=1 business) vs {n_unc:,} uncovered\n")

    rows = []
    for ind in ACS_INDICATORS:
        cov = acs.loc[acs["covered"], ind].dropna()
        unc = acs.loc[~acs["covered"], ind].dropna()
        u, p = mannwhitneyu(cov, unc, alternative="two-sided")
        rows.append({
            "indicator": ind,
            "covered_mean": cov.mean(), "uncovered_mean": unc.mean(),
            "covered_median": cov.median(), "uncovered_median": unc.median(),
            "covered_sd": cov.std(ddof=1), "uncovered_sd": unc.std(ddof=1),
            "median_diff_cov_minus_unc": cov.median() - unc.median(),
            "mannwhitney_u": u, "p_value": p,
            "n_covered": cov.size, "n_uncovered": unc.size,
        })
    cov_tab = pd.DataFrame(rows).set_index("indicator")
    print("  Covered vs uncovered tracts (Mann-Whitney two-sided):")
    show(cov_tab.round(4))
    print("  (interpretation: covered tracts differ systematically from the wider "
          "tract universe — this is selection, to be carried as a caveat)")

    # --- range restriction on the evaluated tracts ---
    spread_ratios(acs)

    # --- temporal composition drift ---
    counts = (meta.groupby(["year", "state_fips"], observed=True).size()
              .unstack(fill_value=0).reindex(CORPUS_YEARS, fill_value=0))
    shares = counts.div(counts.sum(axis=1), axis=0)
    conc = pd.DataFrame({
        "n_reviews": counts.sum(axis=1).astype(int),
        "herfindahl_index": (shares ** 2).sum(axis=1),
        "largest_state_share": shares.max(axis=1),
        "largest_state": [config.STATE_LABELS.get(s, "?") for s in shares.idxmax(axis=1)],
    })
    conc.index.name = "year"
    print("\n  Temporal composition — concentration of the state mix per year:")
    show(conc.round(4))

    top5 = counts.sum(axis=0).sort_values(ascending=False).head(5).index
    fig, ax = plt.subplots(figsize=(9, 5))
    for s in top5:
        ax.plot(shares.index, shares[s], marker="o",
                label=config.STATE_LABELS.get(s, s))
    ax.set_title("Share of reviews by year — top 5 states")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of that year's reviews")
    ax.legend(fontsize=8)
    save_fig(fig, "data_state_share_top5")

    # share-of-year matrix with readable labels
    shares_labelled = shares.rename(
        columns=lambda s: f"{s} {config.STATE_LABELS.get(s, '?')}"
    )

    # --- entry cohorts ---
    biz_first = meta.groupby("business_id", observed=True)["year"].min()
    user_first = meta.groupby("user_id", observed=True)["year"].min()
    cohort = pd.DataFrame({
        "new_businesses": biz_first.value_counts().reindex(CORPUS_YEARS, fill_value=0),
        "new_users": user_first.value_counts().reindex(CORPUS_YEARS, fill_value=0),
    })
    cohort.index.name = "year"
    print("\n  Entry cohorts (first appearance in the corpus, by year):")
    show(cohort)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(cohort.index, cohort["new_businesses"], marker="o", label="New businesses")
    ax.plot(cohort.index, cohort["new_users"], marker="s", color="indianred",
            label="New users")
    ax.set_title("New businesses and users entering the corpus each year")
    ax.set_xlabel("Year of first review")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.legend()
    save_fig(fig, "data_cohort_entry")

    # --- persist ---
    save_table(cov_tab, "c_covered_vs_uncovered")
    save_table(conc, "c_year_concentration")
    save_table(shares_labelled.round(5), "c_state_year_shares")
    save_table(cohort, "c_entry_cohorts")


# =============================================================================
# D. ACS INDICATORS
# =============================================================================
def section_d(biz: pd.DataFrame, acs: pd.DataFrame) -> None:
    banner("D", "ACS indicators (income/rent in USD; poverty/unemployment in [0,1])")

    biz_tracts = set(biz["tract_geoid"].unique())
    acs = acs.copy()
    acs["covered"] = acs["tract_geoid"].isin(biz_tracts)
    covered = acs[acs["covered"]]

    blocks = []
    for ind in ACS_INDICATORS:
        all_s = summarise(acs[ind]).add_prefix("all_")
        cov_s = summarise(covered[ind]).add_prefix("covered_")
        blocks.append(pd.concat([all_s, cov_s]).rename(ACS_LABELS[ind]))
    dist = pd.DataFrame(blocks)
    print("  Distributions — all tracts vs covered tracts:")
    show(dist.round(2))

    miss = pd.DataFrame({
        "covered_n_nan": [covered[i].isna().sum() for i in ACS_INDICATORS],
        "covered_pct_nan": [round(covered[i].isna().mean() * 100, 2)
                            for i in ACS_INDICATORS],
        "all_n_nan": [acs[i].isna().sum() for i in ACS_INDICATORS],
    }, index=[ACS_LABELS[i] for i in ACS_INDICATORS])
    miss.index.name = "indicator"
    print("\n  Missingness:")
    show(miss)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, ind in zip(axes.ravel(), ACS_INDICATORS):
        ax.hist(covered[ind].dropna(), bins=50, color="steelblue", edgecolor="white")
        ax.set_title(ACS_LABELS[ind])
        ax.set_xlabel(ACS_LABELS[ind])
        ax.set_ylabel("Covered tracts")
    fig.suptitle(f"ACS {ACS_YEAR} indicator distributions — covered tracts")
    save_fig(fig, "data_acs_covered_hist")

    save_table(dist, "d_acs_distributions")
    save_table(miss, "d_acs_missingness")


# =============================================================================
# E. PROCESSING ARTEFACTS / TRUNCATION
# =============================================================================
def section_e(char_lens, word_lens, sample_texts, skip_tokeniser: bool) -> None:
    banner("E", "Processing artefacts / truncation")
    n = char_lens.size

    # --- character caps ---
    over_old = int((char_lens > CHAR_CAP_OLD).sum())
    over_new = int((char_lens > CHAR_CAP_NEW).sum())
    char_cap = pd.DataFrame({
        "threshold_chars": [CHAR_CAP_OLD, CHAR_CAP_NEW],
        "label": ["old buggy LLM cap", "corrected Yelp source cap"],
        "n_reviews_over": [over_old, over_new],
        "pct_reviews_over": [round(over_old / n * 100, 4), round(over_new / n * 100, 4)],
    })
    print("  Character-length caps:")
    show(char_cap, n=2)
    print(f"  -> {'OK: ~0 reviews exceed 5,000 chars' if over_new <= max(1, n // 100000) else 'WARNING: reviews exceed the 5,000 cap'} "
          f"({over_new:,} of {n:,})")

    # --- SiEBERT 512-token cap: full-corpus proxy + tokeniser estimate ---
    word_thresh = SIEBERT_TOKEN_LIMIT / WORDS_PER_TOKEN
    over_proxy = int((word_lens > word_thresh).sum())
    rows = [{
        "method": f"full-corpus proxy (words x {WORDS_PER_TOKEN} > {SIEBERT_TOKEN_LIMIT})",
        "n_reviews": n,
        "pct_over_512_tokens": round(over_proxy / n * 100, 3),
    }]

    if skip_tokeniser:
        print("\n  SiEBERT token cap: tokeniser step skipped (--skip-tokeniser); "
              "proxy only.")
    else:
        try:
            tok_lens = _tokenise_lengths(sample_texts)
            over_tok = int((tok_lens > SIEBERT_TOKEN_LIMIT).sum())
            m = len(tok_lens)
            rows.append({
                "method": f"tokeniser estimate ({SIEBERT_MODEL}, n={m:,} sample)",
                "n_reviews": m,
                "pct_over_512_tokens": round(over_tok / m * 100, 3),
            })
            tok_pcts = pd.Series(np.percentile(tok_lens, [50, 90, 95, 99, 100]),
                                 index=["p50", "p90", "p95", "p99", "max"])
            print("\n  Token-length percentiles (sampled, incl. special tokens):")
            show(tok_pcts.round(1))
        except Exception as exc:  # tokeniser unavailable — proxy still stands
            print(f"\n  SiEBERT tokeniser unavailable ({type(exc).__name__}: {exc}); "
                  "reporting the words-based proxy only.")

    token_cap = pd.DataFrame(rows)
    print("\n  Reviews exceeding the SiEBERT 512-token limit "
          "(the real truncation caveat for the SiEBERT arm):")
    show(token_cap, n=len(token_cap))

    save_table(char_cap, "e_character_caps", index=False)
    save_table(token_cap, "e_siebert_token_cap", index=False)


def _tokenise_lengths(texts, model=SIEBERT_MODEL, batch=2000):
    """SiEBERT-tokeniser token counts (incl. 2 special tokens) for the sample."""
    from transformers import AutoTokenizer
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
    print(f"  loading tokeniser {model} ...")
    tok = AutoTokenizer.from_pretrained(model)
    texts = [str(t) for t in texts]
    lens = np.empty(len(texts), dtype="int32")
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], add_special_tokens=True, truncation=False)
        lens[i:i + batch] = [len(x) for x in enc["input_ids"]]
        print(f"    tokenised {min(i + batch, len(texts)):>7,}/{len(texts):,}", end="\r")
    print(f"\n  tokenised {len(texts):,} sampled reviews in "
          f"{time.perf_counter() - t0:.0f} s")
    return lens


# --- Orchestration -----------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sections", default="ABCDE",
        help="which sections to run, e.g. 'A,E' or 'BCD' (default: all)",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="rebuild the cached text-length arrays + token sample from the corpus",
    )
    parser.add_argument(
        "--skip-tokeniser", action="store_true",
        help="Section E: skip the SiEBERT tokeniser estimate (report the proxy only)",
    )
    parser.add_argument(
        "--token-sample-n", type=int, default=TOKEN_SAMPLE_N,
        help=f"size of the seeded token sample (default {TOKEN_SAMPLE_N:,})",
    )
    args = parser.parse_args()

    sections = {c for c in args.sections.upper() if c in "ABCDE"}
    if not sections:
        parser.error("no valid sections selected (choose from A B C D E)")

    # Windows consoles default to cp1252, which mangles em-dashes; force UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    config.ensure_dirs()
    DESC_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"Descriptives — sections {''.join(sorted(sections))}  "
          f"(ACS {ACS_YEAR}, window {WINDOW[0]}-{WINDOW[1]})")
    print(f"tables -> {DESC_DIR.relative_to(config.PROJECT_ROOT)}/  "
          f"figures -> {config.FIGURES_DIR.relative_to(config.PROJECT_ROOT)}/")

    need_meta = bool(sections & set("ABC"))
    need_biz = bool(sections & set("ABCD"))
    need_acs = bool(sections & set("BCD"))
    need_text = bool(sections & set("AE"))

    biz = load_business_table() if need_biz else None
    meta = load_review_meta(biz) if need_meta else None
    acs = pd.read_parquet(ACS_PATH) if need_acs else None
    char_lens = word_lens = sample_texts = None
    if need_text:
        char_lens, word_lens, sample_texts = get_text_features(
            args.refresh, args.token_sample_n
        )

    if "A" in sections:
        section_a(meta, biz, char_lens, word_lens)
    if "B" in sections:
        section_b(meta, biz, acs)
    if "C" in sections:
        section_c(meta, biz, acs)
    if "D" in sections:
        section_d(biz, acs)
    if "E" in sections:
        section_e(char_lens, word_lens, sample_texts, args.skip_tokeniser)

    print(f"\nDone in {(time.perf_counter() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
