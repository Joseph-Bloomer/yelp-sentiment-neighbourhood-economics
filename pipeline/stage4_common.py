"""Shared helpers for the Stage 4 scoring scripts (4a VADER, 4b SiEBERT, 4c LLM).

All three scripts load the corpus through load_scoring_corpus, so a --smoke run
exercises exactly the code path a real run takes — only the row count differs,
and smoke outputs get a `_smoke` suffix so they never clobber real scores. The
ABSA aspect list and polarity encoding also live here, so 4c (writer) and
Stage 5 (reader) share one schema.
"""
from __future__ import annotations

import pandas as pd

import config

CORPUS_PARQUET = config.PROCESSED_DIR / "reviews_corpus.parquet"
# Optional nationwide corpus (every metro, every year), built by
# `stage2_corpus --full`; scoring scripts reach it with --full-corpus. Stage 5
# still subsets to Philadelphia on review_id, so scoring the superset leaves
# the study analysis unchanged.
FULL_CORPUS_PARQUET = config.PROCESSED_DIR / "reviews_corpus_full.parquet"

# Rows scored by a --smoke run: enough to surface schema/parse bugs and show
# both polarities, small enough that even the LLM pass finishes on a CPU
# laptop in minutes.
SMOKE_N = 12

# The five fixed aspect categories, adapted from the SemEval restaurant scheme;
# location is the target signal.
ASPECTS = ["food", "service", "price", "ambience", "location"]

# Wide-column polarity encoding. Absent (not mentioned) is NaN — distinct from
# neutral (mentioned), which is 0.0.
POLARITY_VALUE = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def resolve_device() -> str:
    """Resolve config.DEVICE ("auto" -> cuda if available, else cpu)."""
    if config.DEVICE != "auto":
        return config.DEVICE
    import torch  # deferred so config and 4a stay torch-free

    return "cuda" if torch.cuda.is_available() else "cpu"


def out_path(stem: str, smoke: bool):
    """Output parquet path; smoke runs get a `_smoke` suffix so they never
    overwrite real scores."""
    return config.PROCESSED_DIR / f"{stem}{'_smoke' if smoke else ''}.parquet"


def _window_mask(years: pd.Series, window) -> pd.Series:
    """Boolean mask of review years inside `window`.

    window: "all"             -> everything (no date filter);
            "union"           -> any configured REVIEW_WINDOWS range;
            (start, end)       -> a literal inclusive year range; either bound
                                  may be None for open-ended (e.g. (2013, None)
                                  is "2013 onward" — the LLM scope, config);
            an ACS end year   -> that vintage's REVIEW_WINDOWS window only.
    """
    if window == "all":
        return pd.Series(True, index=years.index)
    if window == "union":
        mask = pd.Series(False, index=years.index)
        for start, end in config.REVIEW_WINDOWS.values():
            mask |= years.between(start, end)
        return mask
    if isinstance(window, (tuple, list)):
        start, end = window
        mask = pd.Series(True, index=years.index)
        if start is not None:
            mask &= years >= start
        if end is not None:
            mask &= years <= end
        return mask
    start, end = config.REVIEW_WINDOWS[int(window)]
    return years.between(start, end)


def _stratified_sample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Sample ~n reviews proportionally across businesses.

    Per-business rounding can drop small businesses to zero rows, so I top up
    (or trim) to exactly n afterwards.
    """
    if n >= len(df):
        return df
    sampled = df.groupby("business_id", group_keys=False).sample(
        frac=n / len(df), random_state=seed
    )
    if len(sampled) < n:
        top_up = df.drop(sampled.index).sample(n - len(sampled), random_state=seed)
        sampled = pd.concat([sampled, top_up])
    return sampled.sample(frac=1, random_state=seed).head(n).reset_index(drop=True)


def load_scoring_corpus(
    window="union", smoke: bool = False, sample_n: int | None = None,
    corpus_path=None,
) -> pd.DataFrame:
    """Load the review corpus filtered and sampled per the run mode.

    Precedence: --smoke (SMOKE_N rows, or `sample_n`) beats DEV_MODE
    (DEV_SAMPLE_N rows) beats the full window; both samples are stratified
    across businesses. `corpus_path` defaults to the Philadelphia corpus; pass
    FULL_CORPUS_PARQUET (--full-corpus) to score the nationwide one.
    """
    path = corpus_path or CORPUS_PARQUET
    if not path.exists():
        hint = " --full" if path == FULL_CORPUS_PARQUET else ""
        raise FileNotFoundError(
            f"Corpus not found: {path}\n"
            f"Run Stage 2 first:  .venv\\Scripts\\python.exe -m pipeline.stage2_corpus{hint}"
        )
    df = pd.read_parquet(path)
    n_total = len(df)

    mask = _window_mask(df["date"].dt.year, window)
    df = df[mask].reset_index(drop=True)
    print(f"[corpus] {n_total:,} reviews total; {len(df):,} in window {window!r}")

    if smoke:
        df = _stratified_sample(df, sample_n or SMOKE_N)
        print(f"[corpus] SMOKE RUN: {len(df)} reviews (plumbing check only)")
    elif config.DEV_MODE:
        df = _stratified_sample(df, config.DEV_SAMPLE_N)
        print(f"[corpus] DEV_MODE: sampled {len(df):,} reviews stratified across businesses")

    return df


def empty_aspect_record() -> dict:
    """One review's wide ABSA record with every aspect absent:
    `{aspect}_mentioned` False, `{aspect}_polarity` NaN.
    """
    record: dict = {}
    for aspect in ASPECTS:
        record[f"{aspect}_mentioned"] = False
        record[f"{aspect}_polarity"] = float("nan")
    return record


def print_aspect_validation(scores: pd.DataFrame) -> None:
    """Per-aspect mention rate and polarity split (ABSA validation, stage 4c).

    location is flagged: it is the target signal and the sparsity risk —
    expect a low rate, but it must not be ~0.
    """
    print("\n    per-aspect mention rate and polarity (of mentions):")
    for aspect in ASPECTS:
        mentioned = scores[f"{aspect}_mentioned"]
        rate = mentioned.mean()
        pol = scores.loc[mentioned, f"{aspect}_polarity"]
        if len(pol):
            split = (
                f"pos {(pol > 0).mean():.0%} / neu {(pol == 0).mean():.0%}"
                f" / neg {(pol < 0).mean():.0%}"
            )
        else:
            split = "no mentions"
        flag = "  <-- TARGET SIGNAL (sparsity risk)" if aspect == "location" else ""
        print(f"      {aspect:9s}: mentioned {rate:6.1%}   {split}{flag}")
