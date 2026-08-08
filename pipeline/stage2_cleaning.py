"""The one place corpus cleaning lives: Stage 2 applies these three filters, in
order, to every chunk it streams off the raw review file. Each returns
``(kept, n_removed)`` so the cleaning funnel can report per-filter drops."""
from __future__ import annotations

import pandas as pd


def apply_geographic_filter(
    df: pd.DataFrame, us_business_ids: set[str]
) -> tuple[pd.DataFrame, int]:
    """Keep only reviews of US study-state businesses.

    ``us_business_ids`` comes from Stage 1's output
    (businesses_with_tract_2018.parquet), where US membership is actually
    decided — this filter just applies that decision to the reviews.
    """
    before = len(df)
    kept = df[df["business_id"].isin(us_business_ids)]
    return kept, before - len(kept)


def apply_date_filter(
    df: pd.DataFrame, start_year: int, end_year: int
) -> tuple[pd.DataFrame, int]:
    """Keep reviews whose year of ``date`` is in [start_year, end_year], inclusive.

    Unparseable dates cannot be placed in the window and are dropped. Called
    with ``config.CORPUS_DATE_WINDOW`` (2013-2022) — the corpus's outer bound,
    deliberately wider than the per-block windows applied downstream. Here it
    drops the pre-2013 tail (nothing uses those years) plus bad dates.
    """
    before = len(df)
    years = pd.to_datetime(df["date"], errors="coerce").dt.year
    in_window = years.between(start_year, end_year)  # NaT -> NaN -> False -> dropped
    kept = df[in_window]
    return kept, before - len(kept)


def apply_empty_text_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop reviews whose text is null or whitespace-only.

    Deliberately conservative: short but real reviews ("Bad.", "Great!") are
    kept — dropping them would bias the corpus towards longer reviews.
    """
    before = len(df)
    text = df["text"]
    non_blank = text.notna() & (text.str.strip().str.len() > 0)
    kept = df[non_blank]
    return kept, before - len(kept)
