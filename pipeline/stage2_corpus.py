"""Stage 2 — assemble the US review corpus, streamed to disk.

Filters the raw review file (per pipeline.stage2_cleaning: blank text, then the
Stage 1 business whitelist — which is the US-only filter — then the 2013-2022
year bound) and prints a cleaning funnel of each filter's removals. The corpus
is tract-agnostic: the business-to-tract join is deferred to Stage 5, so a one-
or two-block design within 2013-2022 stays open.

The review file is ~5.3 GB / ~7M rows and the US corpus ~6.88M rows (~3 GB), so
each chunk is written immediately through one ParquetWriter — memory stays flat
at about one chunk. `date` is parsed and kept; `user_id` is carried for
reviewer-level EDA.

Output: data/processed/reviews_corpus.parquet
        (review_id, business_id, user_id, stars, date, text)

Run:       .venv\\Scripts\\python.exe -m pipeline.stage2_corpus
Full:      .venv\\Scripts\\python.exe -m pipeline.stage2_corpus --full
           (nationwide corpus, all metros incl. non-US, no whitelist filter)
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

import config
from pipeline.stage2_cleaning import (
    apply_date_filter,
    apply_empty_text_filter,
    apply_geographic_filter,
)

# Raw Yelp review file (manual download into data/raw/).
REVIEW_JSON = config.RAW_DIR / "yelp_academic_dataset_review.json"

# Standard output: the US study-state corpus (whitelist applied).
OUT_PARQUET = config.PROCESSED_DIR / "reviews_corpus.parquet"
# --full output: the nationwide corpus (every metro incl. non-US, no whitelist).
FULL_OUT_PARQUET = config.PROCESSED_DIR / "reviews_corpus_full.parquet"

# useful/funny/cool are dropped; user_id is kept for reviewer-level EDA.
KEEP_COLS = ["review_id", "business_id", "user_id", "stars", "date", "text"]
OUT_COLS = ["review_id", "business_id", "user_id", "stars", "date", "text"]

# ~7M reviews at 250k/chunk is ~28 passes — negligible per-chunk overhead, and
# one chunk sits comfortably in RAM.
CHUNKSIZE = 250_000

# Explicit Arrow schema so every chunk writes identical types; the writer
# rejects any later schema drift.
_SCHEMA = pa.schema([
    ("review_id", pa.string()),
    ("business_id", pa.string()),
    ("user_id", pa.string()),
    ("stars", pa.float64()),
    ("date", pa.timestamp("ns")),
    ("text", pa.string()),
])


def _rss_gb() -> float:
    """Resident set size of this process, in GB."""
    return psutil.Process().memory_info().rss / 1e9


# --- 1. In-scope business whitelist ----------------------------------------
def load_business_whitelist() -> set[str]:
    """Union of in-scope business_ids across every per-vintage Stage 1 output."""
    ids: set[str] = set()
    print("[1] In-scope business whitelist (union across vintages)")
    for acs_year in config.ACS_YEARS:
        path = config.INTERIM_DIR / f"businesses_with_tract_{acs_year}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Business file not found: {path}\n"
                "Run Stage 1 first:  .venv\\Scripts\\python.exe -m pipeline.stage1_geography"
            )
        vintage_ids = set(
            pd.read_parquet(path, columns=["business_id"])["business_id"]
        )
        ids |= vintage_ids
        print(f"    ACS {acs_year}: {len(vintage_ids):,} businesses")
    print(f"    union                     : {len(ids):,} businesses")
    return ids


# --- 2. Stream + filter reviews, straight to disk --------------------------
def stream_corpus(
    out_path,
    whitelist: set[str] | None = None,
    review_path=REVIEW_JSON,
    chunksize=CHUNKSIZE,
) -> tuple[int, float]:
    """Stream the review file in chunks, writing kept reviews straight to disk.

    `whitelist=None` keeps every review (the --full corpus); a set keeps only
    in-scope businesses (the US corpus) — the only difference between the two
    paths. Returns (rows written, peak RSS in GB); validation stats are
    accumulated incrementally since no full frame ever exists.
    """
    if not review_path.exists():
        raise FileNotFoundError(
            f"Review file not found: {review_path}\n"
            "Download the Yelp Open Dataset and extract "
            "yelp_academic_dataset_review.json into data/raw/ (see CLAUDE.md)."
        )

    scope = "FULL (all metros, incl. non-US)" if whitelist is None else "US study-state"
    lo, hi = config.CORPUS_DATE_WINDOW
    print(f"[2] Stream reviews and write to disk — scope: {scope}, years {lo}-{hi}")
    n_read = n_blank = n_nonus = n_outside = n_kept = 0
    businesses: set[str] = set()
    date_min = date_max = None
    window_counts = {y: 0 for y in config.REVIEW_WINDOWS}
    samples: list[dict] = []
    peak_rss = _rss_gb()

    writer = None
    # convert_dates=False: `date` is parsed per chunk, explicitly.
    reader = pd.read_json(review_path, lines=True, chunksize=chunksize, convert_dates=False)
    try:
        with reader:
            for i, chunk in enumerate(reader, start=1):
                n_read += len(chunk)
                chunk = chunk[KEEP_COLS]

                # Cleaning funnel: stage2_cleaning filters, in order.
                # 1) blank/null text.
                chunk, removed = apply_empty_text_filter(chunk)
                n_blank += removed
                # 2) US-only whitelist (skipped for --full).
                if whitelist is not None:
                    chunk, removed = apply_geographic_filter(chunk, whitelist)
                    n_nonus += removed

                # Parse `date` once, on the survivors, so the date bound and
                # validation share one parse (the filter's re-parse is a no-op
                # on an already-datetime column).
                if len(chunk):
                    chunk = chunk.copy()
                    chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
                # 3) 2013-2022 corpus date bound (drops ~0; see config).
                chunk, removed = apply_date_filter(chunk, *config.CORPUS_DATE_WINDOW)
                n_outside += removed

                if not len(chunk):
                    peak_rss = max(peak_rss, _rss_gb())
                    if i % 5 == 0:
                        print(f"    chunk {i:>3}: read {n_read:>10,}  "
                              f"kept {n_kept:>9,}  RSS {peak_rss:5.2f} GB")
                    continue

                n_kept += len(chunk)
                businesses.update(chunk["business_id"].unique().tolist())

                valid = chunk["date"].dropna()
                if len(valid):
                    cmin, cmax = valid.min(), valid.max()
                    date_min = cmin if date_min is None else min(date_min, cmin)
                    date_max = cmax if date_max is None else max(date_max, cmax)
                    yr = chunk["date"].dt.year
                    for y, (s, e) in config.REVIEW_WINDOWS.items():
                        window_counts[y] += int(yr.between(s, e).sum())
                if len(samples) < 3:
                    samples.extend(chunk[OUT_COLS].head(3 - len(samples)).to_dict("records"))

                table = pa.Table.from_pandas(chunk[OUT_COLS], preserve_index=False).cast(_SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, _SCHEMA)
                writer.write_table(table)

                peak_rss = max(peak_rss, _rss_gb())
                if i % 5 == 0:
                    print(f"    chunk {i:>3}: read {n_read:>10,}  "
                          f"kept {n_kept:>9,}  RSS {peak_rss:5.2f} GB")
    finally:
        if writer is not None:
            writer.close()

    # Funnel identity: read minus each filter's removals must equal kept — a
    # mismatch means a row was dropped untracked, so it is asserted.
    print("[3] Cleaning funnel")
    print(f"    reviews read (total)          : {n_read:>12,}")
    print(f"    - blank/null text             : {n_blank:>12,}")
    if whitelist is not None:
        print(f"    - outside US study states     : {n_nonus:>12,}  (whitelist / US-only filter)")
    print(f"    - outside review-year scope   : {n_outside:>12,}  ({lo}-{hi}; pre-2013 tail + unparseable)")
    print(f"    = reviews written (kept)      : {n_kept:>12,}")
    assert n_read - n_blank - n_nonus - n_outside == n_kept, (
        f"cleaning funnel does not balance: {n_read:,} - {n_blank:,} - "
        f"{n_nonus:,} - {n_outside:,} != {n_kept:,}"
    )
    print(f"    distinct businesses           : {len(businesses):,}")
    if date_min is not None:
        print(f"    date range                : {date_min}  ->  {date_max}")
    print("    reviews per configured block window:")
    for y, (s, e) in sorted(config.REVIEW_WINDOWS.items()):
        print(f"      ACS {y} ({s}-{e}): {window_counts[y]:,}")
    print("    3 sample rows:")
    for r in samples:
        snippet = " ".join(str(r["text"]).split())[:80]
        date_str = r["date"].strftime("%Y-%m-%d") if pd.notna(r["date"]) else "????-??-??"
        print(f"      {r['review_id']}  stars {float(r['stars']):.0f}  {date_str}  \"{snippet}...\"")
    return n_kept, peak_rss


# --- Orchestration ---------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--full", action="store_true",
        help="nationwide mode: skip the whitelist and keep every review (all "
             "metros incl. non-US, all years) -> reviews_corpus_full.parquet",
    )
    args = parser.parse_args()

    config.ensure_dirs()
    t0 = time.perf_counter()

    if args.full:
        n_kept, peak_rss = stream_corpus(FULL_OUT_PARQUET, whitelist=None)
        out = FULL_OUT_PARQUET
    else:
        whitelist = load_business_whitelist()
        n_kept, peak_rss = stream_corpus(OUT_PARQUET, whitelist=whitelist)
        out = OUT_PARQUET

    runtime = time.perf_counter() - t0
    print(f"\nSaved {n_kept:,} reviews -> {out}")
    print(f"Peak process memory (RSS) : {peak_rss:.2f} GB")
    print(f"Runtime                   : {runtime:.1f} s ({runtime / 60:.1f} min)")


if __name__ == "__main__":
    main()
