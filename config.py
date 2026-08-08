"""Study geography, ACS parameters, run switches, and file paths for the whole pipeline."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Study area: the US metros in the Yelp Open Dataset ---------------------
# ~11 US metros across 13 states. Some metros straddle state lines (Philadelphia:
# PA+NJ+DE, St Louis: MO+IL), so geography is per STATE and tracts are fetched for
# every county. The Stage 1 spatial join keeps only businesses inside these states'
# tracts, which doubles as the US-only filter: Edmonton (the one non-US metro) and
# a handful of mislocated singletons (<=4 per state) never match. Empty tracts are
# harmless — Stage 5 only uses tracts that carry reviews.
# Entries are (state_fips, label); the label is for readability only.
STATES = [
    ("42", "Pennsylvania (Philadelphia)"),
    ("12", "Florida (Tampa)"),
    ("47", "Tennessee (Nashville)"),
    ("18", "Indiana (Indianapolis)"),
    ("29", "Missouri (St Louis)"),
    ("22", "Louisiana (New Orleans)"),
    ("04", "Arizona (Tucson)"),
    ("34", "New Jersey (Philadelphia metro)"),
    ("32", "Nevada (Reno)"),
    ("06", "California (Santa Barbara)"),
    ("16", "Idaho (Boise)"),
    ("10", "Delaware (Wilmington / Philadelphia metro)"),
    ("17", "Illinois (St Louis metro)"),
]

# Convenience lookup: state FIPS -> label (used in diagnostics).
STATE_LABELS = {fips: label for fips, label in STATES}

# --- ACS vintage and review window ------------------------------------------
# One block for now: ACS 5-year ending 2018 (reference period 2014-2018), which
# sits entirely on 2010 tract boundaries. All review-years are parsed and scored
# (VADER/SiEBERT all years; LLM from 2013 — see LLM_SCORE_WINDOW), so a second
# post-COVID block (e.g. ACS 2022, reviews ~2018-2022) only needs its ACS pull and
# a Stage 5/6 re-run — plus its own Stage 1 tract assignment, since it sits on
# 2020 boundaries. A list so the per-vintage loops in Stages 1 and 3 already
# handle more than one block.
ACS_YEARS = [2018]

# acs_year -> (first, last) review year aggregated, inclusive. 2014-2018 matches
# the ACS 2018 reference period one-to-one (~3.63M US reviews fall in it).
REVIEW_WINDOWS = {2018: (2014, 2018)}

ACS_PRODUCT = "acs5"

# A vintage without a window (or vice versa) silently aggregates the wrong years.
assert set(REVIEW_WINDOWS) == set(ACS_YEARS), (
    f"ACS_YEARS {ACS_YEARS} and REVIEW_WINDOWS keys {sorted(REVIEW_WINDOWS)} disagree"
)

# Review years admitted into the corpus at Stage 2. The Yelp dump runs to Jan
# 2022; I drop pre-2013 at source because nothing uses it — the LLM scores 2013+,
# and both candidate ACS blocks (2014-2018, 2018-2022) sit inside 2013-2022. The
# tighter windows (REVIEW_WINDOWS, LLM_SCORE_WINDOW) are applied downstream and
# must stay subsets of this bound.
CORPUS_DATE_WINDOW = (2013, 2022)

# --- Run-mode switches -------------------------------------------------------
# Run Stage 4 on a small random sample while iterating. False for a real run.
DEV_MODE = False
DEV_SAMPLE_N = 5_000

# "auto" resolves to CUDA if available, else CPU, at point of use (config stays
# torch-free).
DEVICE = "auto"

# Scoring is per-review and vintage-independent, so the cheap engines (VADER,
# SiEBERT) score the whole corpus once and Stage 5 picks which reviews to
# aggregate — change the window later and only Stage 5 re-runs. (VADER on the
# laptop; SiEBERT on the A40 cluster.)
SCORE_ALL_REVIEWS = True

# Stage 4c (LLM ABSA) is the expensive pass — one generation per review on the
# A40. Scoring 2013+ (~6.10M of the 6.88M US reviews, 89%) covers any block
# starting 2014+ with a year to spare, without scoring the sparse pre-2013 tail.
# Accepted forms: "all"; "union"; an ACS end year; or a (start, end) tuple with
# end=None open-ended (see pipeline.stage4_common._window_mask).
LLM_SCORE_WINDOW = (2013, None)  # 2013 -> latest

# Stage 4c instruct model (cluster only; swap for a quantised build if VRAM is
# tight). The laptop smoke test substitutes its own tiny model automatically.
LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# Stage 4c writes each shard to data/processed/scores_llm_shards/ as it
# finishes, so a dropped node or OOM loses at most one shard — re-running resumes
# from the first missing part. 50k keeps host RAM bounded while still handing
# vLLM a large batch.
LLM_SHARD_SIZE = 50_000

# Cap on review text (after whitespace-collapse) fed to the model. 5000 is
# Yelp's own review-length limit — the observed max collapsed length is exactly
# 5000 — so this only guards against pathological input. The first full pass used
# 4000; the ~11k reviews it truncated were re-scored in full
# (pipeline/stage4c_rescore_truncated.py) and overwritten in scores_llm.parquet,
# so no review is scored clipped.
LLM_MAX_REVIEW_CHARS = 5000

# Tracts with fewer reviews in a window are too noisy to carry a tract-level
# signal and are excluded from aggregation/evaluation.
MIN_REVIEWS_PER_TRACT = 20

# --- Filesystem layout -----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # manual downloads (Yelp tar/JSON) — gitignored
INTERIM_DIR = DATA_DIR / "interim"      # part-processed (e.g. businesses_with_tract_{year}.parquet)
PROCESSED_DIR = DATA_DIR / "processed"  # analysis-ready outputs
FIGURES_DIR = PROJECT_ROOT / "figures"  # validation + diagnostic plots
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

_DIRS = (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, NOTEBOOKS_DIR)

# --- Secrets ---------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")
# Request a key at https://api.census.gov/data/key_signup.html
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")


def county_geoid(state_fips: str, county_fips: str) -> str:
    """5-digit state+county FIPS (e.g. '42101') — the prefix of every tract GEOID in that county."""
    return f"{state_fips}{county_fips}"


def ensure_dirs() -> None:
    """Create the data/figures directory tree if it does not yet exist."""
    for d in _DIRS:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"Study area (STATES): {len(STATES)} states / ~11 US metros")
    for state_fips, label in STATES:
        print(f"  {label:40s} state FIPS {state_fips}  (tract GEOID prefix {state_fips})")
    print(f"ACS         : {ACS_PRODUCT}, vintages {ACS_YEARS}")
    print("Review windows:")
    for acs_year, (start, end) in sorted(REVIEW_WINDOWS.items()):
        print(f"  ACS {acs_year}: reviews {start}-{end}")
    print("Switches    :")
    print(f"  DEV_MODE={DEV_MODE}  DEV_SAMPLE_N={DEV_SAMPLE_N:,}  DEVICE={DEVICE}")
    print(f"  SCORE_ALL_REVIEWS={SCORE_ALL_REVIEWS}  LLM_SCORE_WINDOW={LLM_SCORE_WINDOW}  "
          f"MIN_REVIEWS_PER_TRACT={MIN_REVIEWS_PER_TRACT}")
    print(f"  LLM_MODEL={LLM_MODEL}")
    print(f"Project root: {PROJECT_ROOT}")
    for name, d in (("raw", RAW_DIR), ("interim", INTERIM_DIR), ("processed", PROCESSED_DIR),
                    ("figures", FIGURES_DIR), ("notebooks", NOTEBOOKS_DIR)):
        print(f"  {name:10s}: {d}  ({'exists' if d.exists() else 'missing'})")
    print(f"Census key  : {'set' if CENSUS_API_KEY else 'NOT set (add CENSUS_API_KEY to .env)'}")
