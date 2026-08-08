"""Stage 3 — pull the ACS 5-year outcome indicators, one file per vintage.

For each vintage in config.ACS_YEARS I pull median household income, median
gross rent, poverty rate and unemployment rate for every tract in the study
states. One file per vintage because each sits on its own tract boundaries
(the 2020 redraw); the join to review aggregates happens in Stage 5.

Output: data/processed/acs_tracts_{acs_year}.parquet
Run:    .venv\\Scripts\\python.exe -m pipeline.stage3_acs
"""
from __future__ import annotations

import pandas as pd
import requests

import config


def acs_base(year: int) -> str:
    """ACS API endpoint for one vintage, e.g. https://api.census.gov/data/2019/acs/acs5."""
    return f"https://api.census.gov/data/{year}/acs/{config.ACS_PRODUCT}"


def out_parquet(acs_year: int):
    return config.PROCESSED_DIR / f"acs_tracts_{acs_year}.parquet"


# Raw estimates: two medians used directly, plus numerator/denominator pairs
# for the two derived rates.
RAW_CODES = [
    "B19013_001E",  # median household income (end-year inflation-adjusted dollars)
    "B25064_001E",  # median gross rent
    "B17001_002E",  # poverty: income in past 12 months below poverty level (numerator)
    "B17001_001E",  # poverty: universe — population with poverty status determined (denominator)
    "B23025_005E",  # employment: unemployed, civilian labour force (numerator)
    "B23025_003E",  # employment: civilian labour force (denominator)
]

# Keyed on tract_geoid; acs_year is carried so a file read alone says its vintage.
INDICATOR_COLS = [
    "median_household_income",
    "median_gross_rent",
    "poverty_rate",
    "unemployment_rate",
]
OUT_COLS = ["tract_geoid", "acs_year", *INDICATOR_COLS]


# --- 0. API key ------------------------------------------------------------
def get_api_key() -> str:
    """Return the Census API key, or fail early with setup instructions.

    Without a key the API answers an HTML page with HTTP 200, which would
    otherwise surface much later as a cryptic JSON-decode error.
    """
    key = config.CENSUS_API_KEY
    if not key:
        raise RuntimeError(
            "CENSUS_API_KEY is not set.\n"
            "Request a free key at https://api.census.gov/data/key_signup.html "
            "and add it to a .env file in the project root as:\n"
            "    CENSUS_API_KEY=your_key_here"
        )
    return key


# --- 1. Pull one state -----------------------------------------------------
def fetch_state(state_fips: str, year: int, key: str) -> pd.DataFrame:
    """Fetch the raw ACS variables for every tract in one state, one vintage.

    Tract queries require a county, so I wildcard it (`county:*`) to pull the
    whole state in one call. Returns tract_geoid plus one numeric column per
    raw code.
    """
    params = {
        "get": ",".join(RAW_CODES),
        "for": "tract:*",
        "in": f"state:{state_fips} county:*",
        "key": key,
    }
    resp = requests.get(acs_base(year), params=params, timeout=180)
    resp.raise_for_status()

    # A missing/invalid key (or bad variable) comes back as HTML with status 200,
    # so test the body rather than trusting the status code.
    try:
        rows = resp.json()
    except ValueError as exc:
        raise RuntimeError(
            "Census API did not return JSON — this is almost always a "
            "missing/invalid CENSUS_API_KEY or an unknown variable code.\n"
            f"First 200 chars of response: {resp.text[:200]!r}"
        ) from exc

    header, *data = rows
    df = pd.DataFrame(data, columns=header)

    # GEOID exactly as pygris formats it in Stage 1: state(2)+county(3)+tract(6).
    df["tract_geoid"] = df["state"] + df["county"] + df["tract"]

    # The API returns estimates as strings; jam handling lives in build_indicators.
    for code in RAW_CODES:
        df[code] = pd.to_numeric(df[code], errors="coerce")

    return df[["tract_geoid", *RAW_CODES]]


# --- 2. Pull every study state, one vintage -------------------------------
def fetch_vintage(year: int, states=config.STATES, key: str | None = None) -> pd.DataFrame:
    """Pull and concatenate raw ACS variables across all study states for one vintage."""
    key = key or get_api_key()
    print(f"[1] Pull raw ACS variables - {config.ACS_PRODUCT} {year}")
    parts = []
    for state_fips, label in states:
        part = fetch_state(state_fips, year, key)
        print(f"    {label:42s} {len(part):5,} tracts")
        parts.append(part)
    raw = pd.concat(parts, ignore_index=True)
    print(f"    total tracts pulled       : {len(raw):,}")
    return raw


# --- 3. Derive the four indicators ----------------------------------------
def build_indicators(raw: pd.DataFrame) -> pd.DataFrame:
    """Map jam values to NaN, derive the two rates, assemble the outcome table.

    ACS encodes suppressed/not-computable estimates as large negative jam values
    (e.g. -666666666). None of these variables can legitimately be negative, so
    I blank any negative raw value; the rates inherit NaN wherever a component
    is missing. Denominators are also guarded against zero — an empty universe
    means an undefined rate, not 0 or inf.
    """
    print("[2] Map suppressed/negative jam values to NaN, derive rates")
    raw = raw.copy()

    n_neg = 0
    for code in RAW_CODES:
        neg = raw[code] < 0
        n_neg += int(neg.sum())
        raw.loc[neg, code] = pd.NA
        raw[code] = pd.to_numeric(raw[code], errors="coerce")  # keep float dtype
    print(f"    negative jam values blanked: {n_neg:,} (across all raw columns)")

    # NaN out non-positive denominators so 0/0 and n/0 become NaN, not inf.
    pov_denom = raw["B17001_001E"].where(raw["B17001_001E"] > 0)
    unemp_denom = raw["B23025_003E"].where(raw["B23025_003E"] > 0)

    out = pd.DataFrame(
        {
            "tract_geoid": raw["tract_geoid"],
            "median_household_income": raw["B19013_001E"],
            "median_gross_rent": raw["B25064_001E"],
            # Rates stored as proportions in [0, 1].
            "poverty_rate": raw["B17001_002E"] / pov_denom,
            "unemployment_rate": raw["B23025_005E"] / unemp_denom,
        }
    )
    out = out[["tract_geoid", *INDICATOR_COLS]].sort_values("tract_geoid").reset_index(drop=True)
    return out


# --- Diagnostics -----------------------------------------------------------
def validate(df: pd.DataFrame, year: int) -> None:
    """Print tract counts, per-indicator summary stats, and missing-value counts."""
    print(f"[3] Validation (ACS {year})")
    print(f"    rows (tracts)             : {len(df):,}")
    print(f"    distinct GEOIDs           : {df['tract_geoid'].nunique():,}")
    geoid_len = df["tract_geoid"].str.len()
    print(f"    GEOID length (all 11?)    : {sorted(geoid_len.unique())}")

    # tract_geoid is the table key within a vintage.
    dupes = int(df.duplicated(subset="tract_geoid").sum())
    print(f"    duplicate GEOIDs          : {dupes:,} (must be 0)")

    print("\n    summary stats per indicator:")
    print(df[INDICATOR_COLS].describe().to_string())

    print("\n    missing values per column:")
    miss = df.isna().sum()
    for col in OUT_COLS:
        print(f"      {col:26s}: {int(miss[col]):>4,} / {len(df):,}")

    # All-missing tracts are typically unpopulated/water — expected, not a pull failure.
    all_missing = df[INDICATOR_COLS].isna().all(axis=1).sum()
    print(f"\n    tracts with all four missing: {int(all_missing):,} "
          f"(expected for unpopulated/water tracts)")

    print("\n    3 sample rows:")
    sample = df.dropna(subset=INDICATOR_COLS).head(3)
    for _, r in sample.iterrows():
        print(
            f"      {r['tract_geoid']}  "
            f"inc ${r['median_household_income']:>8,.0f}  "
            f"rent ${r['median_gross_rent']:>6,.0f}  "
            f"pov {r['poverty_rate']:.1%}  "
            f"unemp {r['unemployment_rate']:.1%}"
        )


# --- Orchestration ---------------------------------------------------------
def main() -> None:
    config.ensure_dirs()
    key = get_api_key()

    print(f"[0] ACS vintages: {config.ACS_YEARS}")

    for year in config.ACS_YEARS:
        print(f"\n=== ACS vintage {year} ===")
        raw = fetch_vintage(year, key=key)
        acs = build_indicators(raw)
        acs.insert(1, "acs_year", year)
        acs = acs[OUT_COLS]
        validate(acs, year)

        out = out_parquet(year)
        acs.to_parquet(out, index=False)
        print(f"\nSaved {len(acs):,} tracts -> {out}")


if __name__ == "__main__":
    main()
