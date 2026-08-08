"""Stage 1 — assign every US Yelp business to a census tract.

Builds points from the lat/long already on each record (no geocoding), fetches
tracts for every study state on the vintage's boundaries, and spatially joins.
Dropping unmatched businesses doubles as the US-only filter: no tracts are
fetched outside config.STATES, so Edmonton and the mislocated singletons never
match. One assignment per vintage because the 2020 decennial redrew boundaries;
with ACS_YEARS == [2018] the loop runs once.

Writes data/interim/businesses_with_tract_{acs_year}.parquet (GeoParquet).
--full instead exports every business nationwide (no join, no filter, includes
non-US) to data/processed/businesses_full.parquet — the metadata companion to
reviews_corpus_full.parquet.

Run:        .venv\\Scripts\\python.exe -m pipeline.stage1_geography
Full table: .venv\\Scripts\\python.exe -m pipeline.stage1_geography --full
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # headless — we only ever save figures, never display
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pygris

import config

# Raw Yelp business file (manual download into data/raw/).
BUSINESS_JSON = config.RAW_DIR / "yelp_academic_dataset_business.json"

# Columns carried forward. Address, hours, attributes, is_open and postal_code
# are irrelevant to the methodology; `state` is kept for per-state validation.
KEEP_COLS = [
    "business_id",
    "name",
    "city",
    "state",
    "latitude",
    "longitude",
    "stars",
    "review_count",
    "categories",
]

# --full export: the same nine attributes for every business nationwide, no
# tract join or filter. Pairs with reviews_corpus_full.parquet.
BUSINESSES_FULL_PARQUET = config.PROCESSED_DIR / "businesses_full.parquet"


def out_parquet(acs_year: int):
    return config.INTERIM_DIR / f"businesses_with_tract_{acs_year}.parquet"


def out_figure(acs_year: int):
    return config.FIGURES_DIR / f"stage1_businesses_on_tracts_{acs_year}.png"


# --- 1. Load businesses ----------------------------------------------------
def load_businesses(path=BUSINESS_JSON) -> pd.DataFrame:
    """Read the newline-delimited business JSON and keep the relevant columns.

    ~120 MB / ~150k rows — fits in memory whole. (The review file in Stage 2 is
    the one that needs chunking.)
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Business file not found: {path}\n"
            "Download the Yelp Open Dataset and extract "
            "yelp_academic_dataset_business.json into data/raw/ (see CLAUDE.md)."
        )

    df = pd.read_json(path, lines=True)
    n_total = len(df)
    df = df[KEEP_COLS].copy()

    # Coordinates must be numeric and present to build a point. Coerce then drop.
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    n_missing = df[["latitude", "longitude"]].isna().any(axis=1).sum()
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    # Coordinates outside the valid globe are corrupt records.
    bad_range = (
        df["latitude"].abs().gt(90) | df["longitude"].abs().gt(180)
    )
    n_bad_range = int(bad_range.sum())
    if n_bad_range:
        df = df[~bad_range].reset_index(drop=True)

    print("[1] Load businesses")
    print(f"    rows in file              : {n_total:,}")
    print(f"    dropped (missing lat/long): {n_missing:,}")
    print(f"    dropped (impossible range): {n_bad_range:,}")
    print(f"    usable businesses         : {len(df):,}")
    return df


# --- 2. Point geometries ---------------------------------------------------
def to_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Build point geometries in EPSG:4326.

    points_from_xy takes (x=lon, y=lat) — the classic place to silently swap
    axes, so the keywords are spelled out.
    """
    geometry = gpd.points_from_xy(x=df["longitude"], y=df["latitude"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    print("[2] Build point geometries")
    print(f"    businesses CRS            : {gdf.crs.to_string()}")
    return gdf


# --- 3. Fetch tracts -------------------------------------------------------
def fetch_tracts(year: int, states=config.STATES) -> gpd.GeoDataFrame:
    """Fetch cartographic-boundary tracts for every study state (all counties).

    Fetching whole states captures metros that straddle state lines, and never
    fetching non-US tracts is what makes the US-only filter fall out of the
    join. cb=True gives generalised boundaries — lighter, accurate enough for
    point-in-polygon. `year` is the ACS vintage; pygris returns the matching
    TIGER vintage, keeping Stage 1's GEOIDs joinable to Stage 3's.
    """
    print(f"[3] Fetch tracts via pygris (boundary year {year}) for {len(states)} states")
    layers = []
    for state_fips, label in states:
        tr = pygris.tracts(state=state_fips, year=year, cb=True)
        print(f"    {label:42s} {len(tr):5,} tracts")
        layers.append(tr)

    tracts = pd.concat(layers, ignore_index=True)
    tracts = gpd.GeoDataFrame(tracts, geometry="geometry", crs=layers[0].crs)

    # GEOID is the 11-digit state+county+tract id; rename so the whole pipeline
    # says `tract_geoid`.
    tracts = tracts.rename(columns={"GEOID": "tract_geoid"})[["tract_geoid", "geometry"]]
    print(f"    total tracts fetched      : {len(tracts):,}  CRS {tracts.crs.to_string()}")
    return tracts


# --- 4. Align CRS ----------------------------------------------------------
def align_crs(
    businesses: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Reproject businesses onto the tracts' CRS before joining.

    Mismatched CRSs can make geopandas silently return null or nonsense matches
    rather than error. The shift itself is tiny (4326 vs 4269 differ by <1 m in
    the US); aligning unconditionally just removes the failure mode.
    """
    print("[4] Align CRS before the join")
    print(f"    businesses (before)       : {businesses.crs.to_string()}")
    print(f"    tracts                    : {tracts.crs.to_string()}")
    aligned = businesses.to_crs(tracts.crs)
    print(f"    businesses (after)        : {aligned.crs.to_string()}")
    print(f"    CRSs match                : {aligned.crs == tracts.crs}")
    return aligned


# --- 5. Spatial join -------------------------------------------------------
def spatial_join(
    businesses: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Assign each business the tract that contains it; drop the unmatched.

    The left join leaves a null tract_geoid on every business outside the study
    states; dropping those rows is simultaneously the US-only filter.
    """
    print("[5] Spatial join (within)")
    joined = gpd.sjoin(
        businesses,
        tracts[["tract_geoid", "geometry"]],
        how="left",
        predicate="within",
    )

    # A point exactly on a shared boundary can match two tracts; keep first.
    n_dupes = int(joined.duplicated(subset="business_id").sum())
    if n_dupes:
        joined = joined.drop_duplicates(subset="business_id").reset_index(drop=True)

    matched = joined.dropna(subset=["tract_geoid"]).copy()
    matched = matched.drop(columns=["index_right"], errors="ignore")

    n_in = len(businesses)
    n_matched = len(matched)
    print(f"    businesses in            : {n_in:,}")
    print(f"    boundary duplicates fixed: {n_dupes:,}")
    print(f"    matched to a tract       : {n_matched:,} ({n_matched / n_in:.1%})")
    print(f"    dropped (outside US area): {n_in - n_matched:,}")
    return matched


# --- Diagnostics -----------------------------------------------------------
def report_dropped(businesses: gpd.GeoDataFrame, matched: gpd.GeoDataFrame) -> None:
    """Audit the US-only filter: expected drops are AB (Edmonton) plus the
    mislocated singletons, and nothing else."""
    dropped = businesses[~businesses["business_id"].isin(matched["business_id"])]
    print("[6] Dropped businesses (outside the US study states = US-only filter)")
    print(f"    dropped total             : {len(dropped):,}")
    if len(dropped):
        print("    by reported state (top 15):")
        vc = dropped["state"].value_counts().head(15)
        for st, n in vc.items():
            print(f"      {str(st):4s}: {n:,}")


def report_per_state(matched: gpd.GeoDataFrame) -> None:
    """Matched businesses per study state (derived from the tract GEOID prefix)."""
    sfips = matched["tract_geoid"].str[:2]
    print("[7] Matched businesses by state (tract GEOID prefix)")
    vc = sfips.value_counts()
    for fips, n in vc.items():
        label = config.STATE_LABELS.get(fips, "(unexpected)")
        print(f"      {fips}  {label:42s} {n:>7,}")


def report_per_tract(matched: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame) -> None:
    """Businesses-per-tract distribution — the first read on location sparsity."""
    counts = matched.groupby("tract_geoid").size()
    n_nonempty = counts.shape[0]
    print("[8] Businesses per tract (across all study states)")
    print(counts.describe().to_string().replace("\n", "\n      "))
    print(f"    tracts with >= 1 business : {n_nonempty:,} of {len(tracts):,} fetched")


def plot_overlay(matched: gpd.GeoDataFrame, acs_year: int) -> None:
    """National dot map of matched businesses, coloured by state.

    Drawing all ~50k tract boundaries would be an unreadable smear, so I plot
    the matched points: metros should show as clusters, with no stray dots
    offshore or in Canada.
    """
    out = out_figure(acs_year)
    fig, ax = plt.subplots(figsize=(13, 7))
    sfips = matched["tract_geoid"].str[:2]
    for fips, idx in sfips.groupby(sfips).groups.items():
        g = matched.loc[idx]
        ax.scatter(g.geometry.x, g.geometry.y, s=1, alpha=0.35,
                   label=config.STATE_LABELS.get(fips, fips))
    ax.set_aspect("equal")
    ax.set_title(
        f"Stage 1 — {len(matched):,} US businesses matched to tracts "
        f"({len(config.STATES)} states, ACS {acs_year} boundaries)"
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(markerscale=6, fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"    figure saved             : {out}")


# --- Orchestration ---------------------------------------------------------
def run_vintage(businesses: gpd.GeoDataFrame, acs_year: int) -> None:
    """Fetch one vintage's boundaries, join, validate, and save its output."""
    print(f"\n=== ACS vintage {acs_year} "
          f"(reviews {config.REVIEW_WINDOWS[acs_year][0]}-"
          f"{config.REVIEW_WINDOWS[acs_year][1]}) ===")
    tracts = fetch_tracts(year=acs_year)
    aligned = align_crs(businesses, tracts)
    matched = spatial_join(aligned, tracts)

    report_dropped(aligned, matched)
    report_per_state(matched)
    report_per_tract(matched, tracts)
    plot_overlay(matched, acs_year)

    out = out_parquet(acs_year)
    matched.to_parquet(out)
    print(f"Saved {len(matched):,} businesses -> {out}")


# --- Full (nationwide) business attributes — no tract join -----------------
def validate_full(df: pd.DataFrame) -> None:
    """A metro/state/ratings read on the nationwide business attribute table."""
    print("[2] Validation (nationwide business attributes)")
    print(f"    businesses                : {len(df):,}")
    print(f"    distinct business_id      : {df['business_id'].nunique():,}")

    print("\n    businesses by state (top 12):")
    print(df["state"].value_counts().head(12).to_string().replace("\n", "\n      "))

    print("\n    businesses by city (top 12):")
    print(df["city"].value_counts().head(12).to_string().replace("\n", "\n      "))

    print("\n    stars / review_count:")
    print(
        df[["stars", "review_count"]].describe().to_string().replace("\n", "\n      ")
    )

    n_no_cat = int(df["categories"].isna().sum())
    print(f"\n    missing categories        : {n_no_cat:,}")


def run_full() -> None:
    """Export every business's attributes nationwide (no spatial join/filter).

    The geographic filter lives entirely in the tract join, so --full is just
    load + validate + save.
    """
    print("=== FULL: nationwide business attributes (no tract join) ===")
    df = load_businesses()
    validate_full(df)
    df.to_parquet(BUSINESSES_FULL_PARQUET, index=False)
    print(f"\nSaved {len(df):,} businesses -> {BUSINESSES_FULL_PARQUET}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--full", action="store_true",
        help="export every business's attributes nationwide (no tract join, no "
             "geographic filter) -> data/processed/businesses_full.parquet",
    )
    args = parser.parse_args()

    config.ensure_dirs()

    if args.full:
        run_full()
        return

    # Load and build points once; the per-vintage loop only redoes what depends
    # on the boundary year.
    businesses = to_geodataframe(load_businesses())
    for acs_year in config.ACS_YEARS:
        run_vintage(businesses, acs_year)


if __name__ == "__main__":
    main()
