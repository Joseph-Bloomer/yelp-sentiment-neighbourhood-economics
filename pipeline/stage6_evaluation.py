"""Stage 6 — the evaluation ladder (single pre-COVID vintage, ACS 2018).

Laptop, CPU — small n, ridge models plus one forest. Stage 5 builds the
features; this stage scores them: do richer sentiment methods buy
out-of-sample accuracy for tract economic outcomes — and does text sentiment
add anything over the free star rating?

Nested feature ladder: one predictor, one set of CV folds, and the only thing
that changes per rung is which columns the model sees, so a jump at a rung is
attributable to that rung's features and nothing else.

    1. naive     — predict the (training-fold) mean. The floor.
    2. + counts  — business / review volume          (replicates Glaeser et al.)
    3. + stars   — mean star rating                   (the free baseline signal)
    4. + VADER    — lexicon document sentiment
    5. + SiEBERT  — transformer document sentiment
    6. + ABSA    — LLM aspect sentiment (food/service/price/ambience/location)

Spatial CV, not a random split: tract outcomes are spatially autocorrelated,
so a random split leaks — a test tract's neighbours sit in training and the
model interpolates. I group tracts into k-means blocks on centroids and leave
one block out at a time; each fold also drops from training every tract
queen-adjacent to the held-out block (a one-ring buffer, Roberts et al. 2017),
so no train tract shares a boundary with a test tract. The same folds serve
every rung and the tree check, so comparisons are like-for-like.

Tracts below MIN_REVIEWS_PER_TRACT (Stage 5's meets_min_reviews flag) carry
sentiment means from a handful of reviews — too noisy — so I exclude them from
the primary ladder. The count is reported at the filter step, and the
threshold stays a single-knob robustness dial.

Tree robustness: a RandomForest on the full rung-6 feature set, same folds,
checks whether a non-linear learner finds signal the ridge ladder misses.

Outputs:
    data/processed/ladder_results.csv   one row per (target x rung) + the RF
                                        row: OOS R2/RMSE, increment over the
                                        previous rung, gain over naive, folds.
    figures/stage6_ladder.png           the ladder across all four targets.

--smoke runs the same pipeline on the *_smoke feature file to prove plumbing;
numbers are meaningless and most method rungs skip for want of populated
columns (which exercises the skip logic too).
"""
from __future__ import annotations

import argparse
import time
import warnings

import matplotlib

matplotlib.use("Agg")  # headless — save figures, never display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from pipeline.stage4_common import ASPECTS

# --- Tunable knobs ----------------------------------------------------------
# Spatial blocks = CV folds. ~3,400 usable tracts / 10 blocks => ~340 held out
# per fold — enough for a stable per-fold read, few enough that the buffer
# still leaves a large training set. (With far-apart metros, k-means on
# centroids largely recovers metros as blocks anyway.)
N_SPATIAL_BLOCKS = 10
RANDOM_STATE = 0

# Skip a fold whose (post-buffer, non-NaN-target) training set is smaller than
# this — too little to fit 16 features without the result being noise.
MIN_TRAIN_TRACTS = 30

# RidgeCV picks alpha by leave-one-out GCV on the training fold only —
# no leakage into the held-out block.
RIDGE_ALPHAS = np.logspace(-3, 4, 22)

# NAD83 / Conus Albers (metres): equal-area across the contiguous US, so
# centroids are undistorted in every metro — unlike UTM 18N (a Philadelphia
# pilot leftover) which skewed the western metros' blocks. k-means input only.
METRIC_CRS = "EPSG:5070"

# The four ACS outcome columns, with display label and a units tag
# (RMSE formatting only).
TARGETS = [
    ("median_household_income", "Median household income", "usd"),
    ("poverty_rate", "Poverty rate", "prop"),
    ("unemployment_rate", "Unemployment rate", "prop"),
    ("median_gross_rent", "Median gross rent", "usd"),
]

# x-axis tick labels for the ladder figure, indexed by rung number.
RUNG_TICK = {1: "naive", 2: "+counts", 3: "+stars",
             4: "+VADER", 5: "+SiEBERT", 6: "+ABSA"}


# --- Path helpers -----------------------------------------------------------
def features_parquet(acs_year: int, smoke: bool):
    suffix = "_smoke" if smoke else ""
    return config.PROCESSED_DIR / f"tract_features_{acs_year}{suffix}.parquet"


def boundaries_cache(acs_year: int):
    # Cached tract polygons so only the first run touches the network.
    return config.INTERIM_DIR / f"tract_boundaries_{acs_year}.parquet"


def _out_suffix(smoke: bool, min_reviews: int | None) -> str:
    """Empty for the primary run (canonical output names); `_minrev{N}` only
    for a --min-reviews sensitivity run, so a sweep never overwrites the
    primary artefacts."""
    suffix = "_smoke" if smoke else ""
    if min_reviews is not None:
        suffix += f"_minrev{min_reviews}"
    return suffix


def results_csv(smoke: bool, min_reviews: int | None = None):
    return config.PROCESSED_DIR / f"ladder_results{_out_suffix(smoke, min_reviews)}.csv"


def fig_path(smoke: bool, min_reviews: int | None = None):
    return config.FIGURES_DIR / f"stage6_ladder{_out_suffix(smoke, min_reviews)}.png"


# --- Rung definitions (nested, additive) ------------------------------------
def rung_specs() -> list[tuple[int, str, list[str]]]:
    """The ladder: (rung number, name, the columns this rung adds).

    Cumulative — rung k sees the union of every rung <= k. Counts enter as
    log1p (volume is right-skewed; the form Glaeser et al. use). ABSA
    contributes mention_rate + polarity_mean per aspect — the pair by which
    Stage 5 encodes absent vs neutral.
    """
    absa_cols: list[str] = []
    for aspect in ASPECTS:
        absa_cols += [f"{aspect}_mention_rate", f"{aspect}_polarity_mean"]
    return [
        (1, "naive", []),
        (2, "counts", ["log_n_reviews", "log_n_businesses"]),
        (3, "stars", ["stars_mean"]),
        (4, "vader", ["vader_mean"]),
        (5, "siebert", ["siebert_share_pos", "siebert_mean_prob"]),
        (6, "absa", absa_cols),
    ]


# --- 1. Load + join inputs --------------------------------------------------
def load_joined(acs_year: int, smoke: bool) -> pd.DataFrame:
    """Join Stage 5 tract features to the Stage 3 ACS outcomes on GEOID,
    asserting a single vintage on both sides."""
    feats_path = features_parquet(acs_year, smoke)
    acs_path = config.PROCESSED_DIR / f"acs_tracts_{acs_year}.parquet"
    if not feats_path.exists():
        raise FileNotFoundError(
            f"Tract features not found: {feats_path}\n"
            "Run Stage 5 first"
            + (" with --smoke" if smoke else "")
            + ":  .venv\\Scripts\\python.exe -m pipeline.stage5_aggregate"
            + (" --smoke" if smoke else "")
        )
    if not acs_path.exists():
        raise FileNotFoundError(
            f"ACS outcomes not found: {acs_path}\n"
            "Run Stage 3 first:  .venv\\Scripts\\python.exe -m pipeline.stage3_acs"
        )

    feats = pd.read_parquet(feats_path)
    acs = pd.read_parquet(acs_path)
    feats["tract_geoid"] = feats["tract_geoid"].astype(str)
    acs["tract_geoid"] = acs["tract_geoid"].astype(str)

    # Single-vintage guard: exactly one ACS year, and the two tables agree.
    for name, df in (("features", feats), ("ACS", acs)):
        years = sorted(df["acs_year"].unique())
        assert len(years) == 1, f"{name} carries multiple vintages {years}; Stage 6 is single-vintage"
    assert feats["acs_year"].iloc[0] == acs["acs_year"].iloc[0] == acs_year, (
        f"vintage mismatch: features {feats['acs_year'].iloc[0]}, "
        f"ACS {acs['acs_year'].iloc[0]}, config {acs_year}"
    )

    target_cols = [t[0] for t in TARGETS]
    merged = feats.merge(
        acs[["tract_geoid", *target_cols]], on="tract_geoid", how="inner", validate="1:1"
    )

    print(f"[1] Load + join (ACS {acs_year}{', SMOKE' if smoke else ''})")
    print(f"    ACS tracts (outcome table)            : {len(acs):,}")
    print(f"    tracts with >= 1 windowed review      : {len(feats):,}")
    print(f"    joined (have both feature & outcome)  : {len(merged):,}")

    # Pre-compute the log-count features the counts rung uses.
    merged["log_n_reviews"] = np.log1p(merged["n_reviews"])
    merged["log_n_businesses"] = np.log1p(merged["n_businesses"])
    return merged


def apply_min_reviews(merged: pd.DataFrame, min_reviews: int | None = None) -> pd.DataFrame:
    """Exclude tracts below the review floor.

    Default: Stage 5's stored `meets_min_reviews` flag (the primary run). With
    --min-reviews N, recompute the mask as `n_reviews >= N` instead — passing
    20 reproduces the primary mask, the sweep's self-check.
    config.MIN_REVIEWS_PER_TRACT is never touched.
    """
    n_before = len(merged)
    if min_reviews is None:
        kept = merged[merged["meets_min_reviews"]].reset_index(drop=True)
        print(f"    meets_min_reviews (>= {config.MIN_REVIEWS_PER_TRACT} reviews) : "
              f"{len(kept):,} kept, {n_before - len(kept):,} excluded (too noisy)")
    else:
        kept = merged[merged["n_reviews"] >= min_reviews].reset_index(drop=True)
        print(f"    --min-reviews override (>= {min_reviews} reviews, recomputed from "
              f"n_reviews) : {len(kept):,} kept, {n_before - len(kept):,} excluded")
    print("    per-target tracts with a non-null outcome:")
    for col, label, _ in TARGETS:
        print(f"      {label:26s}: {int(kept[col].notna().sum()):>4,}")
    return kept


# --- 2. Geometry: tract polygons, centroids, queen adjacency ----------------
def load_tract_boundaries(acs_year: int) -> "object":
    """Tract polygons for the vintage (GEOID + geometry), cached after first pull."""
    import geopandas as gpd

    cache = boundaries_cache(acs_year)
    if cache.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return gpd.read_parquet(cache)

    import pygris

    print(f"    fetching tract boundaries via pygris (year {acs_year}, first run only)")
    layers = []
    for state_fips, _ in config.STATES:
        tr = pygris.tracts(state=state_fips, year=acs_year, cb=True)
        layers.append(tr)
    tracts = pd.concat(layers, ignore_index=True)
    tracts = gpd.GeoDataFrame(tracts, geometry="geometry", crs=layers[0].crs)
    tracts = tracts.rename(columns={"GEOID": "tract_geoid"})[["tract_geoid", "geometry"]]
    tracts["tract_geoid"] = tracts["tract_geoid"].astype(str)
    config.ensure_dirs()
    tracts.to_parquet(cache)
    return tracts


def queen_neighbours(tracts) -> dict[str, set[str]]:
    """Queen-contiguity neighbours per tract via a `touches` self-join —
    shared edge or vertex. A GEOID never touches itself, so no self-loops.
    """
    import geopandas as gpd

    left = tracts[["tract_geoid", "geometry"]].rename(columns={"tract_geoid": "geoid_a"})
    right = tracts[["tract_geoid", "geometry"]].rename(columns={"tract_geoid": "geoid_b"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pairs = gpd.sjoin(left, right, predicate="touches")
    neighbours: dict[str, set[str]] = {g: set() for g in tracts["tract_geoid"]}
    for a, b in zip(pairs["geoid_a"], pairs["geoid_b"]):
        if a != b:
            neighbours[a].add(b)
    return neighbours


def assign_blocks(tracts_analysis) -> dict[str, int]:
    """k-means on metric tract centroids -> a spatial block label per tract.
    Compact, near-contiguous partitions; cheap and reproducible.
    """
    import geopandas as gpd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cent = tracts_analysis.to_crs(METRIC_CRS).geometry.centroid
    xy = np.c_[cent.x.to_numpy(), cent.y.to_numpy()]
    k = min(N_SPATIAL_BLOCKS, len(tracts_analysis))
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(xy)
    return dict(zip(tracts_analysis["tract_geoid"].to_numpy(), km.labels_.tolist()))


def build_folds(
    tract_ids: list[str], block_of: dict[str, int], neighbours: dict[str, set[str]]
) -> list[dict]:
    """Leave-one-block-out folds with a one-ring contiguity buffer.

    Test = the block; buffer = every tract queen-adjacent to a test tract,
    dropped from training; train = the rest. Dropping the buffer is what makes
    the no-shared-boundary guarantee literal.
    """
    idset = set(tract_ids)
    folds = []
    for b in sorted(set(block_of[t] for t in tract_ids)):
        test = [t for t in tract_ids if block_of[t] == b]
        test_set = set(test)
        buffer = set()
        for t in test:
            buffer |= neighbours.get(t, set()) & idset
        buffer -= test_set
        train = [t for t in tract_ids if t not in test_set and t not in buffer]
        folds.append({"block": b, "test": test, "train": train, "buffer": sorted(buffer)})
    return folds


# --- 3. Cross-validated prediction ------------------------------------------
def ridge_pipeline() -> Pipeline:
    """Impute (mean) -> standardise -> RidgeCV. Standardising makes the single
    penalty comparable across features; keep_empty_features stops all-NaN
    columns silently shifting the feature count between folds."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="mean", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
    ])


def forest_pipeline() -> Pipeline:
    """Impute -> RandomForest (no scaling needed for trees)."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="mean", keep_empty_features=True)),
        ("rf", RandomForestRegressor(
            n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)),
    ])


def spatial_cv(
    frame: pd.DataFrame, feature_cols: list[str], target: str,
    folds: list[dict], make_estimator,
) -> dict:
    """Pooled out-of-sample prediction over the spatial folds.

    Each tract is held out exactly once, so concatenating the held-out
    predictions gives one OOS prediction per tract — the basis for the pooled
    R2/RMSE. Per-fold R2 is kept for the CV-detail columns. `make_estimator`
    is a zero-arg factory so each fold fits a fresh model.
    """
    fi = frame.set_index("tract_geoid")
    y_true, y_pred = [], []
    fold_r2, fold_sizes = [], []
    used_folds = 0
    for fold in folds:
        tr = [t for t in fold["train"] if pd.notna(fi.at[t, target])]
        te = [t for t in fold["test"] if pd.notna(fi.at[t, target])]
        if len(te) == 0 or len(tr) < MIN_TRAIN_TRACTS:
            continue
        used_folds += 1
        ytr = fi.loc[tr, target].to_numpy(dtype=float)
        yte = fi.loc[te, target].to_numpy(dtype=float)
        if feature_cols:
            est = make_estimator()
            est.fit(fi.loc[tr, feature_cols].to_numpy(dtype=float), ytr)
            pred = est.predict(fi.loc[te, feature_cols].to_numpy(dtype=float))
        else:
            # Naive rung: the training-fold mean, predicted for every test tract.
            pred = np.full(len(te), ytr.mean())
        y_true.extend(yte.tolist())
        y_pred.extend(np.asarray(pred, dtype=float).tolist())
        fold_sizes.append(len(te))
        fold_r2.append(r2_score(yte, pred) if (len(te) >= 2 and np.var(yte) > 0) else np.nan)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2 or np.var(y_true) == 0:
        return {"r2": np.nan, "rmse": np.nan, "n_test": len(y_true),
                "n_folds": used_folds, "fold_r2": fold_r2, "fold_sizes": fold_sizes}
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"r2": float(r2_score(y_true, y_pred)), "rmse": rmse, "n_test": len(y_true),
            "n_folds": used_folds, "fold_r2": fold_r2, "fold_sizes": fold_sizes}


# --- 4. Run the ladder ------------------------------------------------------
def usable_columns(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    """Columns that exist and carry at least one value.

    A missing Stage 4 score file arrives as an all-NaN column (Stage 5 keeps
    the schema); its rung is reported as skipped rather than fit on nothing.
    That lets Stage 6 run before every heavy scoring pass is in.
    """
    return [c for c in cols if c in frame.columns and frame[c].notna().any()]


def run_ladder(frame: pd.DataFrame, folds: list[dict]) -> pd.DataFrame:
    """The full ladder for every target, plus the RandomForest robustness row."""
    specs = rung_specs()
    rows = []
    for target, label, units in TARGETS:
        print(f"\n[4] Ladder - {label}")
        cumulative: list[str] = []
        prev = None          # last successfully-scored rung (for the increment)
        naive_r2 = None
        full_cols: list[str] = []  # rung-6 cumulative cols, for the RF check
        for rung, name, new_cols in specs:
            avail_new = usable_columns(frame, new_cols)
            cumulative = cumulative + avail_new
            full_cols = list(cumulative)
            skipped = bool(new_cols) and not avail_new  # wanted columns, none usable

            res = spatial_cv(frame, cumulative, target, folds, ridge_pipeline)
            if rung == 1:
                naive_r2 = res["r2"]

            r2_inc = np.nan if prev is None else res["r2"] - prev["r2"]
            rmse_inc = np.nan if prev is None else res["rmse"] - prev["rmse"]
            r2_vs_naive = np.nan if naive_r2 is None else res["r2"] - naive_r2

            status = "skipped (inputs missing)" if skipped else "ok"
            rows.append({
                "target": target, "target_label": label, "rung": rung,
                "rung_name": name, "predictor": "ridge", "status": status,
                "n_features": len(cumulative), "n_test_tracts": res["n_test"],
                "n_folds": res["n_folds"], "r2": res["r2"], "rmse": res["rmse"],
                "r2_increment": r2_inc, "rmse_increment": rmse_inc,
                "r2_vs_naive": r2_vs_naive,
                "fold_r2": _fmt_list(res["fold_r2"]),
                "fold_test_sizes": _fmt_list(res["fold_sizes"], ints=True),
                "features": ";".join(cumulative),
            })
            _print_rung(rung, name, res, r2_inc, r2_vs_naive, units, status)
            prev = res

        # --- Tree robustness: RandomForest on the full (rung-6) feature set ---
        rf = spatial_cv(frame, full_cols, target, folds, forest_pipeline)
        ridge_final = next(r for r in reversed(rows)
                           if r["target"] == target and r["predictor"] == "ridge")
        rows.append({
            "target": target, "target_label": label, "rung": 7,
            "rung_name": "tree_full", "predictor": "random_forest",
            "status": "robustness", "n_features": len(full_cols),
            "n_test_tracts": rf["n_test"], "n_folds": rf["n_folds"],
            "r2": rf["r2"], "rmse": rf["rmse"],
            "r2_increment": rf["r2"] - ridge_final["r2"],   # RF vs ridge final rung
            "rmse_increment": rf["rmse"] - ridge_final["rmse"],
            "r2_vs_naive": np.nan if naive_r2 is None else rf["r2"] - naive_r2,
            "fold_r2": _fmt_list(rf["fold_r2"]),
            "fold_test_sizes": _fmt_list(rf["fold_sizes"], ints=True),
            "features": ";".join(full_cols),
        })
        print(f"    RF (all {len(full_cols)} features)     : R2 {_f(rf['r2'])}  "
              f"(ridge final rung R2 {_f(ridge_final['r2'])}; "
              f"delta {_f(rf['r2'] - ridge_final['r2'])})")

    return pd.DataFrame(rows)


# --- 5. Validation ----------------------------------------------------------
def validate_folds(folds: list[dict], neighbours: dict[str, set[str]],
                   analysis_ids: list[str]) -> None:
    """Blocks partition the tracts; no fold leaks across the train/test
    boundary (the whole point of the buffer)."""
    print("\n[5] Spatial-fold validation")
    # (a) Test blocks partition the analysis tracts: disjoint and covering.
    test_union: list[str] = []
    for f in folds:
        test_union += f["test"]
    assert len(test_union) == len(set(test_union)), "test blocks overlap"
    assert set(test_union) == set(analysis_ids), "test blocks do not cover all tracts"
    sizes = [len(f["test"]) for f in folds]
    print(f"    {len(folds)} spatial blocks partition {len(analysis_ids)} tracts "
          f"(test sizes {min(sizes)}-{max(sizes)}, median {int(np.median(sizes))})")

    # (b) No train tract is queen-adjacent to any test tract (zero leakage).
    total_violations = 0
    for f in folds:
        test_set, train_set = set(f["test"]), set(f["train"])
        v = sum(1 for t in train_set for nb in neighbours.get(t, set()) if nb in test_set)
        total_violations += v
    status = "PASS" if total_violations == 0 else f"FAIL ({total_violations})"
    print(f"    train/test queen-adjacency violations : {total_violations}  [{status}]")
    print(f"    mean train size (after buffer)        : "
          f"{int(np.mean([len(f['train']) for f in folds]))}")
    print(f"    mean buffer size (dropped per fold)   : "
          f"{int(np.mean([len(f['buffer']) for f in folds]))}")
    assert total_violations == 0, "spatial CV leaks — buffer failed to remove adjacency"


def validate_floor(results: pd.DataFrame) -> None:
    """No ridge rung should score below the naive floor. A dip means the added
    features hurt OOS — a real, reportable finding, so this warns rather than
    crashes."""
    print("\n[6] Naive-floor check (ridge rungs)")
    ridge = results[results["predictor"] == "ridge"]
    any_below = False
    for target, label, _ in TARGETS:
        sub = ridge[ridge["target"] == target]
        naive = sub.loc[sub["rung"] == 1, "r2"]
        if naive.empty or pd.isna(naive.iloc[0]):
            print(f"    {label:26s}: naive R2 unavailable - skipped")
            continue
        naive_r2 = naive.iloc[0]
        below = sub[(sub["rung"] > 1) & (sub["r2"] < naive_r2 - 1e-9)]
        if len(below):
            any_below = True
            names = ", ".join(f"{r.rung_name}({_f(r.r2)})" for r in below.itertuples())
            print(f"    {label:26s}: naive R2 {_f(naive_r2)} - BELOW FLOOR: {names}")
        else:
            print(f"    {label:26s}: naive R2 {_f(naive_r2)} - floor respected [ok]")
    if not any_below:
        print("    all rungs >= their naive floor across all targets [ok]")


def report_text_over_stars(results: pd.DataFrame) -> None:
    """The headline test: does text sentiment (rungs 4-6) add anything over
    the free star rating (rung 3)?"""
    print("\n[7] Key test - does text add over stars? (stars=rung 3 -> +ABSA=rung 6)")
    ridge = results[results["predictor"] == "ridge"]
    for target, label, _ in TARGETS:
        sub = ridge[ridge["target"] == target].set_index("rung")
        if 3 not in sub.index or 6 not in sub.index:
            continue
        r3, r6 = sub.loc[3, "r2"], sub.loc[6, "r2"]
        steps = []
        for a, b, tag in [(3, 4, "VADER"), (4, 5, "SiEBERT"), (5, 6, "ABSA")]:
            if a in sub.index and b in sub.index:
                steps.append(f"+{tag} {_f(sub.loc[b, 'r2'] - sub.loc[a, 'r2']):>7}")
        verdict = "text adds signal" if (r6 - r3) > 0.005 else "no gain over stars"
        print(f"    {label:26s}: stars R2 {_f(r3)} -> full-text R2 {_f(r6)} "
              f"(delta {_f(r6 - r3)}; {verdict})")
        print(f"      {'step gains:':26s} " + "  ".join(steps))


def print_full_table(results: pd.DataFrame) -> None:
    print("\n[8] Full ladder table (OOS R2 / RMSE):")
    show = results.copy()
    show["R2"] = show["r2"].map(_f)
    show["dR2"] = show["r2_increment"].map(_f)
    show["R2-naive"] = show["r2_vs_naive"].map(_f)
    show["RMSE"] = show["rmse"].map(lambda v: f"{v:,.4g}" if pd.notna(v) else "  n/a")
    cols = ["target_label", "rung", "rung_name", "predictor", "status",
            "n_features", "n_test_tracts", "n_folds", "R2", "dR2", "R2-naive", "RMSE"]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(show[cols].to_string(index=False))


# --- 6. Figure --------------------------------------------------------------
def plot_ladder(results: pd.DataFrame, smoke: bool, min_reviews: int | None = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    ridge = results[results["predictor"] == "ridge"]
    rf = results[results["predictor"] == "random_forest"]
    for ax, (target, label, _) in zip(axes.ravel(), TARGETS):
        sub = ridge[ridge["target"] == target].sort_values("rung")
        x, y = sub["rung"].to_numpy(), sub["r2"].to_numpy()
        ok = ~np.isnan(y)
        ax.plot(x[ok], y[ok], "-o", color="#1f77b4", label="ridge ladder")

        naive_row = sub[sub["rung"] == 1]
        if len(naive_row) and pd.notna(naive_row["r2"].iloc[0]):
            ax.axhline(naive_row["r2"].iloc[0], color="0.6", ls=":",
                       label="naive floor")
        # Everything right of the stars rung is the "text over stars" region.
        ax.axvline(3, color="firebrick", ls="--", lw=0.8, alpha=0.7,
                   label="stars (free baseline)")
        rf_row = rf[rf["target"] == target]
        if len(rf_row) and pd.notna(rf_row["r2"].iloc[0]):
            ax.axhline(rf_row["r2"].iloc[0], color="seagreen", ls="-.",
                       label="RF (all features)")

        ax.set_title(label)
        ax.set_ylabel("out-of-sample $R^2$")
        ax.set_xticks(list(RUNG_TICK))
        ax.set_xticklabels([RUNG_TICK[r] for r in RUNG_TICK], rotation=30, ha="right")
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(fontsize=8, loc="best")
    title = "Stage 6 — evaluation ladder (spatially-blocked CV)"
    if min_reviews is not None:
        title += f"  [sensitivity: min {min_reviews} reviews/tract]"
    if smoke:
        title += "  [SMOKE — meaningless numbers, plumbing only]"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = fig_path(smoke, min_reviews)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n    figure -> {out}")


# --- small formatting helpers ----------------------------------------------
def _f(v) -> str:
    return "  n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.3f}"


def _fmt_list(values, ints: bool = False) -> str:
    if ints:
        return ";".join(str(int(v)) for v in values)
    return ";".join("nan" if (v is None or np.isnan(v)) else f"{v:.3f}" for v in values)


def _print_rung(rung, name, res, r2_inc, r2_vs_naive, units, status) -> None:
    rmse = "  n/a" if np.isnan(res["rmse"]) else (
        f"{res['rmse']:,.0f}" if units == "usd" else f"{res['rmse']:.4f}")
    tag = "  [skipped]" if status.startswith("skipped") else ""
    print(f"    {rung}. {name:8s} R2 {_f(res['r2'])}  dR2 {_f(r2_inc)}  "
          f"vs-naive {_f(r2_vs_naive)}  RMSE {rmse:>10}  "
          f"(n={res['n_test']}, {res['n_folds']} folds){tag}")


# --- Orchestration ----------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke", action="store_true",
        help="run against tract_features_*_smoke.parquet to prove plumbing "
             "(numbers meaningless); writes ladder_results_smoke.csv / "
             "stage6_ladder_smoke.png",
    )
    parser.add_argument(
        "--min-reviews", type=int, default=None, metavar="N",
        help="sensitivity override: filter analysis tracts on n_reviews >= N "
             "(recomputed from the features frame) instead of Stage 5's stored "
             "meets_min_reviews flag. Suffixes every output _minrev{N} so the "
             "primary artefacts are never touched; config.MIN_REVIEWS_PER_TRACT "
             "(20) stays put. Folds are rebuilt on this run's tract set.",
    )
    args = parser.parse_args()
    min_reviews = args.min_reviews
    if min_reviews is not None and min_reviews < 1:
        parser.error("--min-reviews must be a positive integer")

    config.ensure_dirs()
    t0 = time.perf_counter()

    assert len(config.ACS_YEARS) == 1, (
        f"Stage 6 is single-vintage; config.ACS_YEARS = {config.ACS_YEARS}"
    )
    acs_year = config.ACS_YEARS[0]

    # 1. Inputs + filtering.
    if min_reviews is not None:
        print(f"[0] SENSITIVITY RUN — analysis mask = n_reviews >= {min_reviews} "
              f"(overrides the stored meets_min_reviews flag; primary outputs "
              f"untouched; outputs suffixed _minrev{min_reviews})")
    merged = load_joined(acs_year, smoke=args.smoke)
    analysis = apply_min_reviews(merged, min_reviews=min_reviews)
    analysis_ids = analysis["tract_geoid"].tolist()
    if len(analysis_ids) < N_SPATIAL_BLOCKS:
        raise RuntimeError(
            f"only {len(analysis_ids)} usable tracts — fewer than "
            f"N_SPATIAL_BLOCKS={N_SPATIAL_BLOCKS}; nothing to cross-validate."
        )

    # 2. Geometry -> spatial blocks -> folds.
    print("\n[2] Geometry + spatial blocks")
    tracts = load_tract_boundaries(acs_year)
    tracts_analysis = tracts[tracts["tract_geoid"].isin(analysis_ids)].reset_index(drop=True)
    missing_geom = set(analysis_ids) - set(tracts_analysis["tract_geoid"])
    if missing_geom:
        # Tracts with reviews but no polygon are untestable; loud, not silent.
        # (Should be empty — both sides come from the same vintage.)
        print(f"    WARNING: {len(missing_geom)} analysis tracts lack a boundary "
              f"polygon and are dropped from CV: {sorted(missing_geom)[:5]}...")
        analysis = analysis[analysis["tract_geoid"].isin(tracts_analysis["tract_geoid"])]
        analysis_ids = analysis["tract_geoid"].tolist()
    neighbours = queen_neighbours(tracts_analysis)
    block_of = assign_blocks(tracts_analysis)
    folds = build_folds(analysis_ids, block_of, neighbours)
    print(f"    k-means blocks                        : {len(folds)} "
          f"(target {N_SPATIAL_BLOCKS})")

    # 3. Validate the folds before trusting any score that rests on them.
    validate_folds(folds, neighbours, analysis_ids)

    # 4. The ladder + RF robustness.
    results = run_ladder(analysis, folds)

    # 5. Headline checks + full table.
    validate_floor(results)
    report_text_over_stars(results)
    print_full_table(results)

    # 6. Persist.
    out_csv = results_csv(args.smoke, min_reviews)
    results.to_csv(out_csv, index=False)
    print(f"\n    results -> {out_csv}")
    plot_ladder(results, smoke=args.smoke, min_reviews=min_reviews)

    print(f"\nRuntime: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
