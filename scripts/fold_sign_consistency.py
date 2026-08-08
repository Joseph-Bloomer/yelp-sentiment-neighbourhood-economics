"""Per-fold sign consistency for the Stage 6 ladder.

Pooled out-of-sample R^2 is implicitly fold-size weighted, and test folds span
45 to 1,263 tracts — one large fold could carry a positive pooled increment.
This asks instead: in how many of the ten folds does the richer design beat the
leaner one? The sign is well defined because, for a fixed (target, fold), the
held-out set is identical across designs (fold membership never depends on
feature columns), so SS_tot is shared and sign(delta_R2) = sign(SSE_B - SSE_A).
"Positive" = the richer design wins that fold.

Folds, ridge and scoring all come from ``pipeline.stage6_evaluation``; the only
new code is a per-tract prediction wrapper whose pooled R^2 is asserted equal,
to machine precision, to ladder_results.csv and sec3a_solo_over_stars.csv.

Writes to data/processed/secondary/: sec7_fold_predictions.parquet (per-tract
held-out predictions), sec7_fold_signs.csv (folds won + exact one-sided sign
test + pooled delta R^2), sec7_fold_deltas.csv (per-fold SS_tot/SSE/delta).
Primary artefacts are read-only; their mtimes are asserted unchanged.

Run: .venv\\Scripts\\python.exe scripts/fold_sign_consistency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Project root on the path so `import config` / `import pipeline...` work from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import r2_score

import config
import pipeline.stage6_evaluation as s6  # single source of truth for folds/ridge/scoring

assert len(config.ACS_YEARS) == 1, "single-vintage, like Stage 6"
ACS_YEAR = config.ACS_YEARS[0]

SECONDARY_DIR = config.PROCESSED_DIR / "secondary"

# Primary artefacts this script must never touch; mtimes re-checked at the end.
PRIMARY_ARTEFACTS = [
    config.PROCESSED_DIR / "ladder_results.csv",
    config.PROCESSED_DIR / f"tract_features_{ACS_YEAR}.parquet",
    config.FIGURES_DIR / "stage6_ladder.png",
]

# Machine-precision tolerance for the self-checks.
RTOL, ATOL = 1e-10, 1e-12


# --- Per-tract held-out predictions ------------------------------------------
def oos_predictions(frame, feature_cols, target, folds, make_estimator) -> pd.DataFrame:
    """Per-tract held-out predictions under the ``s6.spatial_cv`` protocol.
    Thin glue only; pooled R^2 is asserted against the canonical values
    downstream, so it cannot silently drift from the stage."""
    fi = frame.set_index("tract_geoid")
    rows = []
    for fold in folds:
        tr = [t for t in fold["train"] if pd.notna(fi.at[t, target])]
        te = [t for t in fold["test"] if pd.notna(fi.at[t, target])]
        if len(te) == 0 or len(tr) < s6.MIN_TRAIN_TRACTS:
            continue
        ytr = fi.loc[tr, target].to_numpy(dtype=float)
        if feature_cols:
            est = make_estimator()
            est.fit(fi.loc[tr, feature_cols].to_numpy(dtype=float), ytr)
            pred = est.predict(fi.loc[te, feature_cols].to_numpy(dtype=float))
        else:
            # Naive rung: predict the training-fold mean.
            pred = np.full(len(te), ytr.mean())
        yte = fi.loc[te, target].to_numpy(dtype=float)
        for t, y_t, y_p in zip(te, yte, np.asarray(pred, dtype=float)):
            rows.append({"tract_geoid": t, "fold_id": int(fold["block"]),
                         "y_true": float(y_t), "y_pred": float(y_p)})
    return pd.DataFrame(rows)


# --- Rebuild the primary folds -----------------------------------------------
def build_analysis_and_folds():
    """Reproduce the primary Stage 6 analysis frame and spatial folds verbatim:
    >=20-review flag, k-means 10 blocks on EPSG:5070 centroids, queen buffer,
    RANDOM_STATE=0. ``validate_folds`` asserts zero train/test adjacency."""
    merged = s6.load_joined(ACS_YEAR, smoke=False)
    analysis = s6.apply_min_reviews(merged)  # no override -> Stage 5's stored flag (primary)
    analysis_ids = analysis["tract_geoid"].tolist()

    tracts = s6.load_tract_boundaries(ACS_YEAR)  # cached parquet, no network
    tracts_analysis = tracts[tracts["tract_geoid"].isin(analysis_ids)].reset_index(drop=True)
    missing = set(analysis_ids) - set(tracts_analysis["tract_geoid"])
    assert not missing, f"{len(missing)} analysis tracts lack a boundary polygon"

    neighbours = s6.queen_neighbours(tracts_analysis)
    block_of = s6.assign_blocks(tracts_analysis)      # k-means, EPSG:5070, random_state=0
    folds = s6.build_folds(analysis_ids, block_of, neighbours)
    s6.validate_folds(folds, neighbours, analysis_ids)  # asserts zero adjacency violations
    return analysis, folds


# --- Design catalogue (feature sets from the rung specs) ---------------------
def build_designs(analysis) -> dict[str, list[str]]:
    """The ten named designs the sign test covers, from ``s6.rung_specs()``.
    Two pairs share a feature set (rung3_stars == solo_baseline, rung4_vader ==
    solo_vader); both are kept and their prediction-identity is asserted."""
    rc = {name: cols for _, name, cols in s6.rung_specs()}
    counts = s6.usable_columns(analysis, rc["counts"])
    stars = s6.usable_columns(analysis, rc["stars"])
    vader = s6.usable_columns(analysis, rc["vader"])
    siebert = s6.usable_columns(analysis, rc["siebert"])
    absa = s6.usable_columns(analysis, rc["absa"])
    full = counts + stars + vader + siebert + absa
    assert len(full) == 16, full

    return {
        # nested ladder rungs 1-6
        "rung1_naive":   [],
        "rung2_counts":  counts,
        "rung3_stars":   counts + stars,
        "rung4_vader":   counts + stars + vader,
        "rung5_siebert": counts + stars + vader + siebert,
        "rung6_absa":    full,
        # sec3a "solo" designs (counts + stars + exactly one method) + baseline
        "solo_baseline": counts + stars,
        "solo_vader":    counts + stars + vader,
        "solo_siebert":  counts + stars + siebert,
        "solo_absa":     counts + stars + absa,
    }


# Self-check anchors: each design's pooled R^2 must equal a canonical CSV value.
LADDER_RUNG = {"rung1_naive": 1, "rung2_counts": 2, "rung3_stars": 3,
               "rung4_vader": 4, "rung5_siebert": 5, "rung6_absa": 6}
SEC3A_MODEL = {"solo_baseline": "counts+stars (baseline)",
               "solo_vader": "counts+stars+VADER",
               "solo_siebert": "counts+stars+SiEBERT",
               "solo_absa": "counts+stars+ABSA"}

# Comparisons: (name, design_A, design_B). "Positive" = A beats B in that fold.
COMPARISONS = [
    ("rung6_vs_rung3", "rung6_absa", "rung3_stars"),
    ("rung6_vs_rung5", "rung6_absa", "rung5_siebert"),
    ("rung5_vs_rung4", "rung5_siebert", "rung4_vader"),
    ("solo_vader", "solo_vader", "solo_baseline"),
    ("solo_siebert", "solo_siebert", "solo_baseline"),
    ("solo_absa", "solo_absa", "solo_baseline"),
]


# --- Step 1: predictions + self-check ----------------------------------------
def compute_predictions(analysis, folds, designs):
    frames = []
    for target, _label, _ in s6.TARGETS:
        for design, cols in designs.items():
            p = oos_predictions(analysis, cols, target, folds, s6.ridge_pipeline)
            p.insert(0, "design", design)
            p.insert(0, "target", target)
            frames.append(p)
    preds = pd.concat(frames, ignore_index=True)
    return preds[["target", "design", "tract_geoid", "fold_id", "y_true", "y_pred"]]


def self_check_pooled_r2(preds, designs) -> dict:
    """Assert pooled R^2 matches the canonical anchors and duplicate designs give
    identical predictions. Returns {(target, design): r2}."""
    ladder = pd.read_csv(config.PROCESSED_DIR / "ladder_results.csv")
    ladder = ladder[ladder["predictor"] == "ridge"]
    r2_ladder = {(r.target, int(r.rung)): float(r.r2) for r in ladder.itertuples()}

    sec3a = pd.read_csv(SECONDARY_DIR / "sec3a_solo_over_stars.csv")
    label_to_key = {label: key for key, label, _ in s6.TARGETS}
    r2_sec3a = {(label_to_key[r.target_label], r.model): float(r.r2)
                for r in sec3a.itertuples()}

    pooled = {}
    max_abs = 0.0
    for target, _label, _ in s6.TARGETS:
        for design in designs:
            sub = preds[(preds["target"] == target) & (preds["design"] == design)]
            r2p = float(r2_score(sub["y_true"], sub["y_pred"]))
            pooled[(target, design)] = r2p
            if design in LADDER_RUNG:
                ref = r2_ladder[(target, LADDER_RUNG[design])]
            else:
                ref = r2_sec3a[(target, SEC3A_MODEL[design])]
            max_abs = max(max_abs, abs(r2p - ref))
            assert np.isclose(r2p, ref, rtol=RTOL, atol=ATOL), \
                f"pooled R2 mismatch: {target}/{design} recomputed {r2p} vs canonical {ref}"

    # Duplicate designs must produce bit-identical held-out predictions.
    for a, b in (("rung3_stars", "solo_baseline"), ("rung4_vader", "solo_vader")):
        pa = preds[preds["design"] == a].set_index(["target", "tract_geoid"])["y_pred"]
        pb = preds[preds["design"] == b].set_index(["target", "tract_geoid"])["y_pred"]
        pb = pb.reindex(pa.index)
        assert np.array_equal(pa.to_numpy(), pb.to_numpy()), \
            f"duplicate designs disagree: {a} vs {b}"

    print(f"SELF-CHECK PASS: pooled R2 matches ladder_results.csv + sec3a on all "
          f"{len(designs)} designs x 4 targets; max |dR2| = {max_abs:.3e}")
    print("SELF-CHECK PASS: duplicate designs (rung3==solo_baseline, "
          "rung4==solo_vader) give identical predictions")
    return pooled


# --- Step 2: per-fold SSE, sign counts, sign test ----------------------------
def fold_deltas_for(preds, target, design_a, design_b) -> pd.DataFrame:
    """Per-fold SS_tot / SSE_A / SSE_B / delta_R2 / sign for one comparison.
    Merge on tract_geoid; fold_id and y_true must agree before differencing."""
    da = preds[(preds["target"] == target) & (preds["design"] == design_a)]
    db = preds[(preds["target"] == target) & (preds["design"] == design_b)]
    m = da.merge(db, on="tract_geoid", suffixes=("_a", "_b"))
    assert len(m) == len(da) == len(db), "designs cover different tracts (should be identical)"
    assert (m["fold_id_a"] == m["fold_id_b"]).all(), "fold_id disagreement across designs"
    assert np.allclose(m["y_true_a"], m["y_true_b"], rtol=0, atol=0), "y_true disagreement"

    rows = []
    for fold_id, g in m.groupby("fold_id_a", sort=True):
        yt = g["y_true_a"].to_numpy(dtype=float)
        ss_tot = float(((yt - yt.mean()) ** 2).sum())
        sse_a = float(((yt - g["y_pred_a"].to_numpy(dtype=float)) ** 2).sum())
        sse_b = float(((yt - g["y_pred_b"].to_numpy(dtype=float)) ** 2).sum())
        delta_r2 = (sse_b - sse_a) / ss_tot if ss_tot > 0 else np.nan
        sign = 1 if sse_a < sse_b else (-1 if sse_a > sse_b else 0)
        rows.append({"fold_id": int(fold_id), "n_test": int(len(g)),
                     "SS_tot": ss_tot, "SSE_A": sse_a, "SSE_B": sse_b,
                     "delta_r2": delta_r2, "sign": sign})
    return pd.DataFrame(rows)


def run_sign_test(preds, pooled):
    """Build sec7_fold_signs (summary) and sec7_fold_deltas (per-fold long)."""
    sign_rows, delta_rows = [], []
    for target, _label, _ in s6.TARGETS:
        for comparison, da, db in COMPARISONS:
            fd = fold_deltas_for(preds, target, da, db)
            fd.insert(0, "comparison", comparison)
            fd.insert(0, "target", target)
            delta_rows.append(fd)

            n_eval = int(len(fd))
            n_pos = int((fd["sign"] == 1).sum())
            p_one = float(binomtest(n_pos, n_eval, 0.5, alternative="greater").pvalue)
            pooled_delta = pooled[(target, da)] - pooled[(target, db)]

            # Pooled delta R^2 is an SS_tot-weighted mix of the per-fold deltas,
            # so winning every fold must imply a positive pooled increment.
            if n_pos == n_eval:
                assert pooled_delta > 0, (
                    f"sign inconsistency: {target}/{comparison} won all "
                    f"{n_eval} folds yet pooled delta R2 = {pooled_delta:.6g} <= 0"
                )

            sign_rows.append({
                "target": target, "comparison": comparison,
                "n_folds_positive": n_pos, "n_folds_evaluated": n_eval,
                "p_one_sided": p_one, "pooled_delta_r2": pooled_delta,
            })
    signs = pd.DataFrame(sign_rows)
    deltas = pd.concat(delta_rows, ignore_index=True)
    return signs, deltas


def cross_check_pooled_deltas(signs) -> None:
    """Anchor pooled_delta_r2 against the canonical increments: rung6_vs_rung3 =
    the ladder's rung6-rung3 gap, each solo_* = sec3a's delta_over_counts_stars.
    Pins the differenced quantity the tables actually show."""
    ladder = pd.read_csv(config.PROCESSED_DIR / "ladder_results.csv")
    ladder = ladder[ladder["predictor"] == "ridge"]
    r2 = {(r.target, int(r.rung)): float(r.r2) for r in ladder.itertuples()}
    sec3a = pd.read_csv(SECONDARY_DIR / "sec3a_solo_over_stars.csv")
    label_to_key = {label: key for key, label, _ in s6.TARGETS}
    d3a = {(label_to_key[r.target_label], r.model): float(r.delta_over_counts_stars)
           for r in sec3a.itertuples()}
    solo_model = {"solo_vader": "counts+stars+VADER",
                  "solo_siebert": "counts+stars+SiEBERT",
                  "solo_absa": "counts+stars+ABSA"}
    for r in signs.itertuples():
        if r.comparison == "rung6_vs_rung3":
            ref = r2[(r.target, 6)] - r2[(r.target, 3)]
        elif r.comparison in solo_model:
            ref = d3a[(r.target, solo_model[r.comparison])]
        else:
            continue
        assert np.isclose(r.pooled_delta_r2, ref, rtol=RTOL, atol=ATOL), \
            f"pooled delta mismatch: {r.target}/{r.comparison} {r.pooled_delta_r2} vs {ref}"
    print("SELF-CHECK PASS: pooled delta R2 matches the ladder Delta-text and "
          "sec3a deltas to machine precision")


# --- mtime guard -------------------------------------------------------------
def snapshot_primary_mtimes() -> dict:
    return {p: p.stat().st_mtime_ns for p in PRIMARY_ARTEFACTS}


def assert_primary_untouched(snapshot) -> None:
    for p, mtime in snapshot.items():
        assert p.stat().st_mtime_ns == mtime, f"primary artefact modified: {p}"
    print("\nprimary artefacts verified untouched (mtime unchanged):")
    for p in snapshot:
        print(f"    {p.relative_to(config.PROJECT_ROOT)}")


# --- Orchestration -----------------------------------------------------------
def main() -> None:
    SECONDARY_DIR.mkdir(parents=True, exist_ok=True)
    mtimes = snapshot_primary_mtimes()

    print("[1] Rebuilding primary analysis frame + spatial folds")
    analysis, folds = build_analysis_and_folds()
    designs = build_designs(analysis)
    print(f"    {len(designs)} designs, {len(folds)} folds, "
          f"{len(analysis):,} analysis tracts")

    print("\n[2] Per-tract held-out predictions for every design x target")
    preds = compute_predictions(analysis, folds, designs)
    pred_path = SECONDARY_DIR / "sec7_fold_predictions.parquet"
    preds.to_parquet(pred_path, index=False)
    print(f"    {len(preds):,} rows -> {pred_path.relative_to(config.PROJECT_ROOT)}")

    print("\n[3] Self-checks against the canonical artefacts")
    pooled = self_check_pooled_r2(preds, designs)

    print("\n[4] Per-fold SSE, sign counts, exact one-sided sign test")
    signs, deltas = run_sign_test(preds, pooled)
    cross_check_pooled_deltas(signs)

    signs_path = SECONDARY_DIR / "sec7_fold_signs.csv"
    deltas_path = SECONDARY_DIR / "sec7_fold_deltas.csv"
    signs.to_csv(signs_path, index=False)
    deltas.to_csv(deltas_path, index=False)

    # --- Print sec7_fold_signs.csv in full -----------------------------------
    print("\n" + "=" * 78)
    print("sec7_fold_signs.csv")
    print("=" * 78)
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:.6f}"):
        print(signs.to_string(index=False))
    print(f"\n    -> {signs_path.relative_to(config.PROJECT_ROOT)}")
    print(f"    -> {deltas_path.relative_to(config.PROJECT_ROOT)}  "
          f"({len(deltas):,} per-fold rows)")

    assert_primary_untouched(mtimes)
    print("\nDone. sec7_fold_predictions.parquet + sec7_fold_signs.csv + "
          "sec7_fold_deltas.csv written.")


if __name__ == "__main__":
    main()
