"""Stage 6 --min-reviews sensitivity sweep.

Runs pipeline/stage6_evaluation.py --min-reviews N per threshold (default
10/20/50) and consolidates the per-threshold ladders into one wide CSV: one row
per target x threshold, with ridge R2 per rung, the RandomForest row, the
text-over-stars delta (+absa minus +stars) and the best rung. Primary artefacts
are never touched; only the _minrev{N} outputs are read.

Run: .venv\\Scripts\\python.exe scripts/minrev_sensitivity_summary.py
     [--thresholds 10 20 50] [--skip-run]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Project root on the path so `import config` works from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

RUNG_ORDER = ["naive", "counts", "stars", "vader", "siebert", "absa"]
RUNG_COLUMNS = {r: ("+" + r if r != "naive" else r) for r in RUNG_ORDER}


def run_stage6(min_reviews: int) -> None:
    print(f"\n=== stage6_evaluation.py --min-reviews {min_reviews} ===")
    result = subprocess.run(
        [sys.executable, "-m", "pipeline.stage6_evaluation", "--min-reviews", str(min_reviews)],
        cwd=config.PROJECT_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"stage6_evaluation.py --min-reviews {min_reviews} exited with code {result.returncode}"
        )


def load_ladder(min_reviews: int) -> pd.DataFrame:
    path = config.PROCESSED_DIR / f"ladder_results_minrev{min_reviews}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run without --skip-run first, or check --thresholds."
        )
    df = pd.read_csv(path)
    df.insert(0, "min_reviews", min_reviews)
    return df


def self_check(long_df: pd.DataFrame) -> None:
    """--min-reviews 20 must reproduce the primary ladder to machine precision."""
    primary_path = config.PROCESSED_DIR / "ladder_results.csv"
    if 20 not in long_df["min_reviews"].values or not primary_path.exists():
        return
    primary = pd.read_csv(primary_path)
    minrev20 = long_df[long_df["min_reviews"] == 20]
    merged = primary.merge(
        minrev20, on=["target", "rung", "predictor"], suffixes=("_primary", "_minrev20")
    )
    max_diff = (merged["r2_primary"] - merged["r2_minrev20"]).abs().max()
    status = "PASS" if max_diff == 0 else "FAIL"
    print(f"\n[Self-check] --min-reviews 20 vs primary ladder_results.csv: "
          f"max |r2 diff| = {max_diff:.3e}  [{status}]")
    if status == "FAIL":
        raise AssertionError("--min-reviews 20 does not reproduce the primary ladder to machine precision")


def build_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    ridge = long_df[long_df["predictor"] == "ridge"]
    rf = long_df[long_df["predictor"] == "random_forest"]

    pivot = ridge.pivot_table(index=["min_reviews", "target_label"], columns="rung_name", values="r2")
    pivot = pivot[RUNG_ORDER].rename(columns=RUNG_COLUMNS)

    n_tracts = ridge.groupby(["min_reviews", "target_label"])["n_test_tracts"].first().rename("n_tracts")
    rf_r2 = rf.set_index(["min_reviews", "target_label"])["r2"].rename("rf")

    summary = pivot.join(n_tracts).join(rf_r2).reset_index()
    summary["text_over_stars_delta"] = summary["+absa"] - summary["+stars"]
    summary["best_rung"] = summary[list(RUNG_COLUMNS.values())].idxmax(axis=1)
    return summary.sort_values(["target_label", "min_reviews"]).reset_index(drop=True)


def report_verdicts(summary: pd.DataFrame) -> None:
    best_is_absa = (summary["best_rung"] == "+absa").all()
    delta_positive = (summary["text_over_stars_delta"] > 0).all()
    n_cells = len(summary)
    print(f"\n(a) ABSA best ridge rung on every target x threshold cell? "
          f"{'YES' if best_is_absa else 'NO'} "
          f"({(summary['best_rung'] == '+absa').sum()}/{n_cells})")
    print(f"(b) Text-over-stars delta positive in every cell? "
          f"{'YES' if delta_positive else 'NO'} "
          f"(range {summary['text_over_stars_delta'].min():+.3f} to "
          f"{summary['text_over_stars_delta'].max():+.3f})")
    non_absa_best = summary[summary["best_rung"] != "+absa"]
    if len(non_absa_best):
        print("(c) Cells where ABSA is not the best rung:")
        print(non_absa_best[["target_label", "min_reviews", "best_rung"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[10, 20, 50],
                         help="--min-reviews values to sweep (default: 10 20 50)")
    parser.add_argument("--skip-run", action="store_true",
                         help="Reuse existing ladder_results_minrev{N}.csv instead of re-running stage6_evaluation.py")
    args = parser.parse_args()

    if not args.skip_run:
        for n in args.thresholds:
            run_stage6(n)

    long_df = pd.concat([load_ladder(n) for n in args.thresholds], ignore_index=True)
    self_check(long_df)

    summary = build_summary(long_df)

    out_path = config.PROCESSED_DIR / "minrev_sensitivity_summary.csv"
    summary.to_csv(out_path, index=False, float_format="%.4f")

    print(f"\n=== Min-reviews sensitivity summary (thresholds: {', '.join(map(str, args.thresholds))}) ===")
    print(summary.to_string(index=False))
    report_verdicts(summary)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
