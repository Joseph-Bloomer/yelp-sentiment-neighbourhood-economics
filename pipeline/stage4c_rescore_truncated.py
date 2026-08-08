"""Stage 4c remediation — re-score the reviews the first full pass truncated.

The first full pass (38 h on the A40) capped reviews at 4000 collapsed chars;
11,094 reviews (0.18% of the ~6.10M scored) exceeded that and lost their tail.
config.LLM_MAX_REVIEW_CHARS is now 5000 — Yelp's own review-length cap and the
observed max collapsed length — so nothing real is clipped. This re-scores only
those reviews with their full text and overwrites their rows in
scores_llm.parquet; the other ~6.085M scores are untouched. Runs on the
cluster with the identical engine as the full pass (same model at full
precision, vLLM, same prompt/parser, temperature 0), so there is no
quant/precision confound on the re-scored 0.18%.

Two phases, so the re-score can be inspected before merging:

  # phase 1 — re-score -> data/processed/scores_llm_rescore.parquet
  tmux new -s rescore
  .venv/bin/python -m pipeline.stage4c_rescore_truncated

  # phase 2 — overwrite their rows in scores_llm.parquet (backs the original
  # up to scores_llm.pre_rescore_backup.parquet first)
  .venv/bin/python -m pipeline.stage4c_rescore_truncated --merge
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

import config
from pipeline.stage4_common import (
    ASPECTS,
    load_scoring_corpus,
    out_path,
    print_aspect_validation,
    resolve_device,
)
from pipeline.stage4c_llm import (
    build_generator,
    build_messages,
    build_scores_frame,
)

# Char cap the first full pass used. Selection keys off this, not the raised
# config.LLM_MAX_REVIEW_CHARS (5000): truncated iff collapsed length > 4000.
PRODUCTION_CAP = 4000

MAIN_OUT = out_path("scores_llm", smoke=False)            # data/processed/scores_llm.parquet
RESCORE_OUT = out_path("scores_llm_rescore", smoke=False)
BACKUP_OUT = config.PROCESSED_DIR / "scores_llm.pre_rescore_backup.parquet"

# Wide per-aspect columns, used for the change diff at merge time.
ASPECT_COLS = [f"{a}_{k}" for a in ASPECTS for k in ("mentioned", "polarity")]


def _collapsed_len(text) -> int:
    """Length after the same whitespace collapse build_messages applies, so
    this matches exactly the string that was, or was not, truncated."""
    return len(" ".join(str(text).split()))


def select_truncated(old_cap: int) -> pd.DataFrame:
    """Reviews the first pass truncated: collapsed length > old_cap, drawn from
    the same scoring window, so every selected id exists in scores_llm.parquet.
    """
    corpus = load_scoring_corpus(window=config.LLM_SCORE_WINDOW)
    corpus = corpus[["review_id", "text"]].copy()
    corpus["clen"] = corpus["text"].map(_collapsed_len)
    affected = (
        corpus[corpus["clen"] > old_cap]
        .sort_values("review_id")
        .reset_index(drop=True)
    )
    print(f"[select] {len(affected):,} reviews exceed {old_cap:,} collapsed chars "
          f"(of {len(corpus):,} in window {config.LLM_SCORE_WINDOW})")
    if not affected.empty:
        print(f"         collapsed length over the cap: min {affected['clen'].min():,}  "
              f"median {affected['clen'].median():.0f}  max {affected['clen'].max():,}")
    return affected


# --- Phase 1: re-score -------------------------------------------------------
def rescore(args) -> None:
    if config.LLM_MAX_REVIEW_CHARS <= args.old_cap:
        raise SystemExit(
            f"config.LLM_MAX_REVIEW_CHARS ({config.LLM_MAX_REVIEW_CHARS}) is not "
            f"above the production cap ({args.old_cap}); re-scoring would change "
            "nothing. Raise the cap in config.py first."
        )
    t0 = time.perf_counter()
    affected = select_truncated(args.old_cap)
    if affected.empty:
        raise SystemExit("No truncated reviews found — nothing to do.")

    backend = args.backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401
            backend = "vllm"
        except ImportError:
            backend = "hf"
    device = resolve_device()
    print(f"[backend] {backend}  device {device}  model {config.LLM_MODEL}  "
          f"cap now {config.LLM_MAX_REVIEW_CHARS:,} chars (full text)")

    generate = build_generator(backend, config.LLM_MODEL, device, args.server_url)
    messages = [build_messages(t) for t in affected["text"]]
    raw_outputs = generate(messages)
    scores = build_scores_frame(affected, raw_outputs)

    n_fail = int((~scores["parse_ok"]).sum())
    print(f"\n[validate] parse failures: {n_fail:,} / {len(scores):,} "
          f"({n_fail / len(scores):.1%}) — failures carry NaNs, never guesses")
    print_aspect_validation(scores)

    scores.to_parquet(RESCORE_OUT, index=False)
    print(f"\nSaved {len(scores):,} re-scored reviews -> {RESCORE_OUT}")
    print(f"Runtime: {time.perf_counter() - t0:.1f} s ({(time.perf_counter() - t0) / 60:.1f} min)")
    print("Inspect this file, then run with --merge to overwrite scores_llm.parquet.")


# --- Phase 2: merge ----------------------------------------------------------
def merge() -> None:
    if not RESCORE_OUT.exists():
        raise SystemExit(f"Re-score file missing: {RESCORE_OUT}. Run phase 1 first.")
    if not MAIN_OUT.exists():
        raise SystemExit(f"Production scores missing: {MAIN_OUT}.")

    main = pd.read_parquet(MAIN_OUT)
    rescore_df = pd.read_parquet(RESCORE_OUT)

    # Same columns both sides (both built by build_scores_frame); guards
    # against schema drift.
    if set(rescore_df.columns) != set(main.columns):
        raise SystemExit(
            f"Column mismatch:\n  main:    {list(main.columns)}\n"
            f"  rescore: {list(rescore_df.columns)}"
        )
    rescore_df = rescore_df[main.columns]

    affected_ids = set(rescore_df["review_id"])
    if not rescore_df["review_id"].is_unique:
        raise SystemExit("Duplicate review_id in the re-score file — aborting.")
    missing = affected_ids - set(main["review_id"])
    if missing:
        raise SystemExit(
            f"{len(missing):,} re-scored review_ids are absent from {MAIN_OUT.name} "
            "— refusing to merge a mismatched set."
        )

    # Back the 38-h output up once; never clobber an existing backup.
    if not BACKUP_OUT.exists():
        main.to_parquet(BACKUP_OUT, index=False)
        print(f"[backup] original -> {BACKUP_OUT}")
    else:
        print(f"[backup] keeping existing backup {BACKUP_OUT.name} (not overwritten)")

    # How many reviews changed a label once the model saw the full text?
    # (Sentinel fill makes NaN==NaN compare equal.)
    a_ids = sorted(affected_ids)
    before = main.set_index("review_id").loc[a_ids, ASPECT_COLS].fillna(-999)
    after = rescore_df.set_index("review_id").loc[a_ids, ASPECT_COLS].fillna(-999)
    changed = int((before != after).any(axis=1).sum())
    print(f"[diff] {changed:,} / {len(a_ids):,} re-scored reviews changed at least "
          "one aspect label after seeing their full text")

    # Overwrite the affected rows, preserving the original (review_id-sorted) order.
    order = main["review_id"].tolist()
    kept = main[~main["review_id"].isin(affected_ids)]
    merged = pd.concat([kept, rescore_df], ignore_index=True)
    assert len(merged) == len(main), f"row count changed: {len(merged):,} != {len(main):,}"
    merged = merged.set_index("review_id").loc[order].reset_index()
    assert (merged["review_id"].to_numpy() == main["review_id"].to_numpy()).all(), \
        "row order/identity drifted after merge"

    merged.to_parquet(MAIN_OUT, index=False)
    print(f"[done] overwrote {len(affected_ids):,} rows -> {MAIN_OUT}")
    print(f"       original preserved at {BACKUP_OUT}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--merge", action="store_true",
        help="phase 2: overwrite the re-scored rows into scores_llm.parquet "
             "(backs the original up first)",
    )
    p.add_argument(
        "--old-cap", type=int, default=PRODUCTION_CAP,
        help=f"the char cap the first full pass used (default {PRODUCTION_CAP})",
    )
    p.add_argument(
        "--backend", choices=["auto", "vllm", "hf", "server"], default="auto",
        help="auto = vLLM if importable (the cluster path), else HF transformers",
    )
    p.add_argument("--server-url", default="http://127.0.0.1:8080/v1",
                   help="base URL for --backend server")
    args = p.parse_args()

    config.ensure_dirs()
    if args.merge:
        merge()
    else:
        rescore(args)


if __name__ == "__main__":
    main()
