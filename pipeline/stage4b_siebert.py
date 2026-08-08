"""Stage 4b — SiEBERT document-level transformer sentiment, one P(positive) per review.

Runs on the cluster/GPU: RoBERTa-large over ~600k reviews is days on a laptop
CPU, hours on one GPU. The laptop runs only --smoke / DEV_MODE — same code
path, tiny row count. SiEBERT is binary; I keep P(positive) in [0, 1] rather
than the hard label so Stage 5 can average a graded signal. Inputs are
truncated to the model's 512-token limit. Aspect-based sentiment is solely the
LLM arm (4c).

Output: data/processed/scores_siebert.parquet  (review_id, siebert_pos_prob)

Run:        .venv\\Scripts\\python.exe -m pipeline.stage4b_siebert
Smoke test: add --smoke (12 reviews, CPU-safe, writes *_smoke.parquet)
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

import config
from pipeline.stage4_common import (
    FULL_CORPUS_PARQUET,
    load_scoring_corpus,
    out_path,
    resolve_device,
)

OUT_STEM = "scores_siebert"
FIG_PATH = config.FIGURES_DIR / "stage4b_siebert_validation.png"

SIEBERT_MODEL = "siebert/sentiment-roberta-large-english"
SIEBERT_MAX_TOKENS = 512  # RoBERTa hard limit; longer reviews are truncated


# --- 1. Score -----------------------------------------------------------------
def score_siebert(texts: list[str], device: str) -> list[float]:
    """P(positive) per review from SiEBERT, batched and truncated."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"[1] SiEBERT ({SIEBERT_MODEL}) on {device}")
    tokenizer = AutoTokenizer.from_pretrained(SIEBERT_MODEL)
    # bf16 on the A40 engages tensor cores; fp32 on CPU (the smoke path).
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = (
        AutoModelForSequenceClassification.from_pretrained(SIEBERT_MODEL, torch_dtype=dtype)
        .to(device)
        .eval()
    )
    pos_idx = int(model.config.label2id.get("POSITIVE", 1))

    bs = 128 if device == "cuda" else 8
    probs: list[float] = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(texts), bs):
            batch = texts[start : start + bs]
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=SIEBERT_MAX_TOKENS,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs.extend(logits.softmax(dim=-1)[:, pos_idx].tolist())
            done = start + len(batch)
            if done % (bs * 50) < bs or done == len(texts):
                rate = done / (time.perf_counter() - t0)
                print(f"    {done:>9,} / {len(texts):,}  ({rate:,.1f} reviews/s)")
    return probs


# --- 2. Validate ----------------------------------------------------------------
def validate(corpus: pd.DataFrame, pos_prob: pd.Series, smoke: bool) -> None:
    """Distribution, stars correlation, and (full runs) a diagnostic figure."""
    print("[2] Validation")
    print("    P(positive) distribution:")
    print(pos_prob.describe().to_string().replace("\n", "\n      "))

    pearson = pos_prob.corr(corpus["stars"])
    spearman = pos_prob.corr(corpus["stars"], method="spearman")
    print(f"\n    SiEBERT vs stars: Pearson r = {pearson:.3f}, Spearman rho = {spearman:.3f}")
    print("    (high correlation feeds the 'text adds little over stars' risk;")
    print("     low correlation questions whether SiEBERT captures the rating signal)")

    print("\n    mean P(positive) by star rating:")
    by_stars = pos_prob.groupby(corpus["stars"]).agg(["mean", "count"])
    print(by_stars.to_string().replace("\n", "\n      "))

    print("\n    3 sample reviews:")
    sample_idx = corpus.sample(n=min(3, len(corpus)), random_state=0).index
    for i in sample_idx:
        snippet = " ".join(str(corpus.loc[i, "text"]).split())[:90]
        print(f'      P(pos)={pos_prob.loc[i]:.3f}  stars={corpus.loc[i, "stars"]:.0f}  "{snippet}..."')

    if smoke:
        print("    (smoke run: figures skipped)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(pos_prob, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("SiEBERT P(positive) — distribution")
    axes[0].set_xlabel("P(positive)")
    axes[0].set_ylabel("reviews")
    axes[1].bar(by_stars.index, by_stars["mean"], width=0.6, color="steelblue")
    axes[1].set_title("Mean P(positive) by star rating")
    axes[1].set_xlabel("stars")
    axes[1].set_ylabel("mean P(positive)")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"    figure -> {FIG_PATH}")


# --- Orchestration ---------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke", action="store_true",
        help="plumbing check: a handful of reviews on CPU, writes *_smoke.parquet",
    )
    parser.add_argument(
        "--full-corpus", action="store_true",
        help="score the nationwide corpus (reviews_corpus_full.parquet, all "
             "metros) instead of the Philadelphia one",
    )
    args = parser.parse_args()

    config.ensure_dirs()
    t0 = time.perf_counter()
    device = resolve_device()
    if device == "cpu" and not (args.smoke or config.DEV_MODE):
        print("WARNING: full run on CPU — RoBERTa-large over the whole corpus "
              "will take days. This stage is meant for the cluster/GPU; "
              "use --smoke or DEV_MODE locally.")

    corpus_path = FULL_CORPUS_PARQUET if args.full_corpus else None
    window = "all" if config.SCORE_ALL_REVIEWS else "union"
    corpus = load_scoring_corpus(window=window, smoke=args.smoke, corpus_path=corpus_path)

    probs = score_siebert(corpus["text"].tolist(), device)
    pos_prob = pd.Series(probs, index=corpus.index, name="siebert_pos_prob")

    validate(corpus, pos_prob, smoke=args.smoke)

    out = out_path(OUT_STEM, smoke=args.smoke)
    pd.DataFrame(
        {"review_id": corpus["review_id"], "siebert_pos_prob": pos_prob}
    ).to_parquet(out, index=False)

    runtime = time.perf_counter() - t0
    print(f"\nSaved {len(pos_prob):,} scores -> {out}")
    print(f"Device : {device}")
    print(f"Runtime: {runtime:.1f} s ({runtime / 60:.1f} min)")


if __name__ == "__main__":
    main()
