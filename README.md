# Yelp sentiment → neighbourhood economics (MSc dissertation code)

I compare three NLP sentiment methods — VADER (lexicon), SiEBERT (transformer)
and LLM aspect-based scoring — by aggregating each method's per-review scores to
US census tracts and testing how well each aggregate predicts tract-level
economic conditions from the ACS. The study covers ~11 US metros from the Yelp
Open Dataset (~6.10M reviews, 2013–2022). The economic outcome is the evaluation
target, not an economic-prediction claim.

## Environments

Two environments ran this work, so there are two requirements files. My laptop
(Python 3.13, Windows, CPU) ran stages 1–3, 4a, 5–6 and every script and
notebook; `requirements-laptop.txt` is an exact freeze of it. The full SiEBERT
and LLM scoring passes (stages 4b/4c) ran on a Linux A40 GPU cluster;
`requirements.txt` pins that scoring stack (transformers 4.51.3,
torch 2.6.0+cu124, vllm 0.8.5 — Linux-only) and pins everything else to the
laptop versions. One merged file would misstate what actually ran where.

Laptop setup:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-laptop.txt
Copy-Item .env.example .env   # then add a free Census API key (link in the file)
```

## Data

No data ships in this repo. The raw Yelp Open Dataset must be downloaded from
Yelp directly — its licence does not let me redistribute it — and unpacked into
`data/raw/`. My processed artefacts are in a HuggingFace dataset,
[Guacamole03/dissertation-corpus](https://huggingface.co/datasets/Guacamole03/dissertation-corpus).
That repo is **private**, because the cleaned corpus contains Yelp review text;
I can grant access on request. It holds the corpus, the SiEBERT and LLM score
files, and the two tables the evaluation actually runs on
(`tract_features_2018.parquet`, `acs_tracts_2018.parquet`). VADER scores are not
archived: stage 4a rebuilds them on a laptop in under an hour.

## Running the pipeline

Six stages, run in order. Stage 4 is three scoring scripts.

```bash
python -m pipeline.stage1_geography    # 1. tract spatial join (doubles as the US filter)
python -m pipeline.stage2_corpus      # 2. review corpus (cleaning filters: stage2_cleaning.py)
python -m pipeline.stage3_acs         # 3. ACS outcomes (needs the Census key)
python -m pipeline.stage4a_vader      # 4a. laptop, CPU
python -m pipeline.stage4b_siebert    # 4b. cluster GPU
python -m pipeline.stage4c_llm        # 4c. cluster GPU
python -m pipeline.stage5_aggregate   # 5. tract-level features
python -m pipeline.stage6_evaluation  # 6. spatially-blocked evaluation ladder
```

`pipeline/stage4c_rescore_truncated.py` is a one-off remediation: it re-scored
the 11,094 reviews the first 4c pass truncated at 4,000 characters. Its output
is already merged into the archived `scores_llm.parquet`.

Each scoring stage also takes `--smoke`: the identical code path on ~12 reviews,
CPU-safe, for a quick local check before committing to a full run.

## Secondary analyses and paper statistics

`scripts/fold_sign_consistency.py` runs the per-fold sign tests reported
alongside the main results; `scripts/minrev_sensitivity_summary.py` runs the
min-reviews sensitivity sweep; `scripts/data_section_stats.py` produces the
Data-section statistics. The notebooks re-derive the headline results and run
the secondary analyses (`stage6_secondary_analyses.ipynb`) and the descriptive
statistics (`data_descriptives.ipynb`).
