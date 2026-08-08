"""Pipeline package — one module per stage, run in order.

Spine:
  1. geography  — spatial join businesses to census tracts -> businesses_with_tract_2018.parquet
  2. corpus     — assemble reviews_corpus.parquet (chunked read; keep `date`)
  3. acs        — pull + join ACS indicators -> acs_tracts.parquet

Method layer:
  4. sentiment  — VADER, SiEBERT, and a local LLM (ABSA) per-review scores
  5. aggregate  — tract-level features per method (absent vs neutral; salience)
  6. evaluate   — evaluation ladder + ridge/lasso modelling (spatially-blocked CV)
"""
