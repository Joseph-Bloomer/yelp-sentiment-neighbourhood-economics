"""Stage 4c — LLM aspect-based sentiment (the sole ABSA engine).

The real pass runs on the cluster only: a ~4B instruct model over the reviews
in config.LLM_SCORE_WINDOW — (2013, None), i.e. 2013 onward, ~6.10M US reviews
(VADER/SiEBERT cover all years). Backend is vLLM if installed (continuous
batching), else HF transformers; not Ollama, which does not suit batch jobs on
offline HPC nodes. One model instance, batched — extra instances on one GPU do
not multiply throughput. Pre-stage the weights before requesting a node
(huggingface-cli download Qwen/Qwen3-4B-Instruct-2507) and run in tmux under
jupyter-keepalive (24 h):

    tmux new -s absa
    .venv/bin/python -m pipeline.stage4c_llm

The laptop runs only the smoke test — same prompt, parser and schema, tiny
model on CPU over a handful of reviews:

    .venv\\Scripts\\python.exe -m pipeline.stage4c_llm --smoke

Closer rehearsal (--backend server): the same model as a quantised GGUF on the
laptop's AMD iGPU via llama.cpp's Vulkan llama-server (OpenAI-compatible), e.g.

    C:\\Users\\joein\\tools\\llama.cpp\\llama-server.exe -m <Qwen3-4B Q4_K_M.gguf> -ngl 99 -c 8192 --parallel 4
    .venv\\Scripts\\python.exe -m pipeline.stage4c_llm --smoke --sample 150 --backend server --tag q4

Aspect-category design: the prompt names the five fixed aspects and the model
returns mentioned-or-absent + polarity per category directly — no term
extraction, no mapping lexicon.

Writes data/processed/scores_llm.parquet
(review_id, {aspect}_mentioned + {aspect}_polarity, llm_raw, parse_ok).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import pandas as pd

import config
from pipeline.stage4_common import (
    ASPECTS,
    FULL_CORPUS_PARQUET,
    POLARITY_VALUE,
    empty_aspect_record,
    load_scoring_corpus,
    out_path,
    print_aspect_validation,
    resolve_device,
)

OUT_STEM = "scores_llm"

# Smoke-test model — quality irrelevant, it only proves prompt/parse/save.
# Qwen2.5-0.5B rather than a small Qwen3: no thinking mode to disable.
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 96

SYSTEM_PROMPT = (
    "You are an aspect-based sentiment classifier for reviews of restaurants "
    "and local businesses. You respond with a single JSON object and nothing else."
)

USER_TEMPLATE = """Classify the review's sentiment towards each of these five aspects: food, service, price, ambience, location.

For each aspect answer with exactly one of:
- "positive", "negative", or "neutral" if the review mentions the aspect (explicitly or implicitly);
- "absent" if the review does not mention the aspect at all.

"neutral" means the aspect IS mentioned but without clear sentiment. Do not confuse it with "absent".

Respond with only a JSON object in exactly this form:
{{"food": "...", "service": "...", "price": "...", "ambience": "...", "location": "..."}}

Review:
\"\"\"{review}\"\"\""""


def build_messages(text: str) -> list[dict]:
    review = " ".join(str(text).split())[:config.LLM_MAX_REVIEW_CHARS]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(review=review)},
    ]


# --- Backends ----------------------------------------------------------------
# Each factory loads its model once and returns a generate(messages) closure
# the shard loop reuses — never reload per shard.
def make_vllm_generator(model_id: str):
    """Cluster path: one vLLM instance, reused across shards.

    vLLM still batches each 50k-row shard's prompts internally, so per-shard
    .chat() calls cost nothing and let finished shards be written and resumed.
    """
    from vllm import LLM, SamplingParams

    print(f"[backend] vLLM, model {model_id} (loaded once, reused per shard)")
    llm = LLM(model=model_id)
    params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    def generate(all_messages: list[list[dict]]) -> list[str]:
        outputs = llm.chat(all_messages, params)
        return [o.outputs[0].text for o in outputs]

    return generate


def make_server_generator(base_url: str, model_id: str):
    """Any OpenAI-compatible endpoint — built for llama.cpp's llama-server
    running the same model as a quantised GGUF on the laptop iGPU. Test harness
    only; the cluster path stays vLLM/HF. A few threads keep the server's
    parallel slots busy.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import requests

    url = base_url.rstrip("/") + "/chat/completions"
    print(f"[backend] OpenAI-compatible server at {url} (model label: {model_id})")

    def one(messages: list[dict]) -> str:
        r = requests.post(
            url,
            json={
                "model": model_id,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": MAX_NEW_TOKENS,
            },
            timeout=600,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def generate(all_messages: list[list[dict]]) -> list[str]:
        outputs: list[str | None] = [None] * len(all_messages)
        t0 = time.perf_counter()
        done = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(one, m): i for i, m in enumerate(all_messages)}
            for fut in as_completed(futures):
                outputs[futures[fut]] = fut.result()
                done += 1
                if done % 10 == 0 or done == len(all_messages):
                    rate = done / (time.perf_counter() - t0)
                    print(f"    {done:>9,} / {len(all_messages):,}  ({rate:.2f} reviews/s)")
        return outputs

    return generate


def make_hf_generator(model_id: str, device: str):
    """Fallback / smoke path: HF transformers, model loaded once, manual batching."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[backend] HF transformers, model {model_id} on {device} (loaded once)")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        .to(device)
        .eval()
    )
    bs = 16 if device == "cuda" else 2

    def generate(all_messages: list[list[dict]]) -> list[str]:
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in all_messages
        ]
        outputs: list[str] = []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(texts), bs):
                enc = tokenizer(
                    texts[start : start + bs],
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",  # decoder-only: pad left so generation continues the prompt
                ).to(device)
                generated = model.generate(
                    **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                new_tokens = generated[:, enc["input_ids"].shape[1]:]
                outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
                done = start + min(bs, len(texts) - start)
                rate = done / (time.perf_counter() - t0)
                print(f"    {done:>9,} / {len(texts):,}  ({rate:.2f} reviews/s)")
        return outputs

    return generate


def build_generator(backend: str, model_id: str, device: str, server_url: str):
    """Load the chosen backend's model once and return its generate closure."""
    if backend == "vllm":
        return make_vllm_generator(model_id)
    if backend == "server":
        return make_server_generator(server_url, model_id)
    return make_hf_generator(model_id, device)


# --- Parsing ------------------------------------------------------------------
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_llm_output(raw: str) -> tuple[dict, bool]:
    """Parse one model response into the wide per-aspect record.

    Lenient on wrapping (first {...} block, case-insensitive values), strict on
    content: an invalid response gets parse_ok=False and NaNs, never guesses.
    """
    record = empty_aspect_record()
    match = _JSON_RE.search(raw)
    if not match:
        return record, False
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return record, False
    if not isinstance(payload, dict):
        return record, False

    payload = {str(k).lower(): str(v).lower() for k, v in payload.items()}
    ok = False
    for aspect in ASPECTS:
        value = payload.get(aspect)
        if value in POLARITY_VALUE:
            record[f"{aspect}_mentioned"] = True
            record[f"{aspect}_polarity"] = POLARITY_VALUE[value]
            ok = True
        elif value == "absent":
            ok = True
        # anything else (missing key, junk value) stays absent/NaN
    return record, ok


# --- Validation ---------------------------------------------------------------
def validate(corpus: pd.DataFrame, scores: pd.DataFrame, smoke: bool) -> None:
    print("[2] Validation")
    n_fail = int((~scores["parse_ok"]).sum())
    print(f"    parse failures: {n_fail:,} / {len(scores):,} "
          f"({n_fail / len(scores):.1%}) — failures carry NaNs, never guesses")

    print_aspect_validation(scores)

    print("\n    5 sample reviews (raw response -> parsed):")
    sample_idx = scores.sample(n=min(5, len(scores)), random_state=0).index
    for i in sample_idx:
        snippet = " ".join(str(corpus.loc[i, "text"]).split())[:90]
        parsed = ", ".join(
            f"{a}={scores.loc[i, f'{a}_polarity']:+.0f}"
            for a in ASPECTS if scores.loc[i, f"{a}_mentioned"]
        ) or "(all absent)"
        raw = " ".join(str(scores.loc[i, "llm_raw"]).split())[:80]
        print(f'      "{snippet}..."')
        print(f"        raw=[{raw}]  parsed: {parsed}  ok={scores.loc[i, 'parse_ok']}")
    if smoke:
        print("\n    SMOKE RUN: a 0.5B model's labels are rough — judge the "
              "plumbing (parse rate, schema), not the classifications.")


# --- Shard I/O ----------------------------------------------------------------
def build_scores_frame(rows: pd.DataFrame, raw_outputs: list[str]) -> pd.DataFrame:
    """Parse one shard's raw responses into the wide per-aspect frame.

    Same schema and column order for every shard (review_id,
    {aspect}_mentioned/_polarity x5, llm_raw, parse_ok), so the concatenation
    of all shards is exactly what Stage 5 reads.
    """
    records, ok_flags = [], []
    for raw in raw_outputs:
        record, ok = parse_llm_output(raw)
        record["llm_raw"] = raw
        records.append(record)
        ok_flags.append(ok)
    scores = pd.DataFrame(records, index=rows.index)
    scores.insert(0, "review_id", rows["review_id"])
    scores["parse_ok"] = ok_flags
    return scores


def _atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
    """Parquet to a .tmp sibling, then os.replace — a part file exists only
    once complete, so resume can trust any present part. os.replace (not
    os.rename) is atomic on Windows too, for the laptop rehearsal path.
    """
    tmp = dest.parent / (dest.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)


# --- Orchestration ---------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke", action="store_true",
        help=f"laptop plumbing check: {SMOKE_MODEL} on CPU over a few reviews, "
             "writes *_smoke.parquet",
    )
    parser.add_argument(
        "--backend", choices=["auto", "vllm", "hf", "server"], default="auto",
        help="auto = vLLM if importable, else HF transformers; server = an "
             "OpenAI-compatible endpoint (e.g. llama-server on the laptop iGPU)",
    )
    parser.add_argument("--model", default=None, help="override config.LLM_MODEL")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="override the --smoke sample size, e.g. --smoke --sample 300 "
             "(for parser checks on a bigger draw)",
    )
    parser.add_argument(
        "--server-url", default="http://127.0.0.1:8080/v1",
        help="base URL for --backend server",
    )
    parser.add_argument(
        "--tag", default=None,
        help="suffix for the output file (scores_llm_<tag>[_smoke].parquet) "
             "so comparison runs do not overwrite each other",
    )
    parser.add_argument(
        "--force-cpu", action="store_true",
        help="allow a full (non-smoke) run on CPU — normally refused",
    )
    parser.add_argument(
        "--full-corpus", action="store_true",
        help="score the nationwide corpus (reviews_corpus_full.parquet, all "
             "metros) and all years — overrides config.LLM_SCORE_WINDOW",
    )
    args = parser.parse_args()

    config.ensure_dirs()
    t0 = time.perf_counter()

    model_id = args.model or (SMOKE_MODEL if args.smoke else config.LLM_MODEL)

    backend = args.backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401
            backend = "vllm"
        except ImportError:
            backend = "hf"
    if args.smoke and backend == "vllm":
        backend = "hf"  # smoke is a CPU plumbing check; vLLM wants the GPU path

    if backend == "server":
        device = "server"  # compute happens wherever the endpoint runs
    else:
        device = resolve_device()
        # The full pass on local CPU is a multi-week mistake, not a run mode.
        if device == "cpu" and not args.smoke and not args.force_cpu:
            raise SystemExit(
                "Refusing the full LLM pass on CPU — this stage is CLUSTER ONLY.\n"
                "Use --smoke for the laptop plumbing check, or --force-cpu if "
                "you really mean it (with DEV_MODE, presumably)."
            )

    corpus_path = FULL_CORPUS_PARQUET if args.full_corpus else None
    window = config.LLM_SCORE_WINDOW  # (2013, None) = 2013 onward; see config
    if args.full_corpus:
        window = "all"  # all metros, all years — no date filter
        print("[scope] --full-corpus: scoring the nationwide corpus, all years")
    corpus = load_scoring_corpus(
        window=window, smoke=args.smoke, sample_n=args.sample, corpus_path=corpus_path
    )
    # Sorting by review_id fixes the shard boundaries: a resumed run rebuilds
    # the same slices, so part files already on disk still line up.
    corpus = (
        corpus[["review_id", "text"]]
        .sort_values("review_id")
        .reset_index(drop=True)
    )
    n = len(corpus)

    shard_size = config.LLM_SHARD_SIZE
    total_shards = max(1, math.ceil(n / shard_size))

    stem = f"{OUT_STEM}_{args.tag}" if args.tag else OUT_STEM
    final_out = out_path(stem, smoke=args.smoke)
    shards_dir = config.PROCESSED_DIR / f"{stem}_shards{'_smoke' if args.smoke else ''}"
    shards_dir.mkdir(parents=True, exist_ok=True)

    def part_path(i: int) -> Path:
        return shards_dir / f"part_{i:05d}.parquet"

    def shard_bounds(i: int) -> tuple[int, int]:
        return i * shard_size, min((i + 1) * shard_size, n)

    remaining = [i for i in range(total_shards) if not part_path(i).exists()]
    n_done = total_shards - len(remaining)
    print(f"[shards] {n:,} reviews -> {total_shards} shard(s) of up to "
          f"{shard_size:,} rows in {shards_dir}")
    if n_done:
        print(f"resuming: {n_done}/{total_shards} shards already done")

    if remaining:
        generate = build_generator(backend, model_id, device, args.server_url)
        cumulative = sum(
            shard_bounds(i)[1] - shard_bounds(i)[0]
            for i in range(total_shards) if part_path(i).exists()
        )
        for i in remaining:
            lo, hi = shard_bounds(i)
            rows = corpus.iloc[lo:hi]
            messages = [build_messages(t) for t in rows["text"]]
            raw_outputs = generate(messages)
            shard_scores = build_scores_frame(rows, raw_outputs)
            _atomic_write_parquet(shard_scores, part_path(i))
            cumulative += len(rows)
            print(f"[shard {i + 1}/{total_shards}] wrote {len(rows):,} rows -> "
                  f"{part_path(i).name}  cumulative {cumulative:,}/{n:,}  "
                  f"elapsed {time.perf_counter() - t0:.1f}s")
            del messages, raw_outputs, shard_scores
    else:
        print("[shards] all shards already present — going straight to finalise")

    # --- Finalise: concatenate every shard into the file Stage 5 reads --------
    missing = [i for i in range(total_shards) if not part_path(i).exists()]
    if missing:
        raise SystemExit(
            f"Aborting before finalise: {len(missing)} shard(s) still missing "
            f"(first few: {missing[:5]}). Re-run to resume the pass."
        )

    scores = pd.concat(
        (pd.read_parquet(part_path(i)) for i in range(total_shards)),
        ignore_index=True,
    )
    assert len(scores) == n, f"shard rows {len(scores):,} != corpus {n:,}"
    # Shards were built from the sorted corpus and read back in order, so row
    # positions match `corpus`; validate() relies on that.
    if not (scores["review_id"].to_numpy() == corpus["review_id"].to_numpy()).all():
        raise SystemExit(
            "Shard review_id order does not match the corpus — refusing to write "
            "a misaligned final file. Delete the shards dir and re-run."
        )

    validate(corpus, scores, smoke=args.smoke)

    scores.to_parquet(final_out, index=False)

    runtime = time.perf_counter() - t0
    print(f"\nSaved {len(scores):,} scored reviews -> {final_out}")
    print(f"Shards: {total_shards} part file(s) kept in {shards_dir}")
    print(f"Backend: {backend}  Model: {model_id}  Device: {device}")
    print(f"Runtime: {runtime:.1f} s ({runtime / 60:.1f} min)")


if __name__ == "__main__":
    main()
