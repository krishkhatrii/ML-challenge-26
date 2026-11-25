# Business Entity Resolution — ML Challenge 2026

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

Final configuration: **inverted-index blocking (9 blocks, including the exact token-set
keys B8/B9) → IDF-weighted Jaccard re-rank at cap 30 → 35 pairwise features → LightGBM →
τ=0.25 / δ=0.35 decision layer with greedy one-to-one resolution.**
Val macro-F_0.5 = **0.866908** on a 440,875-entity held-out fold.

---

## 1. Environment

8 vCPU / 64 GB, CPU only, no GPU, no internet access at run time (no external data lookup
of any kind is performed). Python 3.11+.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Pinned versions (`requirements.txt`): polars 1.44.2, duckdb 1.5.5, Unidecode 1.4.0,
indic-transliteration 2.3.82, rapidfuzz 3.14.6, lightgbm 4.7.0, scikit-learn 1.9.1,
numpy 2.5.3, pyarrow 25.0.1. All MIT/Apache-2.0/BSD; LightGBM is the only model and is far
under the 8 B-parameter cap.

Random seed is **13** on every stage of the submitted path. The train/val fold split is
defined once in `src/scoring.py` and imported by every other stage — never re-implement it
(a DuckDB-`hash()` vs Polars-`hash(seed=13)` mismatch once leaked 20% of val into
training). Two dev-only paths use other seeds and touch nothing that ships: the S1
subsample hash in `features.py` / `tools/make_eval_ids.py` (97) and the diagnostic samplers
in `normalize.py` / `recon.py` (42).

## 2. Layout expected

Run every command from the directory that contains `src/`, with the dataset laid out as:

```
dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/test_source1.tsv    test_source2.tsv    test_source3.tsv
```

Stages communicate through Parquet directories, selected by environment variable:

| variable | default | meaning |
|---|---|---|
| `DATA_DIR` | `dataset` | raw TSV root |
| `PARQUET_DIR` | `data/parquet` | converted Parquet — **always set this to `data/parquet_full`**; the default points at a 50 k-row sample |
| `NORM_DIR` | `data/normalized` | normalized fields |
| `INDEX_DIR` | `data/index` | inverted index |
| `CAND_DIR` | `data/cand` | candidate + re-ranked pairs |
| `FEAT_DIR` | `data/features` | feature Parquet parts |
| `ENABLE_SETKEY` | unset | **must be `1`** — enables blocks B8/B9, worth +0.0199 val macro-F_0.5 |
| `WORKERS` | 8 | normalize parallelism |
| `DUCKDB_MEM` | — | DuckDB `memory_limit`; blocking fails below ~4 GB |

`ENABLE_PRODPAIR` and `ENABLE_NUMNORM` gate two coded-but-unrun experimental blocks. Leave
them unset to reproduce the submitted result.

## 3. End-to-end reproduction

### 3.1 Convert (once, both splits) — ~7 s

```bash
DATA_DIR=dataset PARQUET_DIR=data/parquet_full .venv/bin/python src/convert.py
```

### 3.2 Normalize — 179 s train, 279 s test

```bash
PARQUET_DIR=data/parquet_full WORKERS=8 .venv/bin/python src/normalize.py --apply --split train
PARQUET_DIR=data/parquet_full WORKERS=8 .venv/bin/python src/normalize.py --apply --split test
```

8 Indic scripts (Devanagari, Bengali, Gujarati, Gurmukhi, Kannada, Malayalam, Oriya, Tamil,
Telugu) → ITRANS → Latin, plus legal-suffix and street-abbreviation folding and house-number
/ PIN extraction. Offline and rule-based. `--verify` re-checks the parallel path against the
single-core one (they are byte-identical by sha256).

### 3.3 Test-side submission — one command, ~55 min

> **`ENABLE_SETKEY=1` is required.** It defaults to `0`, and without it blocks B8/B9 are
> silently dropped — the run completes, validates and scores, but reproduces the pre-Exp-7
> baseline (val 0.847) instead of 0.866908. `run_test_side.sh` does not set it internally.

```bash
ENABLE_SETKEY=1 PARQUET_DIR=data/parquet_full tools/run_test_side.sh exp07 30
```

This runs the four stages below, writes to `stage/exp07/`, and runs the validator. It never
touches `output/` — promote only after the validator exits 0.

Equivalent explicit sequence:

```bash
export PARQUET_DIR=data/parquet_full ENABLE_SETKEY=1
export CAND_DIR=data/cand_test_exp07 FEAT_DIR=data/features_test_exp07

.venv/bin/python src/blocking.py candidates --split test --cap 30   # ~350 s
.venv/bin/python src/blocking.py rerank     --split test --cap 30   # ~270 s
.venv/bin/python src/features.py --split test --fold all --cap 30   # ~1,950 s, 17 parts
.venv/bin/python src/decide.py --apply --split test --cap 30 \
    --feat-glob 'data/features_test_exp07/test_all_part*.parquet' \
    --out stage/exp07                                              # ~390 s
```

`features.py` writes one Parquet part per chunk of S1 entities (17 parts, ~2.87 M pairs
each, 48,959,160 total) and frees memory between chunks — the full pair set is never held in
RAM. It is resumable: existing parts are skipped, so a killed kernel costs one chunk.

`decide.py --apply` reads `data/model/lgbm.txt` and `data/model/decision.json`
(`tau=0.25, delta=0.35`) and writes both TSVs with `quote_style="never"`.

### 3.4 Validate, then promote — mandatory

```bash
.venv/bin/python utils/validate_submission.py \
    --matching stage/exp07/matching_results.tsv \
    --candidate stage/exp07/candidate_pairs.tsv \
    --test-dir dataset/test
echo "exit code: $?"          # 0 = safe

wc -l stage/exp07/matching_results.tsv          # must be 1732545 (1732544 + header)
grep -c '"' stage/exp07/matching_results.tsv    # must be 0

cp stage/exp07/matching_results.tsv stage/exp07/candidate_pairs.tsv output/
```

**`quote_style="never"` is not optional.** Polars' `write_csv` emits `""` for an empty
string; the format spec is unquoted, so a singleton row round-trips as a literal
two-character `""` ID and the scorer counts a wrong prediction on every entity that should
have been empty. Measured: the predict-empty baseline scored 0.000 instead of 0.0558 that
way, with nothing in the file that looks wrong.

Expected output: 1,732,544 rows, 18,066 empty (1.04%), 1,714,478 with ≥ 1 match,
5,372,239 matched IDs.

## 4. Retraining from scratch (optional — the shipped model is in `data/model/`)

```bash
export PARQUET_DIR=data/parquet_full ENABLE_SETKEY=1

.venv/bin/python src/blocking.py candidates --split train --cap 30
.venv/bin/python src/blocking.py rerank     --split train --cap 30
.venv/bin/python src/features.py --split train --fold train --cap 30 --s1-frac 0.10
.venv/bin/python src/features.py --split train --fold val   --cap 30
.venv/bin/python src/train.py  --feat-glob 'data/features/train_train_part*.parquet' \
                               --max-pairs 5000000
.venv/bin/python src/decide.py --sweep --feat-glob 'data/features/train_val_part*.parquet'
```

`train.py` uses GroupKFold(4) grouped by S1 entity. `decide.py --sweep` grids τ × δ against
**macro-F_0.5** and writes the winner to `data/model/decision.json`; it prints CV AUC but
never selects on it — AUC moved *opposite* to macro-F_0.5 in three separate experiments
here, so selecting on AUC would have rejected the single best change (see the ablation table
in `Documentation_template.md`).

Fast train-side experiment loop (`tools/run_exp.sh`) restricts the S1 side to the 617,069
entities actually evaluated, which reproduces every full-run number to within 0.1pp and cuts
the candidate stage from 487 s to 92 s:

```bash
.venv/bin/python tools/make_eval_ids.py
ENABLE_SETKEY=1 PARQUET_DIR=data/parquet_full \
  S1_IDS=data/eval_ids.parquet CAND_DIR=data/cand_exp \
  tools/run_exp.sh exp07setkey 30 5000000
```

## 5. Module map

| file | role |
|---|---|
| `src/convert.py` | 7 TSVs → Parquet. Never `pandas.read_csv` a source file. |
| `src/recon.py` | EDA: match distribution, singleton rate, one-to-one check, scripts, NULLs |
| `src/normalize.py` | Indic transliteration + field normalization, 8-way parallel |
| `src/blocking.py` | inverted index, the 9 blocks (switches ~lines 390–425), IDF-Jaccard re-rank, cap |
| `src/features.py` | the 35 pairwise features, chunked on `hash(s1)` |
| `src/train.py` | LightGBM, GroupKFold(4) by S1 |
| `src/decide.py` | τ/δ sweep, absolute-gap rule (`p >= best - δ`), top-1 floor, greedy one-to-one, submission writer |
| `src/scoring.py` | validated macro-F_0.5 + the canonical fold split — **do not modify** |
| `src/shard.py` | 5%-of-S1 dev shard |
| `utils/validate_submission.py` | pre-upload validator (stdlib only) |
| `tools/run_exp.sh` | one-command train-side experiment |
| `tools/run_test_side.sh` | one-command test-side regeneration (used for this submission) |
| `tools/make_eval_ids.py` | builds `data/eval_ids.parquet` |
| `tools/exp01_misses.py` | miss diagnostic that produced the token-set-key insight |
| `tools/exp01b_levers.py` | prices each candidate blocking rule against the misses |

## 6. Notes that matter when re-running

- `country` is an **open set of string labels** used only as an equality feature. Test
  contains France, absent from training; nothing is hard-coded or one-hot encoded to
  `{US, India}`, and every test S1 entity appears in the output, singletons included.
- No external data lookup: no geocoding, registries, ER services or network calls. The only
  external resources are the offline rule-based libraries in `requirements.txt`.
- Blocking is inverted-index based — dict/table lookups, no similarity search over dense
  embeddings, no cross-encoder, no scan of the 2.2 M × 10 M cross product.
- `PARQUET_DIR` defaults to a 50 k-row sample. Always set it explicitly; a run against the
  sample agrees to 0.03pp, which makes the mistake hard to notice.
- Blocking's union phase is partitioned on `hash(s1) % 8`; a single un-partitioned group-by
  over 405 M rows overflowed 35 GiB of spill.
