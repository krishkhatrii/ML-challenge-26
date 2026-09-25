# ML Challenge 2026 — Business Entity Resolution

## Task

Given business records from 3 independent noisy sources, find which Source 2 / Source 3
records refer to the same real-world business as each Source 1 record. Source 1 is the
deduplicated reference. Each S1 entity matches zero, one, or many S2/S3 records.

Fields: `entity_id`, `business_name`, `business_address`, `country`. Tab-separated.
Source is implied by the entity_id prefix (`S1-`, `S2-`, `S3-`).

**Pipeline design and per-stage detail: see `docs/PLAN.md`. Read it when starting a new
stage.**

## Scale and hardware — read before writing any code

| File | Rows |
|---|---|
| train_source1 | 2.21 M |
| train_source2 | 5.03 M |
| train_source3 | 5.29 M |
| train_ground_truth | 2.21 M |
| test_source1 | 1.73 M |
| test_source2 | 4.89 M |
| test_source3 | 5.08 M |

~4.6 GB of TSV. Running on SageMaker `ml.r7i.2xlarge` — **8 vCPU, 64 GB RAM, no GPU.**

Non-negotiable consequences:

- **Never call `pandas.read_csv` on a source file.** Use `polars` or `duckdb`.
  `pandas` only for small result frames.
- All 7 TSVs convert to Parquet once, up front. Every stage reads and writes Parquet.
- No dense embedding of 10M records. No cross-encoder. No pairwise similarity over the
  full cross product (2.2M x 10M = 22 trillion pairs).
- Blocking is **inverted-index based** — dict/table lookups, not similarity search.
- Feature generation is **chunked**: ~100k S1 records at a time, write to Parquet, free
  memory, continue. Never hold the full pair set in RAM.
- Checkpoint every stage to `s3://<bucket>/artifacts/<stage>/`. The kernel will die.

## Metric

`F_0.5 = (1.25 * P * R) / (0.25 * P + R)`, computed **per Source 1 entity**, then
macro-averaged across all S1 entities.

- Precision weighted 2x recall. A false merge costs more than a missed match.
- **Singletons are scored.** Correct empty prediction = 1.0; any false match on a
  singleton = 0.0.
- Tune every threshold directly against macro-F_0.5. Never against AUC or F1.
- When in doubt, predict nothing.

## Hard constraints

- Test contains **France**, absent from training. Do not hard-code, filter, or one-hot
  to `{US, India}`. Country is an open set of string labels, used only as an equality
  feature. Every test S1 entity must appear in the output.
- **No external data lookup** — no geocoding APIs, registries, ER services, or internet
  augmentation. Disqualifying. Offline rule-based libraries (`unidecode`,
  `indic-transliteration`) are fine.
- Final model must be MIT or Apache-2.0 licensed, <= 8B parameters.

## Output format

Both tab-separated, in `output/`:

- `matching_results.tsv` — `source1_entity_id`, `matched_entity_ids`. Exactly one row
  per test S1 entity. Empty string for singletons. Comma-separated, no quoting, no
  duplicates within a list, S2-/S3- IDs only.
- `candidate_pairs.tsv` — `source1_entity_id`, `candidate_entity_ids`. The **final**
  pre-inference candidate set: exactly what the model scored. Matches must be a subset
  of candidates.

### Writing submissions — hard rule

**Every submission write must pass `quote_style="never"`.**

```python
df.write_csv(path, separator="\t", quote_style="never")   # polars
```

Polars' `write_csv` emits `""` for an empty string by default. The format spec is
unquoted, so a singleton row round-trips as a literal two-character `""` ID: the
scorer sees a wrong prediction on every entity that should have been empty, and the
leaderboard reports a near-zero score with nothing in the file that looks wrong.
Measured: the predict-empty baseline scored 0.000 instead of 0.0558 this way.

`src/scoring.py` strips wrapping quotes defensively and warns, but the leaderboard
does not. Applies to `matching_results.tsv` and `candidate_pairs.tsv` alike.

Validate before every leaderboard upload:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Working conventions

- Fix the random seed. Pin versions in `requirements.txt` as you install.
- Keep a running `NOTES.md`: what was tried, what it scored, what changed. The
  methodology write-up is a graded deliverable — do not leave it to the end.
- Commit after each working stage. Tag every leaderboard upload:
  `git tag -a lb-0.72 -m "inverted-index blocking + lgbm, thr=0.55"`.
- `dataset/` is gitignored. Never commit it.
- When a stage is built, update its section in `docs/PLAN.md` to describe what was
  actually done. A stale plan is worse than no plan.

## Deliverable

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/
│   ├── README.md          # exact end-to-end reproduction steps
│   └── requirements.txt   # pinned
└── Documentation_template.md
```
