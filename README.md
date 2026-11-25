# Business Entity Resolution (ML Challenge 2026)

Our team's entry (K-Fold Duo) for ML Challenge 2026. The task: given three large, messy
business datasets, find every record in sources 2 and 3 that refers to the same real
business as each record in source 1.

The data is ~10M records per split across the US, India and (in the test set only)
France. Names and addresses come with typos, missing fields, abbreviations, house numbers
written different ways, and a share of records in Indian scripts such as Devanagari,
Tamil and Bengali. Everything ran offline on a MacBook (M2 Pro, 16 GB RAM), CPU only,
with no external lookups.

**Result:** 0.867 macro-F0.5 on our held-out validation fold, up from a 0.847 baseline
(which scored 0.840 on the public leaderboard).

## How it works

Comparing every record with every other record (2.2M × 10M) is impossible, so the
pipeline works in stages:

1. **Normalize:** transliterate Indic scripts to Latin, fold legal suffixes and street
   abbreviations, and pull out house numbers and postcodes.
2. **Block:** an inverted index plus 9 blocking rules generates a shortlist of likely
   matches for each source 1 record, keeping 90% of true matches while cutting the
   comparison space by over 99.999%.
3. **Re-rank and cap:** an IDF-weighted Jaccard score keeps the top 30 candidates per
   record.
4. **Score:** a LightGBM classifier over 35 pairwise features (string similarity, address
   numbers, postcode match, and how each candidate compares with the other candidates
   for the same record).
5. **Decide:** thresholds tuned directly on macro-F0.5, then a one-to-one pass so no
   record is matched to two different businesses.

Built with Polars and DuckDB over Parquet. The full test run takes about 55 minutes.

## What we learned

- **Rare-token blocking didn't fit this data.** We looked at the matches our blocking
  was missing and found that almost none of them shared a rare word. The names and
  addresses were built from small word pools, so individual words were all common, but
  the *combination* was unique. Hashing each record's full set of words into one
  blocking key was the biggest single improvement (+0.020 F0.5, recall from 86.9% to
  90.5%).
- **AUC pointed the wrong way.** Twice, a change raised the classifier's AUC while
  lowering the actual competition metric, and our best change lowered AUC. If we had
  picked models on AUC we would have thrown away our best idea, so every decision was
  made on macro-F0.5.
- **More candidates isn't always better.** Widening the blocking rules found more true
  matches, but the extra pairs were weak and got pushed below the cap anyway. The
  bottleneck had moved from finding candidates to ranking them.
- **A small output bug can cost everything.** Polars writes empty strings as `""`, which
  the scorer read as a wrong ID. A baseline that should have scored 0.056 scored 0.000,
  and nothing in the file looked wrong.

## What we'd fix next

About 5.6% of source 1 records have no true match, and the right answer for them is to
predict nothing. We found after submitting that our decision layer almost never does:
a fallback rule always keeps the best candidate. Fixing this, either by gating that
fallback or by training a separate "does this record have any match?" model, is worth
up to ~0.045 F0.5, more than all of our other improvements combined.

## Repo layout

```
src/          pipeline: convert, normalize, blocking, features, train, decide, scoring
tools/        experiment runners and the miss diagnostics
utils/        submission validator
artifacts/    metrics per experiment and the final LightGBM model
docs/         experiment log, working notes, plan, and full reproduction steps
logs/         raw run logs
Documentation.md   the write-up we submitted, with full numbers and ablations
```

## Running it

The competition dataset is not included in this repo. With the data in `dataset/`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then follow [docs/REPRODUCE.md](docs/REPRODUCE.md) for the full pipeline.

## Team

Krish Khatri and Karishma Doshi (K-Fold Duo)
