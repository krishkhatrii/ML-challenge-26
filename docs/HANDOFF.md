# HANDOFF — ML Challenge 2026, Business Entity Resolution

Written 2026-09-26. Deadline 27 Sep. Everything below is measured, not estimated,
unless it says "estimate".

---

# 0. THE 60-SECOND VERSION

- **Current best config = "Exp 7" = `ENABLE_SETKEY=1`. Val macro-F_0.5 = 0.866908.**
- **`output/` currently holds the OLD baseline** (leaderboard 0.8397, val 0.847028).
  It is valid and submittable, just worse.
- **The single most valuable remaining action is finishing the Exp 7 test-side run
  and overwriting `output/`.** Expected leaderboard ≈ 0.859 ± 0.008. Two of its four
  stages are already done and saved to disk.
- Everything else (Exp 8, 9, cap sweep, Exp 5, 6, docs, zip) is optional upside.

---

# 1. ENVIRONMENT

- Local M2 Pro MacBook, 16 GB RAM, 10 cores. **No SageMaker, no GPU.**
  (`CLAUDE.md` says 8 vCPU / 64 GB SageMaker — that is STALE. Ignore it.)
- Python: **always `.venv/bin/python`**, never bare `python3`.
- Working dir for every command below: `/Users/krishbharatkhatri/Documents/coding/ML_challenge26/student_resource`
- Disk: 52 GB free. Nothing below needs more than ~20 GB.

## Two environment rules that will silently ruin results if broken

1. **`PARQUET_DIR=data/parquet_full` on EVERY run.**
   The default is `data/parquet`, which is the **50k-row SAMPLE**. If unset, recall is
   measured against 172,727 sample ground-truth pairs instead of 7,638,365 and you will
   not notice. This already bit us once.
2. **`quote_style="never"` on every submission write.**
   Already correct in `src/decide.py`. Polars otherwise writes `""` for empty strings,
   which the scorer reads back as a literal 2-character ID, scoring ~0 on the leaderboard
   with nothing visibly wrong in the file.

---

# 2. WHAT THE WINNING IDEA IS (read this before changing anything)

The data is generated from **small word pools**. `Balaji Investment Private Limited`,
`Surgical Metro Physicians`, `5441 W Mount Drive, Nashville, NC`. Consequence:

> **No single token is ever rare. The *combination* of tokens is what identifies a record.**

This was established by sampling 50 ground-truth pairs the blocking stage fails to
generate and printing them in full (`logs/exp01_misses.log`). The decisive measurement:
across all 831,258 missed pairs, **not one** has a shared token with df <= 200. The mass
sits at df 5,000–100,000.

So:
- Widening a rare-token cap (Exp 2) buys +1.59pp of raw recall and **loses** score.
- Making the **whole token set** the blocking key (Exp 7) buys +4.71pp and **keeps** it.

**Corollary you must respect: CV AUC is anti-correlated with the objective here.**
Exp 2 and Exp 3 both raised CV AUC while lowering macro-F_0.5. Exp 7 *lowered* AUC
(0.998501 -> 0.998262) and raised macro-F_0.5 by +0.0199. Also, the candidate set differs
between configs, so AUC is computed on a different pair distribution each time and is not
comparable across experiments. **Record AUC, never reason from it. Judge only on
val macro-F_0.5.**

---

# 3. RESULTS SO FAR

All val macro-F_0.5 on the clean disjoint val fold (440,875 S1 entities,
`scoring.assign_fold`, seed 13), under one fixed protocol (see §4).

| exp | change | blocking recall @cap30 | val macro-F_0.5 | verdict |
|---|---|---|---|---|
| 00 | baseline re-anchored | 86.87% | 0.847028 | reference |
| 1 | diagnosis only (no change) | — | — | reordered the backlog |
| 2 | B5/B6 df<=200 -> 1,000 | 86.81% | 0.846530 | **REVERT** |
| 3 | cap 20 | 86.42% | 0.845706 | **REVERT** |
| 3 | cap 50 | 87.19% | 0.846777 | **REVERT** |
| 3 | cap 30 (unchanged) | 86.87% | 0.847028 | **KEEP** |
| **7** | **`ENABLE_SETKEY=1`** | **90.46%** | **0.866908** | **KEEP — current best** |

Exp 7 detail:

| metric | baseline | Exp 7 |
|---|---|---|
| uncapped union recall | 89.15% | **93.86%** |
| recall @30 after jaccard-idf re-rank | 86.87% | **90.46%** |
| S1 entities with ZERO candidates | 13,638 (2.2%) | **231 (0.0%)** |
| union pairs | 104.7M | 127.8M (**+22% only**) |
| val macro-F_0.5 | 0.847028 | **0.866908** |
| best tau / delta | 0.20 / 0.35 | **0.25 / 0.35** |

Note the original `NOTES.md` baseline says val 0.847264; our protocol measures 0.847028
because it fixes the train subsample at `--s1-frac 0.10` (the original used ~0.115).
**Only compare numbers produced under the same protocol.**

---

# 4. THE MEASUREMENT PROTOCOL (why experiments only take ~25 min)

Blocking is **exactly per-S1 independent** — document frequencies come from S2+S3 only,
and ranking/capping partition by S1. So restricting the S1 side to just the entities you
evaluate produces **byte-identical candidates** for them. `blocking.py --s1-ids` does this.

Verified against a full 2.2M-S1 run: uncapped recall 89.15% vs 89.12%, cap-30 84.97% vs
84.98%, every per-block recall within 0.1pp — at **92s instead of 487s**.

The eval id set is `data/eval_ids.parquet` (already built):
`val fold (440,875) ∪ train fold @0.10 (176,194) = 617,069 S1 ids (28.0% of S1)`.

Rebuild if ever needed: `.venv/bin/python tools/make_eval_ids.py 0.10`

**Full train-side experiment = one command, ~25 min:**

```bash
cd /Users/krishbharatkhatri/Documents/coding/ML_challenge26/student_resource
<ENV VARS> S1_IDS=data/eval_ids.parquet CAND_DIR=data/cand_exp \
  tools/run_exp.sh <tag> 30 5000000
```

It runs: candidates -> rerank -> features(train@0.10) -> features(val) -> train -> sweep,
logging each stage to `logs/<tag>_*.log`. The final val score is the `BEST tau=... delta=...`
line in `logs/<tag>_sweep.log`.

### ⚠️ DANGER: `run_exp.sh` destroys the current best model

It does `rm -rf data/features` and retrains into `data/model/`, **overwriting
`lgbm.txt` and `decision.json`**. Exp 7's model and features are backed up at:

- `artifacts/exp07_model_backup/` (lgbm.txt, decision.json, features.json, metrics.json)
- `artifacts/exp07_features_backup/` (44 parquet parts, 926 MB)

**Restore before doing anything that needs the Exp 7 model:**
```bash
cp artifacts/exp07_model_backup/. data/model/ -R
```

---

# 5. STEP 1 — FINISH THE EXP 7 SUBMISSION  ← DO THIS FIRST

## Status: 2 of 4 stages COMPLETE and on disk

| stage | status | output | time |
|---|---|---|---|
| 1. test candidates | ✅ **DONE** | `data/cand_test_exp07/test_candidates_cap30.parquet` | 348s |
| 2. test re-rank | ✅ **DONE** | `data/cand_test_exp07/test_reranked_cap30.parquet` | 270s |
| 3. test features | ⏳ was at chunk 7/17 when paused | `data/features_test_exp07/test_all_part*.parquet` | ~33 min total |
| 4. apply + validate | ❌ not started | `stage/exp07/*.tsv` | ~5 min |

Measured at stage 1: **1,732,544 test S1 entities, 48,959,160 candidates at cap 30
(28.3 per S1), reduction ratio 0.9999971655.**

## First: is stage 3 still running?

```bash
ps aux | grep "[f]eatures.py" | wc -l      # 0 = dead, >0 = still going
tail -3 logs/exp07_feat.log                # shows "chunk N/17"
```

It was launched from Claude Code, so **it may have been killed when that session ended.**

- If it prints `DONE ... pairs in 17 parts` → stage 3 finished, skip to 5.2.
- If a process is alive → let it finish (~20 min from chunk 7), then 5.2.
- If it is dead and incomplete → **rerun 5.1 from scratch** (it is not resumable;
  it overwrites parts, so a partial dir is not safe to reuse).

## 5.1 Rerun stage 3 only (if needed) — ~33 min

```bash
cd /Users/krishbharatkhatri/Documents/coding/ML_challenge26/student_resource
rm -rf data/features_test_exp07
PARQUET_DIR=data/parquet_full \
ENABLE_SETKEY=1 \
CAND_DIR=data/cand_test_exp07 \
FEAT_DIR=data/features_test_exp07 \
  .venv/bin/python src/features.py --split test --fold all --cap 30 \
  2>&1 | tee logs/exp07_feat.log
```
Expect 17 chunks, ~2.88M pairs each, ~49M pairs total.

## 5.2 Apply the model -> staged TSVs — ~5 min

**Check first that `data/model/decision.json` says `"tau": 0.25, "delta": 0.35,
"val_macro_f05": 0.8669...`.** If not, restore: `cp -R artifacts/exp07_model_backup/. data/model/`

```bash
PARQUET_DIR=data/parquet_full \
CAND_DIR=data/cand_test_exp07 \
  .venv/bin/python src/decide.py --apply --split test --cap 30 \
    --feat-glob 'data/features_test_exp07/test_all_part*.parquet' \
    --out stage/exp07 \
  2>&1 | tee logs/exp07_apply.log
```

## 5.3 Validate — must pass BEFORE touching output/

```bash
.venv/bin/python utils/validate_submission.py \
    --matching stage/exp07/matching_results.tsv \
    --candidate stage/exp07/candidate_pairs.tsv \
    --test-dir dataset/test
echo "exit code: $?"      # 0 = safe to submit
```

Sanity checks — `matching_results.tsv` must have **1,732,545 lines** (1,732,544 test S1
entities + 1 header). And confirm zero quote characters:
```bash
wc -l stage/exp07/matching_results.tsv
grep -c '"' stage/exp07/matching_results.tsv        # must print 0
```

## 5.4 Promote and snapshot — only after exit code 0

```bash
cp stage/exp07/matching_results.tsv stage/exp07/candidate_pairs.tsv output/
cp stage/exp07/matching_results.tsv stage/exp07/candidate_pairs.tsv best/exp07_setkey_0.8669/
git add -A
git commit -m "exp7 submission: test-side regenerated, validator passes"
```

**Then upload `output/matching_results.tsv`.** Expected ≈ 0.859 (val has tracked the
leaderboard within 0.008 historically).

---

# 6. WHAT IS BUILT vs WHAT NEEDS CODE

This is the part that matters most for planning. **Do not assume the backlog items are
runnable.**

| item | code written? | ever executed? | how to run |
|---|---|---|---|
| Exp 7 set-key blocks | ✅ yes | ✅ yes, train side | `ENABLE_SETKEY=1` |
| **Exp 8** B10 product-DF pair block | ✅ **yes, gated** | ❌ **NEVER RUN** | `ENABLE_PRODPAIR=1` |
| **Exp 9** street-number normalization | ✅ **yes, gated** | ❌ **NEVER RUN** | `ENABLE_NUMNORM=1` |
| Exp 3 cap re-sweep | ✅ yes | ✅ (on old config) | `--cap` argument |
| **Exp 5** per-country tau/delta | ❌ **NOT WRITTEN** | ❌ | needs new code in `decide.py` |
| **Exp 6** delta by shortlist size | ❌ **NOT WRITTEN** | ❌ | needs new code in `decide.py` |
| Exp 4 train on 20M pairs | ✅ (flag exists) | ❌ | `--max-pairs`, but see §8 |

Verified: `grep -c country src/decide.py` = 0, `grep -c shortlist src/decide.py` = 0.
**Exp 5 and Exp 6 are ideas, not experiments. They require writing code first.**

Exp 8 and Exp 9 are written and syntax-checked but **have never executed a single SQL
statement**. Their first run may surface a DuckDB error. Budget for that.

---

# 7. REMAINING EXPERIMENTS — EXACT COMMANDS, IN PRIORITY ORDER

Run each, then read `logs/<tag>_sweep.log` for `BEST tau=... macro-F_0.5 = ...`.
**Keep it only if that number beats 0.866908.** If it does not, revert (just stop
setting the env var) and move on.

Before each: `cd` to the repo root. After each: if it wins, back up the new
`data/model/` the same way §4 describes.

## 7.1 Exp 8 — product-DF pair block  (~30 min, biggest remaining upside)

Predicted +5.93pp uncapped recall from the Exp 1 diagnostic. **Run on top of Exp 7**, so
both switches on:

```bash
ENABLE_SETKEY=1 ENABLE_PRODPAIR=1 PROD_BUDGET=50 K_PROD=8 \
PARQUET_DIR=data/parquet_full \
S1_IDS=data/eval_ids.parquet CAND_DIR=data/cand_exp \
  tools/run_exp.sh exp08prodpair 30 5000000 \
  2>&1 | tee logs/exp08_driver.log
```

**The idea:** the current pair blocks require *each* token to have df <= 5,000. That
forbids a 20k x 20k pair whose expected intersection is only
`20000*20000/10.3M = 39 records`, while allowing 5k x 5k (expected 2.4). The budget
belongs on the **product** `df1*df2 <= 50*N`, i.e. on the expected intersection size.

**Watch for the Exp 2 failure mode.** These pairs share two *mid-frequency* tokens, so
their IDF-Jaccard is middling and the re-ranker may bury them below cap 30 — exactly what
killed Exp 2. Check `logs/exp08prodpair_cand.log` (uncapped union recall) against
`logs/exp08prodpair_rerank.log` (recall @30 after re-rank). **If uncapped recall rises a
lot but recall @30 and val F_0.5 do not, that is the ranker's fault, not the block's —
revert it and say so. Do not tune around it.**

Risk: pair volume. If `pprod_s23` explodes or DuckDB spills past ~40 GB, lower the budget
(`PROD_BUDGET=20`) or the fan-out (`K_PROD=6`) and rerun. Watch `df -h .`.

## 7.2 Exp 9 — street-number normalization  (~30 min, small but cheap)

Predicted +0.83pp. Strips leading zeros and splits ranges, so `5110`/`005110`, `7`/`07`,
`899`/`00899`, `1223`/`1223-1225` stop failing string equality.

```bash
ENABLE_SETKEY=1 ENABLE_NUMNORM=1 \
PARQUET_DIR=data/parquet_full \
S1_IDS=data/eval_ids.parquet CAND_DIR=data/cand_exp \
  tools/run_exp.sh exp09numnorm 30 5000000 \
  2>&1 | tee logs/exp09_driver.log
```

(If Exp 8 won, add `ENABLE_PRODPAIR=1 PROD_BUDGET=50 K_PROD=8` too.)

**If it wins**, the regex currently only rewrites numbers *inside blocking*. Move it into
`src/normalize.py` so the `a_numov` / `a_numfirst` / `a_numboth` features see it too —
`a_numov` is the 4th-largest feature by gain, so this is likely worth more than the
blocking effect. That means re-running `normalize.py` (179s train / 279s test) and then
the whole pipeline.

## 7.3 Cap re-sweep on the new best  (~50 min for two points)

Cap 30 was optimal at 86.87% recall. At 90.46%, with a better-ordered shortlist
(recall @10 is now 87.68%, above the *old* @30), **the optimum may have moved down**.
Test 20 first — that is the direction the evidence points.

```bash
# cap 20
ENABLE_SETKEY=1 PARQUET_DIR=data/parquet_full \
S1_IDS=data/eval_ids.parquet CAND_DIR=data/cand_exp \
  tools/run_exp.sh exp10cap20 20 5000000 2>&1 | tee logs/exp10_driver.log

# cap 50 (only if 20 beat 30)
ENABLE_SETKEY=1 PARQUET_DIR=data/parquet_full \
S1_IDS=data/eval_ids.parquet CAND_DIR=data/cand_exp \
  tools/run_exp.sh exp11cap50 50 5000000 2>&1 | tee logs/exp11_driver.log
```

## 7.4 Exp 5 / Exp 6 — REQUIRE NEW CODE (~3 min each once written)

These only touch `src/decide.py`, so once written they need **no** candidate or feature
regeneration — just restore the Exp 7 features + model and re-sweep. That makes them the
cheapest possible experiments, but someone has to write them.

```bash
cp -R artifacts/exp07_model_backup/. data/model/
mkdir -p data/features && cp artifacts/exp07_features_backup/*.parquet data/features/
PARQUET_DIR=data/parquet_full .venv/bin/python src/decide.py --sweep \
  --feat-glob 'data/features/train_val_part*.parquet'
```

- **Exp 5, per-country tau/delta.** Sweep tau/delta separately for US and India. France is
  ~15% of test and has **no training signal at all**, so pick whichever of US/India
  transfers better in a leave-one-country-out test (fit on US, score India, and vice
  versa) and use that for France. Do **not** one-hot the country — it is an open set.
- **Exp 6, delta scaled by shortlist size.** Instead of one global delta=0.35, scale it by
  how many candidates the entity has. Motivation: the sweep surface is nearly flat in tau
  and strongly shaped by delta, which means the question is *how many to accept*, not
  whether — and that should depend on the shortlist.

## 7.5 Exp 4 — train on 20M pairs (LOWEST priority, probably skip)

`--max-pairs 20000000`. But the 10% train subsample only *yields* ~4.3M pairs, so you must
first regenerate features at `--s1-frac 0.5` (~5x the feature cost, ~25 min extra), and
20M x 35 float32 = 2.8 GB before LightGBM copies it — **tight on 16 GB**.
Expected value is low: Exp 2 and Exp 3 both showed the classifier is not the constraint.
**Recommend skipping unless everything else is done.**

---

# 8. KNOWN BUGS STILL PRESENT (found, not fixed)

1. **`normalize.py` misclassifies house numbers as postcodes.**
   `00899 JEFFERSON RD` produces `postcode='00899'`. So block B1 partially joins on house
   numbers and the `a_pcmatch` / `a_pcboth` features are partly noise. **Not currently
   costing recall** — postcode equality holds for 0 of the 831k missed pairs — but it is
   wrong and it pollutes two features.
2. **Indic names are unreachable via the name field, by construction.** Transliteration is
   phonetic and one-way: `सूर्य`->`sury` vs S1 `surya`, `ईस्टर्न`->`istarn` vs `eastern`.
   Per-script recall confirms it — Devanagari sat at 66.03% in Exp 7, essentially
   unchanged from baseline 66.05%, while Latin went 86.27% -> 90.33%. **Any further Indic
   gain must come from the address, not the name.** Do not spend time on transliteration.

## Bugs already FIXED (do not reintroduce)

- `blocking.py` line ~592 had `del gt_path.fetchone()[0]`, a botched edit that made the
  per-block recall report crash with `NameError` on any split with labels. Fixed.
- The union phase scanned the 373M-row block glob **twice** per partition. Now one pass.
- `quote_style="never"`, the `\p{M}` tokenizer class, the anusvara `M`->`n` corruption, and
  the train/val fold-hash mismatch — all fixed earlier, all documented in `NOTES.md`.

---

# 9. STEP 7 — DOCUMENTATION AND ZIP (not started)

`Documentation_template.md` is still the blank template. Fill from `docs/EXPERIMENTS.md`
and `NOTES.md`. The organizers review top teams' packages, so measured numbers matter.
Must cover:

- **Methodology** — the word-pool insight from §2, and why set-key blocking follows from it.
- **Blocking strategy** — the 9 blocks, per-block recall table, uncapped union 93.86%,
  recall @cap30 90.46%, **reduction ratio 0.9999971655**, 28.3 candidates per S1.
- **Model** — LightGBM, 35 features, GroupKFold(4) by S1. Top gains: `jac_idf` ~39%,
  `c_ratio_mean` ~14%, `c_rank` ~10%, `a_numov` ~9%. Context features are ~25% of total
  gain and outrank every individual string metric — matching here is *competitive*, not
  independent.
- **Ablation table** — the §3 scoreboard, including the reverts. The negative results are
  the most informative part; say so.
- **Threshold sweep** — the tau x delta grid in `logs/exp07setkey_sweep.log`. One-to-one
  conflict resolution is worth +0.0022.
- **Bugs found** — §8 plus the fixed list. The `quote_style` trap is worth calling out:
  it silently scored a correct baseline as 0.000 instead of 0.0558.

Then build the zip:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/
│   ├── README.md          # exact end-to-end reproduction steps
│   └── requirements.txt   # pinned — copy the existing one, already pinned
└── Documentation_template.md
```

`requirements.txt` is already pinned:
`polars==1.44.2, duckdb==1.5.5, Unidecode==1.4.0, indic-transliteration==2.3.82,
rapidfuzz==3.14.6, lightgbm==4.7.0, scikit-learn==1.9.1, numpy==2.5.3, pyarrow==25.0.1`

**`dataset/` is gitignored — never include it in the zip.**

---

# 10. FILE MAP

| path | what |
|---|---|
| `src/blocking.py` | 9 blocks, re-rank. Switches at lines ~390-425. |
| `src/features.py` | 35 pairwise features, chunked by `hash(s1)`. |
| `src/train.py` | LightGBM, GroupKFold by S1. |
| `src/decide.py` | tau/delta sweep, floor rule, greedy one-to-one, submission writer. |
| `src/scoring.py` | **DO NOT MODIFY.** Validated metric + the fold split. |
| `src/normalize.py` | 8 Indic scripts -> ITRANS -> Latin, parallelized. |
| `tools/run_exp.sh` | one-command train-side experiment |
| `tools/run_test_side.sh` | one-command test-side submission regeneration |
| `tools/make_eval_ids.py` | builds `data/eval_ids.parquet` |
| `tools/exp01_misses.py` | the miss diagnostic that produced the key insight |
| `tools/exp01b_levers.py` | prices each candidate blocking rule against the misses |
| `docs/EXPERIMENTS.md` | full log, one entry per experiment |
| `NOTES.md` | earlier stages, recon findings, bug history |
| `best/exp00_baseline_0.8397/` | **never modify** |
| `best/exp07_setkey_0.8669/` | current best: src + config + logs (TSVs pending §5.4) |
| `artifacts/exp07_model_backup/` | Exp 7 lgbm.txt + decision.json |
| `artifacts/exp07_features_backup/` | Exp 7 train+val features, 926 MB |
| `logs/` | raw stdout of every run |

Git: last commit `60f0c71`, tags `exp00-0.8397` and `exp07-0.8669`.
Uncommitted: `tools/run_test_side.sh`, `.gitignore` (+`stage/`), the `logs/exp07_*` files.

---

# 11. RULES THAT KEEP THIS HONEST

- Judge every change on **val macro-F_0.5**. Record CV AUC, never argue from it.
- **One variable per experiment.** If it does not beat the current best, revert before the
  next one.
- **`output/` must always hold a validated submission.** Stage to `stage/<tag>/`, validate,
  only then copy over.
- `best/exp00_baseline_0.8397/` is never modified.
- Fixed seed 13 everywhere. Never re-implement the fold split — import it from
  `scoring.py`. (A DuckDB-vs-Polars hash mismatch once leaked 20% of val into training and
  showed up as F_0.5 0.214 against AUC 0.9985.)
- Log each experiment in `docs/EXPERIMENTS.md`: hypothesis, change, result table, verdict,
  cost.
- **No external data lookup** — no geocoding, registries, or internet augmentation. It is
  disqualifying. Offline rule-based libraries are fine.
- France is in test, not train (~15%). Country is an **open set of string labels**, used
  only as an equality feature. Never hard-code `{US, India}`.
- Every test S1 entity must appear in the output, including singletons (empty string).
