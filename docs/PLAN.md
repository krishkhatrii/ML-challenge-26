# Pipeline plan

Stage-by-stage design. Read the section for the stage you are building.
When a stage is done, replace its section with what was actually implemented.

Constraints live in `CLAUDE.md` and always apply.

---

## Build order

1. `src/convert.py` — TSV -> Parquet, all 7 files. **DONE.**
2. `src/recon.py` — data understanding. **DONE** — findings below and in `NOTES.md`.
3. `src/scoring.py` — macro-F_0.5 + S1-entity-grouped train/val split. **DONE.**
4. `src/shard.py` — the 5% dev shard. **DONE.**
5. `src/normalize.py`
6. `src/blocking.py` + recall measurement. Gate: >= 97% recall before proceeding.
7. `src/features.py`
8. `src/train.py`
9. `src/decide.py`

Get a crude end-to-end submission onto the leaderboard before improving anything.

**STATUS: end-to-end submission built and validated. Val macro-F_0.5 = 0.8473.**
`output/matching_results.tsv` + `output/candidate_pairs.tsv` both PASS
`utils/validate_submission.py --check-ids`.

Biggest remaining lever, in order:
1. **Blocking recall (86.85% @30)** is the binding constraint — the classifier is at
   0.99853 AUC. Widening the single-token blocks to df<=1,000 is measured to be worth
   roughly +8-12pp of union recall for ~1e9 extra pairs.
2. **Train on all folds, not a 0.12 sample** of the train fold (5M pairs was the brief).
3. **Per-country / per-script thresholds** — India trails US by ~5pp on recall, and
   France is unseen in training entirely.
4. **Calibrate delta per shortlist size** rather than one global value.

---

## Stage 1 — Convert (`src/convert.py`) — BUILT

Polars `scan_csv` -> `sink_parquet`, streaming, all 7 TSVs to `data/parquet/`.
Explicit all-`String` schema (entity IDs look numeric after the prefix; PIN/ZIP codes
carry leading zeros). `quote_char=None` — the files are unquoted, so a stray `"` is data.

Idempotent via a sidecar `_manifest.json` (source path + bytes + mtime-ns), **not**
mtime alone: the real `dataset/` TSVs predate any freshly built sample Parquet, so a
plain mtime check silently skips every file when switching `DATA_DIR`.

`DATA_DIR` (default `dataset`), `PARQUET_DIR` (default `data/parquet`), `FORCE=1`.

Measured: full 2.3 GB -> 975 MB Parquet in **6.9 s** locally. Fast enough that the
"convert once on S3" framing is moot; just re-run it.

---

## Stage 2 — Recon (`src/recon.py`) — BUILT

DuckDB over the Parquet; 8 sections printed and appended to `NOTES.md`. Numbers below
are from the **full dataset**, reconfirmed locally (the 50k `dataset_sample/` agrees to
within 0.1pp on every rate).

### Findings that change the plan

| Finding | Number | Consequence |
|---|---|---|
| **One-to-one holds** | 0 violations in 7,638,365 pairs | Stage 9 greedy conflict resolution is live |
| Singleton rate | **5.58%** (123,247 / 2,206,821) | Stage 9 must **not** favour empty predictions |
| Matches per S1 | mean **3.46**, median 3, **47.9% have 4+**, max 11 | Multi-match is the norm, not the exception |
| GT self-consistency | every matched ID exists in S2/S3 | No dangling references to defend against |
| Non-Latin names | **8 Indic scripts**, not just Devanagari | Stage 5 needs per-script routing |
| S1 script | **100% Latin**, train and test | All transliteration is one-directional |
| France | ~15% of test rows, absent from train | Stage 5 needs French suffixes and street words |
| NULL `business_address` | ~3% of S2/S3, never empty-string | Stage 7 missing-component flags |

Script counts (`business_name`, train+test S2/S3): Latin ~89-95%, Devanagari 3.0-6.4%,
and the remainder spread across **Telugu, Kannada, Tamil, Bengali, Gujarati,
Malayalam, Oriya, Gurmukhi** — the per-block counts sum exactly to the "Other" total,
so there is no unexplained residue. `business_address` is Indic for ~9% of S2/S3 rows,
so **address is not a script-invariant field either**.

---

## Stage 3 — Scoring (`src/scoring.py`) — BUILT

Macro-F_0.5 per the spec, singletons included, plus the S1-grouped train/val split.

**Closed form.** Substituting `P = tp/|pred|` and `R = tp/|truth|` into the specified
formula cancels the products:

    F_0.5 = 1.25 * tp / (0.25 * |truth| + |pred|)

Algebraically identical, but with no division by zero when `tp > 0` and no per-entity
Python loop, so it vectorises over 2.2M rows. The only case it cannot express is
`|truth| = |pred| = 0`, scored 1.0 by rule. `--selftest` asserts the closed form and
the literal precision/recall form agree to 2.2e-16 over 4,000 random cases.

**Validated three ways on the full training data** — each an independent check that the
metric is wired to reality, not just internally consistent:

| Baseline | macro-F_0.5 | Why it proves something |
|---|---|---|
| README worked example | **0.7143** | Matches the brief's 0.714 |
| predict-empty everywhere | **0.055848** | Exactly 123,247/2,206,821 — the recon singleton rate |
| perfect oracle | **1.000000** | |
| drop one true match per entity | 0.815198 | Precision intact, recall degraded |

**Split.** Hashed from the S1 entity ID, not shuffled: stable under row reordering and
data growth, reproducible across runs and machines, and recomputable by any later stage
without persisting a split file. One ground-truth row *is* one S1 entity, so hashing the
S1 ID gives the required grouped split by construction. Measured on full data:
1,765,946 train / 440,875 val (80.02 / 19.98), singleton rate 5.59% vs 5.55% across
folds, zero entities in both. Depends on Polars' seeded hash, so Polars is pinned.

**Trap found, and guarded.** Polars' `write_csv` emits `""` for an empty string. Read
back with the spec-mandated `quote_char=None`, every singleton row becomes a literal
two-character `""` ID — the predict-empty baseline scored 0.000 instead of 0.0558
before this was caught. **Stage 9 must write submissions with `quote_style="never"`.**
`read_id_file` strips wrapping quotes defensively and warns, because on the leaderboard
this failure is silent and costs a submission.

---

## Stage 4 — Dev shard (`src/shard.py`) — BUILT

Two modes. The originally planned approach was implemented, measured, and demoted.

### Why key-space sharding fails here

The plan assumed a blocking key whose value agrees between an S1 record and its true
match. Measured on 1,000,000 true pairs from the full training data:

| Candidate key | Coverage | Agreement (both present) | Agreement (overall) |
|---|---|---|---|
| leading trigram of name | 93.3% | 82.8% | **77.3%** |
| first name token | 93.3% | 77.8% | 72.6% |
| any shared address digit token | 87.2% | 91.6% | 79.9% |
| first address digit-run | 87.2% | 79.0% | 68.9% |
| address PIN/ZIP (5-6 digit) | **5.5%** | 91.1% | 5.0% |
| country | 100% | 100% | 100% (only 3 values — unusable) |

No key clears ~80%. The cause is structural: S1 is 100% Latin while 5-10% of S2/S3 is in
one of 8 Indic scripts, and `business_address` is Indic for ~9% of S2/S3 rows too — so
**no script-invariant raw field exists** before stage 5 normalization lands.

Running `--mode keyspace` on the full data confirms the consequence:
**81.73% ground-truth self-consistency** (55,260 of 302,470 matched IDs absent from the
shard), capping measurable blocking recall at 81.7% and making the >= 97% gate
unevaluable. It also skews the shard by country — India 3.21% kept vs US 4.45% —
under-representing exactly the non-Latin records that are hardest. `--repair` restores
100% consistency by adding the missing records back, at the cost of inflating match
density above the full run's.

### `--mode s1` (default)

Hash 5% of **S1 entity IDs**; keep the S2/S3 corpora **whole**. Measured on full data:

- 110,227 S1 entities (4.99%), ground truth 110,227 rows (4.99%)
- **rival retention ratio 1.0000** — mean 17,179.5 S2/S3 records share a kept entity's
  blocking key, identical in shard and full run
- **ground-truth self-consistency 100.00%** (0 of 382,090 matched IDs missing)
- country skew none: US 4.98%, India 5.02%

Cost falls where it matters: feature generation and inference scale with
(S1 count x candidates per S1), so 5% of S1 is ~5% of the pair work. Only the blocking
index build still pays full price, and that is the cheap half. S2/S3 Parquet is
hardlinked rather than copied.

The right density metric is **rivals per S1 entity** (how long a blocking posting list
is), not the S2/S3-to-S1 corpus ratio — corpus ratio rises to 19.7x in `s1` mode while
the competition each entity actually faces is unchanged.

### Notes

- Output filenames match `src/convert.py`, so later stages just repoint `PARQUET_DIR`.
- Train only; the shard exists to iterate on train/val.
- Exits non-zero when the shard is not self-consistent.
- **`dataset_sample/` cannot validate a shard.** 97.74% of its ground-truth rows
  reference an S1 entity absent from its `train_source1` — the sample takes the first
  50k rows of each file *independently*, so the files describe near-disjoint entity
  sets. `shard.py` detects this and warns rather than emitting a quietly wrong shard.
  Shard work must be validated against the full dataset.

---

## Stage 5 — Normalization (`src/normalize.py`) — BUILT

### Verdict: a FEATURE-QUALITY improvement, not a blocking prerequisite

This was an open question and it is now settled by measurement. **Transliteration does
not make Indic records blockable.** For the 551,240 ground-truth pairs whose S2/S3 side
is Indic-script, the DF of the *S1 side's* name tokens in the S2+S3 index is:

| match script | GT pairs | p10 | p50 | p90 | S1 has token df<=1k | df<=100 |
|---|---|---|---|---|---|---|
| Latin | 7,087,122 | 34 | 818 | 21,882 | 51.4% | 21.4% |
| Devanagari | 312,725 | 8,928 | 11,730 | 34,082 | **0.0%** | **0.0%** |
| Other Indic | 238,515 | 8,928 | 11,730 | 33,781 | **0.0%** | **0.0%** |

Not one S1 record with an Indic match has a name token rarer than df=1,000. Their names
are generic templates — "Balaji Investment Private Limited", "White Consultants Private
Limited", "Shyam Technology Private Limited" — so a perfect transliteration recovers
exactly those common words. There is no rare token to key on in *either* script.

**The address is the route for these records**, and it works: 82.2% of Indic-matched
pairs have an S1 address token with df <= 1,000 (51.7% below df=100), slightly *better*
than the 78.2% for Latin-matched pairs. Address normalization is therefore on the
critical path and gets equal care.

### Measured payoff (token Jaccard, S1 name vs its true match, 30k sampled pairs)

| match script | raw mean | normalized | raw share at 0 | normalized share at 0 |
|---|---|---|---|---|
| Latin | 0.666 | 0.714 | 8.3% | 8.1% |
| Devanagari | 0.036 | **0.198** | 92.8% | **18.4%** |
| Other Indic | 0.042 | **0.167** | 91.7% | **33.1%** |

Normalization takes the Indic name field from *useless* — 93% of true pairs sharing no
token at all, nothing for stage 7's name features to score — to usable. That is the
whole justification for this stage, and it is a stage 7 justification, not a stage 6 one.

### What it does

- `trim_junk` first, by Unicode **category** not an ASCII class: `-- Holloway Peak`,
  `<< Team Ecole`. A naive `\W` trim eats the final matra off an Indic word and the
  accent off `Café`; both bugs were caught by the assertions.
- Lowercase, NFKC, `unidecode`, `&` -> `and`.
- **Indic -> Latin only**, one direction, since S1 is 100% Latin. Script detected by
  Unicode block; routed through `indic-transliteration` `sanscript` ITRANS, then
  `unidecode`. Offline and rule-based, so fair-play rules hold.
  - Anusvara: ITRANS writes `M`, which unidecode renders `m` ("marketimga"). Mapped to
    `n` first, giving "marketing" — the spelling the S1 side actually uses.
  - Final schwa: ITRANS renders the inherent vowel ("limiteda", "rama"). A single
    trailing schwa after a consonant is dropped, recovering "ltd"/"ram".
  - Net: `राम मार्केटिंग प्राइवेट लिमिटेड` -> `ram marketing praivet ltd` against S1's
    `ram marketing pvt ltd` — 3 of 4 tokens exact.
- Legal suffixes canonicalised (corporation->corp, limited->ltd, private->pvt) including
  French `sarl/sas/sasu/eurl/sci/snc/sa`. Emits both `norm` and suffix-stripped `core`.
- Address abbreviations expanded (rd->road, st->street, nr->near, opp->opposite) with
  French street words preserved (rue, avenue, boulevard, place, impasse, chemin, allee).
- Structured extraction: house/street numbers (`A/101` -> `101`, `6/1` -> `6/1`) and
  PIN/ZIP (`\d{6}` or `\d{5}(-\d{4})?`).
- Transliteration cached per token at 2M entries. Names need almost none (171 distinct
  Devanagari tokens corpus-wide) but Indic **addresses** carry ~1.2M distinct tokens; at
  200k the cache thrashed (1.24M misses / 11.1M hits).

### Cost, and the one thing to fix

Full train (12.53M rows): **9m23s**, ~22k rows/s, writing 1.50 GB of normalized Parquet.
It ran at **107% CPU — single-core bound**, because the transliteration path is pure
Python under the GIL. Convert is 7s and the index build is 10s, so this stage is now the
pipeline's slowest step by two orders of magnitude. **Parallelise it with
`multiprocessing` (8 workers, chunked by row range) before the full run** — it should
drop to roughly 70-90s and it is embarrassingly parallel.

### Original plan (unchanged where not superseded)

One shared function used by blocking, features, everything.

- Strip leading/trailing junk (`-- Holloway Peak Inc`, `<< Team Ecole`) **first** —
  non-alphanumeric prefixes poison tokenization and sort keys.
- Lowercase, unicode fold (`unidecode`), collapse punctuation, `&` -> `and`.
- **Eight Indic scripts, not just Devanagari.** Detect the script by Unicode block
  (Devanagari 0900-097F, Bengali 0980-09FF, Gurmukhi 0A00-0A7F, Gujarati 0A80-0AFF,
  Oriya 0B00-0B7F, Tamil 0B80-0BFF, Telugu 0C00-0C7F, Kannada 0C80-0CFF), then route
  through `indic-transliteration` `sanscript` with the matching source scheme.
  Offline and rule-based, so it stays inside the fair-play rules.
- **Transliteration is one-directional: Indic -> Latin.** S1 is 100% Latin in both
  train and test, so there is never a reason to go Latin -> Indic. Normalize every
  record into a single Latin space and compare there.
- Apply to `business_address` too — it is Indic for ~9% of S2/S3 rows.
- Keep a `script` flag as a feature. Measure blocking recall **per script** —
  non-Latin records fail silently otherwise.
- Legal suffix dictionary: corp/corporation, pvt/private, ltd/limited, llc, inc, plc.
- **French forms matter — France is ~15% of the test set and absent from training.**
  Suffixes: `sarl`, `sas`, `sasu`, `sa`, `eurl`, `sci`, `snc`. Street words:
  `rue`, `avenue`, `boulevard`, `place`, `impasse`, `chemin`, `allee`/`allée`.
  These are unexercised by any training-derived validation split, so they must be
  unit-tested directly rather than tuned.
- Address abbreviations: rd/road, st/street, blvd, ave, nr/near, opp/opposite.
- Extract structured tokens separately: **house/street numbers, PIN/ZIP codes.** These
  are the strongest discriminators in the data.
- Emit both a "core name" (suffixes stripped) and the raw normalized name.

---

## Stage 6 — Blocking (`src/blocking.py`) — CANDIDATES BUILT, GATE NOT MET (84.98%)

### Measured: index build over full-scale S2+S3 names

Build only — no candidate generation, no recall measurement. Run locally on the full
dataset (10.3M train records, 10 vCPU, 16 GB); SageMaker has 8 vCPU and 64 GB.

| | train | test |
|---|---|---|
| records indexed | 10,320,219 | 9,969,589 |
| postings | 35,830,415 | 35,456,434 |
| unique tokens | 1,457,777 | 1,224,944 |
| mean tokens/record | 3.47 | 3.56 |
| **wall clock** | **~10-14 s** | **~10 s** |
| **peak RSS** | **3.7-4.4 GB** | **4.3-4.5 GB** |
| **index on disk** | **236 MB** | **231 MB** |

Phase split (train): tokenize+explode 21%, DF 5%, posting lists 30%, write 43%.
Throughput ~780k-890k records/s. Index is 0.58x the source Parquet, ~6.6 bytes/posting.

**Memory is elastic but has a floor.** Peak RSS tracks DuckDB's `memory_limit`
(4 GB limit -> 3.7 GB RSS; 10 GB limit -> 4.4 GB). It **fails below ~4 GB** — 1 GB and
2 GB limits both abort. `preserve_insertion_order=false` is set (nothing here depends on
row order) but made no measurable difference; the floor is the posting-list aggregate,
which must materialise all 35.8M postings. At 64 GB there is ~60 GB of headroom, so the
index build is not a memory risk — **stage 7 feature generation is where the budget goes.**

Correctness verified: `sum(df) == sum(len(ids)) == 35,830,415`, no token has
`df != len(ids)`, no duplicate tokens, and posting lists round-trip for both Latin and
Devanagari tokens.

### Tokenizer bug worth remembering

The obvious character class `[^\p{L}\p{N}]` **deletes Devanagari combining marks**
(vowel signs are `\p{M}`), shattering `राम मार्केटिंग प्राइवेट लिमिटेड` into 15
single-character fragments instead of 4 tokens. Caught before the first full run; the
class must be `[^\p{L}\p{N}\p{M}]`. `src/shard.py` had the same bug and is now aligned.

### Two findings that constrain the design

**1. Name-only blocking cannot work — the address index is mandatory, not a supplement.**
A record is retrievable only if it owns at least one token rare enough to survive the IDF
cutoff. Counting records with no such token:

| script | records | no token df<=100 | no token df<=1k | no token df<=10k |
|---|---|---|---|---|
| Latin | 9,563,627 | 63% | 36% | 15% |
| Devanagari | 427,427 | **100%** | **100%** | 2% |
| Other Indic | 325,442 | 99% | 7% | 1% |

Even 36% of *Latin* records have no name token rarer than df=1,000. The plan's
"keep records sharing >= 2 rare tokens (or 1 very rare one)" retrieves nothing for those.

**2. The Indic vocabulary is tiny and uniformly common.** There are only **171 distinct
Devanagari tokens** across 1.6M postings, with a **minimum DF of 1,834** — the data
generator drew Indic names from a small fixed vocabulary. So *every* Devanagari token is
cut by any sane IDF threshold, and **100% of Devanagari records become unreachable by
name blocking**. Transliteration (stage 5) is what rescues them: it maps those tokens
into the Latin space where the rare-token signal actually lives. This is a hard
prerequisite for the >= 97% gate, not an optimisation.

IDF cutoff trade-off measured on train (tokens kept / postings kept):
df<=10 -> 94.2% / 5.2%; df<=100 -> 98.6% / 11.2%; df<=1,000 -> 99.8% / 23.4%;
df<=10,000 -> 100.0% / 41.4%. 79.8% of tokens are hapax (df=1) — near-proof of a match.
Top tokens are exactly the legal suffixes: limited (1.21M), private (1.18M), llc (1.11M),
ltd (832k), inc (826k).

### Conjunctive blocking, measured

Re-measured reachability with a token **pair** rule — does a record own two tokens whose
posting lists intersect in <= 50 records — over tokens with df <= 50,000, intersections
computed exactly, tokens deduplicated per record:

| script | records | reachable | no pair available | pairs exist, all intersect >50 |
|---|---|---|---|---|
| Latin | 9,567,346 | **65%** | 21% | 13% |
| Devanagari | 427,427 | **29%** | 1% | 70% |
| Other Indic | 325,442 | **99%** | 0% | 0% |
| (single-token df<=1k baseline) | | Latin 64% / Devan. 0% / Other Indic 93% | | |

Conjunction rescues Other Indic (93% -> 99%) and moves Devanagari off zero (0% -> 29%),
but **barely helps Latin: 64% -> 65%**. The reason splits cleanly:

- **21% of Latin records have fewer than two tokens with df <= 50k** — there is no pair
  to form. More conjunction cannot help them; only the address index can.
- **70% of Devanagari records have pairs that all intersect above 50** — the 171-token
  vocabulary means its tokens co-occur constantly. Transliteration moves them into the
  Latin token space, which is the right fix, but per the verdict above that is a feature
  win, not a blocking one.

### Blocking cost, measured on 2,206,821 S1 records

Work = posting-list entries touched. "Rarest-first" scans only the rarest token's list
then verifies; "naive k=3" scans and intersects all three.

| df cap | S1 with >=1 usable token | rarest-first | naive k=3 | mean/S1 | p99 |
|---|---|---|---|---|---|
| 500 | 935,875 | 0.13e9 | 0.17e9 | 138 | 475 |
| 1,000 | 1,051,697 | 0.21e9 | 0.30e9 | 200 | 920 |
| 5,000 | 1,422,340 | 1.02e9 | 2.30e9 | 718 | 4,288 |
| 10,000 | 1,694,602 | 3.24e9 | 6.20e9 | 1,913 | 9,679 |
| 50,000 | 2,194,380 | **14.04e9** | **59.87e9** | 6,396 | 38,385 |

At the requested k=3 / df<=50k the naive cost is **59.9e9 posting touches** — not
materialisable as a join. Rarest-first with early termination cuts that to 14.0e9, and
dropping the cap to df<=5,000 cuts it to 1.0e9 while still covering 64.5% of S1 records
by name. **Operating point: rarest-first, df cap ~5,000, address index for the rest.**

Coverage is the binding constraint, not cost: at df<=1,000, **52.3% of S1 records have
no usable name token at all**; even at df<=50,000 it is 0.6%.

### Jaccard re-rank (stage 6b) — +1.87pp at cap 30

Re-ranking the pre-capped candidates by token-set Jaccard over combined name+address
tokens. `PRE_CAP` was raised from 50 to 100 so the re-rank has something to reorder; the
df caps and the blocks themselves are unchanged.

| ranking | @10 | @20 | @30 | @50 |
|---|---|---|---|---|
| blocking IDF score (baseline) | 81.02% | 83.80% | 84.98% | 86.13% |
| jaccard plain | 84.61% | 86.16% | 86.66% | 87.05% |
| **jaccard IDF-weighted (winner)** | **84.90%** | **86.40%** | **86.85%** | **87.17%** |

IDF-weighted wins at every cap, and the gain is largest where it matters most for cost
(+3.88pp at cap 10). Only the intersection is unnested — `sum_idf(A u B) = sum_idf(A) +
sum_idf(B) - sum_idf(A n B)` with the per-record sums precomputed — and the whole thing
is partitioned on `hash(s1) % 8`, because materialising 130M rows carrying list columns
overflowed 40 GiB of spill in one pass.

### Candidate generation — ADDRESS-FIRST. This inverts the original design.

**Names are template-generated from small word pools, so the address is the primary
discriminator.** The original plan led with name blocking; the data does not support it.
There are only 171 distinct Devanagari name tokens corpus-wide; every ground-truth pair
with an Indic match has an S1 rarest-name-token DF >= 8,928; and the name-only ceiling
(pairs sharing any name token at all) is 87.26% versus 95.30% for addresses. Name blocks
are kept, but as a secondary signal.

Seven blocks, measured independently on the full training data (7,638,365 GT pairs):

| block | pairs generated | GT found | recall alone |
|---|---|---|---|
| B1 postcode exact | 7,196,072 | 381,343 | 4.99% |
| B2 street-number + rare addr token | 101,710,279 | 4,646,638 | 60.83% |
| B3 rare addr token pair | 115,209,674 | 3,235,165 | 42.35% |
| B4 rare name token pair | 11,101,070 | 1,650,620 | 21.61% |
| B5 single addr token (df<=200) | 72,669,297 | 2,662,129 | 34.85% |
| B6 single name token (df<=200) | 68,710,486 | 2,174,227 | 28.46% |
| **B7 addr-token x name-token** | **33,546,509** | **4,949,430** | **64.80%** |
| **UNION (uncapped)** | **373,476,010** | **6,807,107** | **89.12%** |

**B7 is the best block in the set by a wide margin** — the cross-field composite gets
64.80% recall from 33.5M pairs, while the within-field address pair block (B3) gets
42.35% from 115M. Neither field alone is reliably rare; the combination is selective.

### Results at cap 30

| | value |
|---|---|
| recall @ cap 10 / 20 / 30 / 50 | 81.02% / 83.80% / **84.98%** / 86.13% |
| mean / p99 candidates per S1 | 24.7 / 30 |
| S1 with zero candidates | 48,698 (2.2%) |
| reduction ratio | 0.9999976550 |
| runtime / peak RSS | 364s / 6.4 GB |

Per script (capped): Latin 86.25%, Malayalam 78.85%, Gurmukhi 73.31%, Bengali 73.90%,
Gujarati 71.81%, Tamil 70.36%, Telugu 68.55%, Devanagari 66.69%, Kannada 66.44%.
Per country (capped): US 87.12%, India 81.77%.

### GATE: FAIL — 84.98% against a 97% target. Where the 12pp sits

Decomposed, so the next move is not a guess:

- **10.9pp is generation.** The uncapped union reaches only 89.12%.
- **4.1pp is capping.** 89.12% uncapped -> 84.98% at 30, so ranking drops true pairs that
  were generated.

**The ceiling is not the problem.** Of all 7,638,365 GT pairs, **99.94% share at least one
token** (address 95.30%, name 87.26%, either 99.94%) — only 4,333 pairs share nothing at
all. So 97% is reachable in principle; the loss is entirely in the DF caps that keep the
join affordable, not in normalization quality.

The gap is concentrated where the caps bite. From the v1 miss analysis: of the pairs the
union missed, 53.5% shared **zero** address tokens below df=5,000 and 30.0% shared exactly
one; for names 40.3% and 44.0%. Widening is the lever, and its cost is measured:

| single-token block | df<=200 | df<=1,000 | df<=5,000 |
|---|---|---|---|
| address: recall alone | 33.82% | 55.92% | 78.56% |
| address: join pairs | 76.5M | 702.7M | 5,143.8M |
| name: recall alone | 28.46% | 42.36% | 56.36% |
| name: join pairs | 69.3M | 298.4M | 2,368.4M |

Raising B5/B6 to df<=1,000 should add roughly 8-12pp to the union for ~1e9 extra pairs
(about 3x the current 373M) — feasible but it needs the partitioned union and more disk.
The 4.1pp lost to capping wants a better ranker than summed key-token IDF; a cheap
Jaccard over the full token sets, computed only on the ~53M capped candidates, is the
obvious next step and is nearly free at that size.

### Engineering notes

- **405M block rows keyed on two entity-id strings overflowed 35 GiB of spill** in a
  single group-by. The union is now partitioned on `hash(s1) % 8`, which is exact — a
  given S1 lives in one partition, so per-S1 ranking and distinct counts are unaffected.
- **Stage 5 bug found here, not in stage 5's own tests.** Mixed-script fields
  ("5/177-H, ERNAKULAM, കേരളം" — Latin number, Latin city, Malayalam state) routed their
  ASCII tokens through Indic transliteration, so the anusvara rule turned every capital M
  into n: `ERNAKULAM` -> `ernakulan`, `Mumbai` -> `numbai`, `Mullai` -> `nullai`. Those
  records then shared no token with their Latin S1 counterpart. ASCII tokens now pass
  through untouched. Worth +1.2pp overall and +4-5pp on Indic scripts.

### Still to build

Candidate generation refinements: rare-token lookup, posting-list union, the >= 2-rare-tokens rule,
the separate address index (PIN/ZIP, street number, rare address tokens), the ~30-per-S1
cap, and recall measurement against the val split.

### Original plan (unchanged where not superseded)

Inverted index, not similarity search. Instead of "which records are similar to this
one?", ask "which records share a rare token with this one?" — a dict lookup returning
~200 records instead of scanning 10M.

- Tokenize normalized S2+S3 names and addresses; compute document frequency.
- Drop high-DF tokens via an IDF cutoff. "Limited" appears in a million records and
  carries no signal; a rare surname is near-proof.
- For each S1 record: look up its rare tokens, union the posting lists, keep records
  sharing >= 2 rare tokens (or 1 very rare one).
- Separate index on address: PIN/ZIP, street number, rare address tokens.
- Candidate set = union of name and address blocks, capped at ~30 per S1.
- Build the index as a DuckDB table if a Python dict will not fit in RAM.

**Gate: measure blocking recall on the validation split. Target >= 97%. Do not start
modelling until it clears.** Also report reduction ratio and average candidates per S1 —
the organizers audit blocking quality, and these numbers go in the write-up.

**Report recall broken down per script and per country, not just overall.** An overall
97% can hide a collapse on the ~5% of records in a minority script, or on France, which
has no training signal at all. A headline average is exactly where that failure hides.

---

## Stage 7 — Features (`src/features.py`) — BUILT

35 features per pair, chunked by `hash(s1)` so a whole shortlist is always in one chunk —
the context features need it. String metrics via `rapidfuzz.process.cpdist` (elementwise,
C++, `workers=-1`). Char-3-gram cosine is the one metric RapidFuzz lacks; it uses a sparse
hashing vectorizer and a row-wise dot product rather than per-pair Python set arithmetic,
which would have dominated runtime.

Throughput ~23k pairs/s: 5.13M train-fold pairs in 233s, 10.67M val in 449s.

**A leak caught here, worth remembering.** The fold filter was first written in SQL as
`hash(c.s1) % 1000000 / 1e6 < 0.2`. DuckDB's `hash()` and Polars' `hash(seed=13)` are
different functions, so the "val" feature set overlapped the real val fold by only 20% —
and a fifth of the val entities ended up in the training sample. It showed up as a
macro-F_0.5 of 0.214 with an AUC of 0.9985, a gap too large to be anything but a wiring
bug. **The fold must be derived by `src/scoring.py` itself** and handed to DuckDB as a
registered table; it is never re-implemented in SQL.

### Original plan

Per (S1, candidate) pair. Use **RapidFuzz** — C++, releases the GIL, so
`process.cdist` genuinely uses all 8 cores. Biggest runtime lever in the pipeline.

Name similarity:
- Jaro-Winkler, Levenshtein ratio, token-set ratio, token-sort ratio, char-ngram cosine.
- **IDF-weighted token overlap** — rare-token agreement is far stronger evidence than
  common-token agreement. This feature usually carries the model.

Address:
- Same similarity set.
- Numeric-token exact match, PIN/ZIP match, street-number match. Note PIN/ZIP is
  extractable from only ~5.5% of addresses — high precision, very low coverage.
- **"Component missing on one side" flags.** ~3% of S2/S3 addresses are NULL (never
  empty-string), so the address feature block is absent for those pairs. A null must
  read as "unknown", never as "no similarity" — an imputed zero says the addresses
  disagree, which is a false-merge signal pointing the wrong way.

Metadata:
- Country equality. Source indicator (S2 vs S3 — different noise profiles).
- Script match/mismatch flag.

**Context features — do not skip.** Matching is competitive, not independent. A 0.7
score means something very different when the runner-up scored 0.2 versus when three
others also scored 0.7. Give the model the field:
- candidate's rank within its S1's shortlist
- score minus the best score for that S1
- score minus the second-best
- ratio to the mean score for that S1
- count of candidates above a cutoff for that S1

---

## Stage 8 — Model (`src/train.py`) — BUILT

LightGBM binary, 5,016,119 pairs (637,156 positives, 12.70%), 35 features, GroupKFold(4)
grouped by S1 entity. **CV AUC 0.99853 +/- 0.00001, OOF AUC 0.99853.**

Feature gain: `jac_idf` **38.9%**, `c_ratio_mean` 14.3%, `c_rank` 10.4%, `a_numov` 8.8%,
`n_ngcos` 3.9%, `a_tset` 3.5%, `jac_plain` 3.0%, `n_tsort` 2.5%, `n_tset` 2.3%.

Two things stand out. The IDF-weighted token Jaccard alone carries 39% of the gain, which
is why it was worth building as a blocking re-ranker first. And the **context features are
the second-largest block at ~25% combined** — `c_ratio_mean` and `c_rank` outrank every
individual string metric, confirming that matching here is competitive rather than
independent.

### Original plan

- LightGBM binary classifier on pairs.
- Positives from ground truth; negatives are blocked candidates that aren't positives
  (naturally hard negatives).
- Train on a stratified sample (~5M pairs). The full pair set is neither necessary nor
  loadable.
- Cross-validation grouped by S1 entity.

---

## Stage 9 — Decision layer (`src/decide.py`) — BUILT

Where F_0.5 is actually won. Most teams stop at "threshold the classifier".

**Built. Val macro-F_0.5 = 0.847264** at tau=0.25, delta=0.35, swept jointly:

| tau \\ delta | 0.05 | 0.10 | 0.20 | 0.35 | 0.50 | 1.00 |
|---|---|---|---|---|---|---|
| 0.15 | 0.8259 | 0.8365 | 0.8445 | 0.8472 | 0.8434 | 0.8072 |
| **0.25** | 0.8259 | 0.8365 | 0.8445 | **0.8473** | 0.8436 | 0.8245 |
| 0.50 | 0.8258 | 0.8362 | 0.8441 | 0.8467 | 0.8431 | 0.8431 |

The surface is almost flat in tau and strongly shaped by delta, which is the signature of
a well-separated classifier: what matters is not *whether* to accept but *how many*.
delta=0.35 beats both 0.05 (too strict, one match only) and 1.00 (accept everything above
tau) — consistent with a median of 3 true matches per entity.

One-to-one conflict resolution is worth **+0.00185** (0.845415 -> 0.847264). Small but
free, and it is the right sign.

Recon overturned the original premise here. The singleton rate is **5.58%**, not high,
and the mean S1 entity has **3.46** matches with 47.9% having four or more.

- Global threshold swept directly against macro-F_0.5.
- **Do NOT favour empty predictions.** Predicting empty for every S1 entity scores
  macro-F_0.5 = 0.0558 — a floor, not a strategy. Silence is only correct for 1 entity
  in 18.
- **Floor rule: if no candidate clears tau, still emit the top-1.** Emitting nothing
  scores 0.0 on the 94.4% of entities that do have a match, so a wrong guess costs
  nothing relative to silence while a right one gains everything. The expected value of
  a top-1 guess is positive whenever its precision exceeds ~0 — which it always does.
  Tune tau for **precision without silence**, not for when to abstain.
- Margin rule: accept the 2nd/3rd match only if within delta of the top score. With a
  median of 3 true matches, delta should be tuned to *admit* several matches, not to
  suppress them — the original "one confident match beats three shaky ones" intuition
  is wrong for this distribution.
- **One-to-one conflict resolution — CONFIRMED, implement it.** Zero violations across
  all 7,638,365 ground-truth pairs on the full dataset. If an S2/S3 ID is claimed by
  two S1 entities, award it to the higher score (greedy or Hungarian). Pure precision
  gain, zero recall cost.

---

## Unseen-country robustness

France is in test, not train — **~15% of test rows**, a large blind spot. One-hot
country encoding would leave the model with nothing for France — confidently wrong, and
invisible until the private leaderboard.

Explicit check: **train on US only, score on India**, and vice versa. If performance
collapses, features are overfit to country-specific patterns — drop or regularize them.

---

## Write-up requirements

Documentation must cover methodology, blocking strategy, model architecture, and feature
engineering. Include measured numbers: blocking recall, reduction ratio, an ablation
table, and the threshold sweep curve. Top teams' packages are reviewed in detail before
final rankings — evidence of measured trade-offs reads as rigor.
