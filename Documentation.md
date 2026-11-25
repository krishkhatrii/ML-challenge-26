# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** K-Fold Duo
**Team Members:** Krish Khatri, Karishma Doshi
**Submission Date:** 2026-10-01
**Submitted run:** Exp 7 test-side apply, generated 2026-09-27 (the file on the leaderboard)

---

## 1. Executive Summary

Inverted-index blocking plus a LightGBM pairwise classifier and an F_0.5-tuned decision
layer, run end to end on the full 4.6 GB dataset (10.3 M train / 10.0 M test records) on
an 8-vCPU, 64 GB CPU-only box using Polars and DuckDB over Parquet — no dense embeddings,
no cross-encoder, no pairwise scan of the cross product. The core contribution is a
diagnostic finding turned into a blocking rule: because this corpus is generated from
small word pools, **no single token is ever rare — the combination of tokens is what
identifies a record** — so hashing a record's entire sorted token set into one blocking
key (blocks B8/B9) lifted candidate recall @cap 30 from 86.87% to **90.46%** and val
macro-F_0.5 from 0.847028 to **0.866908**.

---

## 2. Methodology

### 2.1 Problem Analysis

Findings from `src/recon.py` on the full training set, plus the miss diagnostic
(`tools/exp01_misses.py`, `tools/exp01b_levers.py`):

**Label structure**
- 2,206,821 S1 entities, 7,638,365 ground-truth pairs, mean 3.46 matches per entity
  (median 3, p95 6).
- **Singleton rate 5.58%** (123,247 entities). Predicting empty everywhere scores exactly
  macro-F_0.5 = 0.055848 — the singleton rate. The pipeline must be willing to predict
  nothing.
- **One-to-one holds: 0 violations across 7,638,365 pairs.** No S2/S3 record is claimed by
  two S1 entities, so contested IDs can be resolved greedily — pure precision gain at zero
  recall cost, and F_0.5 weights precision 2x.

**Noise and field quality**
- `business_address` is NULL on 2.7–3.5% of S2/S3 rows; S1 is complete on every field.
- S1 `business_name` is 100% Latin. S2/S3 carry **8 Indic scripts** (Devanagari 3–6%,
  plus Bengali, Gujarati, Gurmukhi, Kannada, Malayalam, Oriya, Tamil, Telugu ≈ 2–5%),
  addresses included. Only **171 distinct Devanagari tokens exist, minimum DF 1,834** —
  the generator used a small fixed Indic vocabulary, so *any* IDF cutoff removes all of
  them and name blocking cannot reach a single Devanagari record without normalization.
- Countries: train = {US, India}; test = {US, India, **France** ≈ 15%}. France has no
  training signal, so `country` is used **only as an equality feature** — never one-hot,
  never filtered.

**The decisive finding — why rare-token blocking is the wrong instrument here**

Anti-joining the ground truth against the uncapped block union left 831,258 pairs (10.88%)
that blocking never generates. Only **0.20% of them share no token at all** — 99.8% are
visible to the data and invisible to the rules. The DF of the rarest *shared* token over
those misses:

| rarest shared token DF | name field | addr field |
|---|---|---|
| none shared | 26.9% | 23.8% |
| **≤ 200 (current cap)** | **0.0%** | **0.0%** |
| 201 – 1k | 9.3% | 5.4% |
| 1k – 5k | 11.1% | 8.3% |
| 5k – 20k | 18.4% | 23.9% |
| 20k – 100k | 33.6% | 23.2% |
| > 100k | 0.7% | 15.5% |

Not one miss has a shared token below df = 200. The DF caps are exactly the binding
constraint, and the mass sits at 5k–100k — the signature of names and addresses drawn from
small word pools. The modal miss is a pair whose normalized token sets are *identical* but
composed entirely of common words (`Surgical Metro Physicians` at
`5441 W Mount Drive, Nashville, NC` vs `5441 1/2 WEST MOUNT DR, NASHVILLE, NC`: shared
tokens `{drive, mount, nashville, nc}`, nothing rare, nothing blocks).

The recurring noise operators, from 50 misses read in full:
1. every token is common (modal case);
2. the S2/S3 address is deleted outright (8 of 50);
3. a typo lands on the one distinctive token (`saffira`/`saffria`, `creston`/`crestton`,
   `96th`/`96nd`);
4. house numbers zero-padded or ranged (`5110`/`005110`, `1223`/`1223-1225`);
5. address truncated to city + state;
6. Indic transliteration is phonetic and one-way (`सूर्य`→`sury` vs `surya`);
7. the name is mangled into a domain (`Pacific Foundation, LLC` → `PACIFICFOUNDATION.COM`)
   — tokens concatenated, so token equality fails entirely;
8. junk affixes (`-- Prime Téchnologies`, `Dermatology Partners [Có]`).

### 2.2 Solution Strategy

**Approach Type:** Blocking + pairwise classifier + F_0.5-tuned decision layer (hybrid,
fully CPU).

**Core Innovation:** **Exact token-set-key blocking.** Instead of asking "does this pair
share a *rare* token?", hash the whole sorted normalized token set of a field into a single
blocking key and equi-join on it. Priced beforehand at +4.86pp of uncapped recall by the
miss diagnostic; measured +4.71pp (97% of prediction) for only +22% more pairs and +26%
candidate-stage runtime. It is the one rule that reaches records whose address was deleted:
the zero-candidate population fell from 13,638 entities to **231**.

Pipeline: convert TSV → Parquet (7 s) → normalize (8 Indic scripts → ITRANS → Latin,
parallelized, 179 s train / 279 s test) → build inverted index (35.8 M postings, ~10 s) →
9 blocking rules → IDF-weighted Jaccard re-rank, cap 30 → 35 pairwise features, chunked →
LightGBM → τ/δ threshold sweep + greedy one-to-one resolution → TSV.

Everything is Parquet in, Parquet out; features are generated in 17 chunks keyed on
`hash(s1)` so the full pair set never sits in RAM. Seed 13 on the submitted path; the train/val
fold split is defined once in `src/scoring.py` and imported, never re-implemented.

### 2.3 Compliance — fair play and model licensing

**No external data lookup of any kind.** The pipeline reads only the TSVs shipped with the
challenge. There is no geocoding, no business-registry or government-database lookup, no
commercial entity-resolution API or service, no web scraping, and no data augmentation
from any internet source. Normalization relies on offline, rule-based libraries —
`Unidecode` (character folding) and `indic-transliteration` (deterministic ITRANS
transliteration) — together with hand-written legal-suffix and address-abbreviation
dictionaries kept in `src/normalize.py`. Neither library consults a network service, and
neither contains business identity data. Every stage runs with no internet access; the
training and test runs that produced the submitted files were executed offline.

**Model licensing and size.** The only learned model is a LightGBM binary classifier
(**MIT**, 4.27 M training pairs, 35 tabular features) — a gradient-boosted tree ensemble,
so parameter count is not remotely near the 8 B cap. No pretrained language model,
embedding model, or cross-encoder is used anywhere in the pipeline, at training or at
inference. Every pinned dependency is MIT, Apache-2.0 or BSD: polars, duckdb, Unidecode,
indic-transliteration, rapidfuzz, lightgbm, scikit-learn, numpy, pyarrow.

**Reproducibility.** Seed 13 on every stage of the submitted path; the train/val fold
split is defined once in `src/scoring.py` and imported by every stage. `code/business_entity_resolution/README.md`
gives the exact end-to-end command sequence that regenerates both output files from the
provided data alone.

---

## 3. Candidate Generation (Blocking)

**Blocking keys used** — 9 rules over the S2+S3 inverted index (35,830,415 postings,
1,457,777 unique tokens, built in ~10 s). Address-first: name-only blocking is not viable
because 36% of Latin records have no name token rarer than df = 1,000.

| block | key | pairs | GT pairs found | recall alone |
|---|---|---|---|---|
| B1_postcode | extracted PIN/ZIP equality | 1,996,587 | 106,302 | 4.97% |
| B2_number_addrtok | street number + rare address token | 28,520,150 | 1,299,472 | 60.81% |
| B3_addr_pair | two address tokens, each df ≤ 5,000 | 32,534,429 | 906,953 | 42.44% |
| B4_name_pair | two name tokens, each df ≤ 5,000 | 3,100,125 | 461,943 | 21.62% |
| B5_addr_single | one address token, df ≤ 200 | 20,331,180 | 744,793 | 34.85% |
| B6_name_single | one name token, df ≤ 200 | 19,181,226 | 609,831 | 28.54% |
| B7_addr_x_name | address token × name token composite | 9,360,898 | 1,384,741 | **64.80%** |
| **B8_name_set** | **hash of full name token set** (key df ≤ 1,000, ≥ 2 tokens) | 22,257,647 | 1,076,780 | **50.39%** |
| **B9_addr_set** | **hash of full address token set** (key df ≤ 1,000, ≥ 2 tokens) | 3,594,088 | 510,669 | 23.90% |

B7 (cross-field composite) is the single strongest classical block — 64.80% recall from
9.4 M pairs, versus 42.44% from 32.5 M for the within-field address pair block. B8 alone
reaches 50.39%.

**Candidate pairs generated**
- Uncapped union: **127,782,287 pairs, 93.86% ground-truth recall.**
- After IDF-weighted Jaccard re-rank and cap 30: **90.46% recall**, mean **26.7 candidates
  per S1**, only **231 S1 entities with zero candidates** (0.0%).
- **Reduction ratio 0.9999974091** (train-side measurement).
- Test side: **48,959,160 candidate pairs** over 1,732,275 of the 1,732,544 test S1
  entities — **28.3 candidates per S1, reduction ratio 0.9999971655**. This is the set
  shipped as `output/candidate_pairs.tsv`; it is exactly what the model scored.

Per country @cap 30: US 91.06%, India 85.13%. Per script @cap 30: Latin 90.33%,
Devanagari 66.03%.

**How true matches were protected from being lost**
1. **Measure recall at every stage against the full 7.6 M-pair ground truth**, per block,
   per country and per script — non-Latin records fail silently otherwise.
2. **Union many complementary rules** rather than tightening one. Address-first, plus a
   cross-field composite, plus set keys; the union is 93.86% against 64.80% for the best
   single rule.
3. **Re-rank before capping.** The blocking score ordering gives 88.68% @30; IDF-weighted
   Jaccard gives **90.46%** @30 — +1.78pp for free, and it beat plain Jaccard at every cap:

   | ranking | @10 | @20 | @30 | @50 | @75 | @100 |
   |---|---|---|---|---|---|---|
   | blocking IDF score | 84.84% | 87.60% | 88.68% | 89.77% | 90.66% | 91.33% |
   | Jaccard plain | 87.13% | 89.41% | 90.18% | 90.81% | 91.13% | 91.33% |
   | **Jaccard IDF-weighted** | **87.68%** | **89.78%** | **90.46%** | 90.98% | 91.22% | 91.33% |

   The pre-cap width was raised from 50 to 100 so the re-ranker has room to work.
4. **Normalize before indexing.** Transliteration lifts token Jaccard against the true
   match from 0.036 → 0.198 for Devanagari (share of pairs sharing *no* token: 92.8% →
   18.4%) and 0.042 → 0.167 for other Indic scripts.
5. **Tune the cap against the metric, not against cost** (see §5, Exp 3): 20, 30 and 50
   were all measured end to end; 30 is an interior optimum.

The ≥ 97% recall gate in the original plan was **not** met — the pipeline reaches 90.46%
@cap 30 against an absolute ceiling of 99.94% (only 4,333 of 7,638,365 GT pairs share no
token at all). Recall, not the classifier, is the binding constraint on the final score,
and the two remaining priced levers are recorded in §5.

---

## 4. Matching Model

**Features used — 35 in total**, all computed pairwise in chunks of ~100 k S1 entities.

- **Name features (7):** `n_jw` (Jaro-Winkler), `n_lev` (normalized Levenshtein),
  `n_tset` (token-set ratio), `n_tsort` (token-sort ratio), `n_ngcos` (hashed character
  n-gram cosine — reaches the domain-mangled names that token equality cannot),
  `n_idfov` (IDF-weighted token overlap), `n_jac` (token Jaccard).
- **Address features (14):** `a_jw`, `a_lev`, `a_tset`, `a_tsort`, `a_ngcos`, `a_idfov`,
  `a_jac`, plus `a_numov` (street-number overlap), `a_numfirst`, `a_numboth`,
  `a_pcmatch` / `a_pcboth` (postcode equality / presence), and `a_null_s1` /
  `a_null_s23` (address-missing flags — 2.7–3.5% of S2/S3 rows).
- **Meta (4):** `m_country_eq` (**equality only — country is an open set of string labels,
  never one-hot, so the unseen France behaves exactly like US and India**), `m_is_s2`,
  `m_script_match`, `m_script_s23`.
- **Blocking provenance (4):** `blk_score`, `n_blocks` (how many rules generated the pair),
  `jac_plain`, `jac_idf`.
- **Context / competitive features (6):** `c_rank`, `c_gap_best`, `c_gap_second`,
  `c_ratio_mean`, `c_n_above`, `c_shortlist` — each candidate's standing *relative to the
  other candidates of the same S1 entity*.

**Model type:** LightGBM (MIT licensed, far under the 8 B-parameter limit), binary
objective, **GroupKFold(4) grouped by S1 entity** so no entity straddles folds.
Trained on 4.27 M pairs (12.70% positive). CV AUC 0.998262 ± 0.00001.

Top feature gains: `jac_idf` **38.9%**, `c_ratio_mean` **14.3%**, `c_rank` **10.4%**,
`a_numov` **8.8%**. The context features contribute ≈ 25% of total gain and **outrank every
individual string metric** — confirming that matching here is *competitive* (is this the
best candidate for this entity?) rather than independent (is this pair similar in the
abstract?).

**Threshold selection method:** direct grid sweep of macro-F_0.5 on the held-out val fold
(440,875 S1 entities, disjoint by construction), never AUC or F1. Two parameters:

- **τ** — absolute score floor a candidate must clear, *subject to the floor rule below*,
  which prevents it from ever removing an entity's last candidate.
- **δ** — **absolute gap** below that entity's best score: accept a candidate only if
  `p ≥ best − δ`. Since scores are probabilities in [0, 1], *small δ is the strict end*
  (δ = 0.05 keeps essentially only the top candidate) and *δ = 1.0 is inert* (no gap
  constraint at all, leaving τ as the sole filter). The decision δ governs is therefore
  *how many to accept* per entity.

A third rule, applied after both: a **top-1 floor**. Any entity whose candidates all fall
below τ has its single highest-scoring candidate added back regardless of score
(`src/decide.py:62-64`). Its only precondition is that the entity has at least one scored
candidate. This matters for reading the table below, and §5 returns to it.

| τ | δ=0.05 | δ=0.1 | δ=0.2 | **δ=0.35** | δ=0.5 | δ=1.0 |
|---|---|---|---|---|---|---|
| 0.05 | 0.8427 | 0.8542 | 0.8633 | 0.8664 | 0.8618 | 0.7743 |
| 0.10 | 0.8427 | 0.8543 | 0.8635 | 0.8668 | 0.8624 | 0.8058 |
| 0.15 | 0.8427 | 0.8543 | 0.8635 | 0.8669 | 0.8627 | 0.8222 |
| 0.20 | 0.8427 | 0.8543 | 0.8635 | 0.8669 | 0.8628 | 0.8333 |
| **0.25** | 0.8427 | 0.8543 | 0.8635 | **0.8669** | 0.8628 | 0.8412 |
| 0.30 | 0.8427 | 0.8542 | 0.8634 | 0.8668 | 0.8628 | 0.8476 |
| 0.40 | 0.8426 | 0.8541 | 0.8633 | 0.8666 | 0.8626 | 0.8568 |
| 0.50 | 0.8426 | 0.8540 | 0.8630 | 0.8663 | 0.8624 | 0.8624 |
| 0.60 | 0.8425 | 0.8538 | 0.8627 | 0.8658 | 0.8651 | 0.8651 |
| 0.70 | 0.8424 | 0.8536 | 0.8622 | 0.8648 | 0.8648 | 0.8648 |

**Selected: τ = 0.25, δ = 0.35 → val macro-F_0.5 = 0.866908.** The surface is nearly flat
in τ and strongly shaped by δ, and the two end columns explain why — they are the clean
test of the semantics above.

At **δ = 0.05** the column is constant at 0.8427 across every τ from 0.05 to 0.70. The gap
rule has already reduced the prediction to roughly the top candidate alone, and the top
candidate is exactly what the floor rule reinstates when τ would have removed it, so τ has
nothing left to act on. **At δ = 1.0** the gap rule is inert and τ becomes the only filter,
which is why that column alone swings widely — 0.7743 at τ = 0.05 up to 0.8651 at τ = 0.60.
Both ends are consistent only with δ as an absolute gap plus an unconditional top-1 floor.

The practical consequence: **τ is a weak lever by construction, not because the classifier
is well separated.** It cannot remove an entity's best candidate — only its second, third
and fourth. The real decision this layer makes is how far down the shortlist to go, and
δ = 0.35 is where that is priced best. The strict end (δ = 0.05, effectively top-1 only)
costs 0.024, which is the cost of ignoring that the mean entity has 3.46 true matches.

**Greedy one-to-one resolution**, licensed by the 0-violation recon finding: a contested
S2/S3 ID is awarded to the highest-scoring S1 entity and dropped from the rest. Worth
**+0.002205** (0.864703 → 0.866908).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro): 0.866908** on the held-out val fold (440,875 S1 entities,
  τ = 0.25, δ = 0.35). The previous configuration's val 0.847264 scored **0.8397** on the
  leaderboard, so val has tracked the leaderboard within ~0.008. (That 0.847264 is the
  full-run figure; the ablation table in this section reports the same baseline as
  **0.847028** because every ablation row is re-anchored under the restricted-S1
  measurement protocol described below. The 0.000236 difference between the two is
  protocol, not configuration.)
- **Submitted test-side run:** 48,959,160 candidate pairs scored, 1,732,544 rows (every
  test S1 entity), 1,714,478 with ≥ 1 match (99.0%), 5,372,239 matched IDs.
  `utils/validate_submission.py` **PASS**; 0 quote characters in either file.

### Singleton behaviour — the pipeline cannot abstain, and this is its largest known defect

The training singleton rate is 5.58%, but the submitted run predicts empty for only
**18,066 of 1,732,544 test entities (1.04%)**. A code audit of `src/decide.py` established
that this is not a tuning outcome. It is structural, and the mechanism is not the one the
threshold sweep would suggest.

**τ cannot produce an empty prediction.** The top-1 floor rule (§4, `decide.py:62-64`)
reinstates the best candidate for any entity whose candidates all fall below τ, gated only
on the entity having at least one scored candidate. Every entity that reaches the decision
layer with a candidate therefore leaves it with at least one row, at any τ. The two
parameters swept in §4 control *how many* matches an entity receives, never *whether* it
receives one.

The 18,066 empty rows decompose accordingly:

| source of the empty prediction | count | mechanism |
|---|---|---|
| no candidate generated by blocking | 269 | entity never enters the scored frame |
| lost every one-to-one contest | ≈ 17,797 | every ID it claimed was awarded to a higher-scoring entity |
| τ / δ thresholding | **0** | overridden by the top-1 floor |

(The 17,797 is obtained by subtraction from the validator counts and the apply log, not
counted directly from the decision frame.)

So the ~96,700 true singletons we should expect in a 1.73 M-entity test set are, with few
exceptions, being assigned a match — and each one scores 0.0 rather than 1.0. What
abstention does occur is **incidental**: a by-product of greedy one-to-one resolution
stripping an entity of every ID it claimed, with no particular tendency to select true
singletons. The one rule in the pipeline that produces empty predictions was not designed
to produce them.

**Why this went unnoticed.** The floor rule is a sensible default in a recall-oriented
setting and was carried forward without being re-examined against F_0.5, where a correct
abstention is worth a full 1.0. Because τ appears in the sweep and the sweep is flat in τ,
the surface *looks* like evidence that abstention was priced and rejected. It is not —
τ was inert the whole time. The sweep measured a lever that was disconnected, which is a
more dangerous failure than a lever measured and found wanting.

**Headroom.** At a 5.58% prior and 1.0 per correct abstention, the singleton population is
worth up to ~0.045 macro-F_0.5 — larger than the +0.0199 that Exp 7 contributed, and
larger than anything remaining in blocking. Two changes follow, in order of cost:

1. **Gate the floor rule** on a minimum top-1 score, or remove it, and re-sweep τ. This is
   a few lines and would make τ a live parameter for the first time. We did not run it: it
   lands after the final test-side apply, and shipping an unvalidated decision layer
   against an audited one was not a trade worth making at this point in the schedule.
2. **A dedicated per-entity has-any-match classifier** over the shortlist-shape features
   that already carry ≈ 25% of the pairwise model's gain (`c_gap_best`, `c_gap_second`,
   `c_ratio_mean`, `c_n_above`, `n_blocks`), deciding abstention before any per-pair
   threshold. A true singleton's best candidate is a plausible business that simply is not
   the right one, so it is not separable on absolute score — but the *shape* of its
   shortlist (no standout, tight cluster of mediocre scores) is exactly what those features
   encode. This is the principled fix and the first thing we would build with more time.

This is reported in full because the measurement discipline elsewhere in this document is
only worth something if it also reports the place where it failed. The finding came from
auditing the code against this write-up, after the submitted run was frozen.

### Ablation table

Every row is val macro-F_0.5 on the same clean disjoint val fold under one measurement
protocol, one variable per experiment. (Blocking is exactly per-S1 independent, so the S1
side is restricted to the 617,069 entities actually evaluated; this reproduces every
full-run number to within 0.1pp and cuts the candidate stage from 487 s to 92 s.)

| exp | change | blocking recall @30 | CV AUC | val macro-F_0.5 | verdict |
|---|---|---|---|---|---|
| 00 | baseline, re-anchored under this protocol | 86.87% | 0.998501 | 0.847028 | reference |
| 01 | miss diagnostic (no change) | — | — | — | KEEP — reordered the backlog |
| 02 | B5/B6 single-token cap df ≤ 200 → 1,000 | 86.81% | 0.998670 | 0.846530 | **REVERT** |
| 03 | re-rank cap 20 | 86.42% | 0.997870 | 0.845706 | **REVERT** |
| 03 | re-rank cap 50 | 87.19% | 0.998990 | 0.846777 | **REVERT** |
| 03 | re-rank cap 30 (unchanged) | 86.87% | 0.998501 | 0.847028 | **KEEP** |
| **07** | **B8/B9 exact token-set keys** | **90.46%** | 0.998262 | **0.866908** | **KEEP — best** |

Net gain **+0.019880**, all of it from Exp 7. **The negative results are the most
informative part of this table**, for three reasons:

1. **Exp 2 — extra recall that does not survive the cap.** Widening the single-token DF cap
   added +1.59pp of *uncapped* recall (exactly the +1.59pp the miss diagnostic predicted)
   and 3.08x the pairs — yet recall @30 after re-ranking went **down** 0.06pp and F_0.5 fell.
   The pairs it adds share only one common token, so their IDF-Jaccard is low and the
   re-ranker buries them below 30, while 218 M extra false pairs give it more chances to
   err. Generation stopped being the binding constraint at cap 30; ranking became it.
2. **Exp 3 — cap 30 is an interior optimum, and the two ends fail differently.** Cap 20
   loses recall faster than it gains precision (−0.45pp recall costs −0.0013 F_0.5: even at
   2x precision weighting, dropping a true match hurts more than the false ones avoided,
   because the mean entity has 3.46 matches). Cap 50 gains +0.32pp recall the decision layer
   cannot convert — τ has to rise from 0.20 to 0.30 to hold back 20 extra candidates per
   entity, and it cannot do that selectively.
3. **CV AUC is anti-correlated with the objective here.** Exp 2 and Exp 3 both *raised* AUC
   while lowering F_0.5 (bigger shortlists add easy negatives — free AUC, negative F_0.5).
   Exp 7 *lowered* AUC by 0.00024 and raised F_0.5 by 0.0199, because the pairs it adds are
   hard (near-duplicate names at different addresses) while making the shortlist far more
   likely to contain the right answer. **Anyone tuning this pipeline on AUC would have
   rejected its single best change.** This is why every threshold and every block was
   judged on macro-F_0.5.

The diagnostic proved trustworthy as a planning instrument twice: predicted +1.59pp for
Exp 2, measured +1.59pp; predicted +4.86pp for Exp 7, measured +4.71pp.

### Common false positives (wrong merges)

No FP-specific diagnostic was run, so this is what the sweep and the recon show rather than
a taxonomy. The precision risk is concentrated in the *shape of the shortlist*, not in any
one feature: the sweep surface is nearly flat in τ but collapses as δ moves away from 0.35,
which says wrong merges are mostly *extra* candidates on entities that already have a
correct best match rather than absolute-score failures. Two mechanisms are handled
explicitly — contested IDs (greedy one-to-one, +0.002205, licensed by 0 one-to-one
violations in the ground truth) and singletons, which are 5.58% of entities and score 1.0
only when predicted empty. The decision layer as shipped does **not** emit nothing on
score — see "Singleton behaviour" above, where this is reported as the pipeline's largest
known defect.

### Common false negatives (missed matches)

**Blocking is the dominant loss: 9.54% of ground-truth pairs never reach the model at
cap 30.** The residue after the Exp 7 keys, priced against 831,258 originally-missed pairs:

- **Mid-frequency-token pairs (largest remaining bucket).** Pairs sharing two tokens of
  df 5k–100k each. The current rules cap *each* token's df, which forbids a 20k × 20k pair
  whose expected intersection is 39 records while admitting 5k × 5k whose expected
  intersection is 2.4. A product-DF budget (`df1 × df2 ≤ 50 N`) is priced at **+5.93pp**
  uncapped and is implemented behind `ENABLE_PRODPAIR=1`, unrun.
- **Zero-padded and ranged street numbers.** `5110`/`005110`, `7`/`07`, `1223`/`1223-1225`
  fail string equality; 51.97% of misses already share a street number. Priced at
  **+0.83pp**, implemented behind `ENABLE_NUMNORM=1`, unrun.
- **Indic names, unreachable via the name field by construction.** Transliteration is
  phonetic and one-way (`सूर्य`→`sury` vs `surya`, `ईस्टर्न`→`istarn` vs `eastern`), and
  per-script recall confirms it: Devanagari stayed at 66.03% through Exp 7 (from 66.05%)
  while Latin went 86.27% → 90.33%. Any further Indic gain must come from the address.
- **Empty-address, template-name pairs.** 66.9% of the 83,668 pairs that remain unreachable
  even under the full priced package have an empty S2/S3 address. Nothing address-based can
  ever reach them.

---

## 6. Conclusion

A measurement-first pipeline — inverted-index blocking, an IDF-weighted Jaccard re-rank,
35 pairwise features, LightGBM, and a τ/δ decision layer swept directly against
macro-F_0.5 — reaches **val macro-F_0.5 0.866908** at full dataset scale on 8 CPU cores.
The lesson that produced most of the gain was diagnostic, not algorithmic: reading the
misses showed that rare-token blocking is simply the wrong instrument for a corpus
generated from small word pools, and that making the whole token set the key gains 4.71pp
of recall where widening a rarity cap gains 1.59pp and *loses* score at the cap. The
second lesson is that the standard proxy actively misleads here — CV AUC moved opposite to
macro-F_0.5 in three separate experiments — and the third is that the biggest single risk
to this submission was never modelling at all, but a formatting default: Polars writing
`""` for an empty string silently scored a correct baseline as 0.000 instead of 0.0558.

The fourth lesson arrived last and is the one we would act on first. Auditing the code
against this document revealed that the top-1 floor rule in `decide.py` makes τ incapable
of producing an empty prediction, so the pipeline effectively never abstains — 1.04%
predicted empty against a 5.58% singleton prior — and the τ sweep that appeared to price
abstention was measuring an inert parameter. A flat response surface is not evidence that
a lever was tested; it can equally mean the lever is disconnected. On this metric that
oversight is worth more than every blocking gain in the ablation table combined.

---

## Appendix

### A. Code Artefacts

Complete runnable code ships under `code/business_entity_resolution/`, with pinned
`requirements.txt` (polars 1.44.2, duckdb 1.5.5, Unidecode 1.4.0,
indic-transliteration 2.3.82, rapidfuzz 3.14.6, lightgbm 4.7.0, scikit-learn 1.9.1,
numpy 2.5.3, pyarrow 25.0.1) and a `README.md` giving exact end-to-end reproduction steps.

| module | role |
|---|---|
| `src/convert.py` | 7 TSVs → Parquet, once, up front (2.3 GB in 6.9 s) |
| `src/recon.py` | EDA: match distribution, singleton rate, one-to-one check, scripts, NULLs |
| `src/normalize.py` | 8 Indic scripts → ITRANS → Latin, legal-suffix and abbreviation folding, house-number and PIN extraction; 8-way parallel, byte-identical to single-core |
| `src/blocking.py` | inverted index, the 9 blocks (switches ~lines 390–425), IDF-Jaccard re-rank, cap |
| `src/features.py` | the 35 pairwise features, chunked on `hash(s1)` |
| `src/train.py` | LightGBM, GroupKFold(4) by S1 entity |
| `src/decide.py` | τ/δ sweep, absolute-gap rule, top-1 floor, greedy one-to-one, submission writer |
| `src/scoring.py` | validated macro-F_0.5 and the canonical fold split — imported, never re-implemented |
| `src/shard.py` | 5%-of-S1 dev shard (GT self-consistency 100%, no country skew) |
| `utils/validate_submission.py` | pre-upload submission validator |
| `tools/run_exp.sh`, `tools/run_test_side.sh` | one-command train-side experiment / test-side regeneration |
| `tools/exp01_misses.py`, `tools/exp01b_levers.py` | the miss diagnostic and the lever pricing behind §2.1 |

Entry points to reproduce the two output files are in the README; the test-side sequence is
`convert → normalize → blocking → features → decide --apply`, then
`utils/validate_submission.py`.

Measured wall clock on 8 threads: convert 7 s; normalize 179 s train / 279 s test;
candidates 116 s; re-rank 139 s; features 236 s train-fold, 486 s val, 1,950 s test
(48.96 M pairs in 17 chunks); train 349 s; sweep 146 s; apply ~200 s.

### B. Additional Results

**Priced but unrun levers** (upper bounds on uncapped recall from `tools/exp01b_levers.py`,
measured against the 831,258 missed pairs — each says the rule *can* see the pair, not what
pair volume comes with it):

| lever | misses reached | +pp GT recall | status |
|---|---|---|---|
| identical name token set | 313,380 (37.7%) | +4.10 | **shipped as B8** |
| identical addr token set | 81,109 (9.8%) | +1.06 | **shipped as B9** |
| product-DF pair budget `df1·df2 ≤ 50 N` | 453,273 (54.5%) | +5.93 | coded, `ENABLE_PRODPAIR=1`, unrun |
| street-number normalization (zeros, ranges) | 63,570 (7.6%) | +0.83 | coded, `ENABLE_NUMNORM=1`, unrun |
| ≥ 2 shared tokens, any DF | 780,443 (93.9%) | +10.22 | rejected — pair volume |

A triple-token rule is strictly worse per unit of budget than the pair rule
(`df1·df2·df3 ≤ 50 N²` reaches 70.4%, below the pair rule at `200 N`), so pairs are the
right arity. The full priced package would imply 98.88% uncapped union recall; its residue
is 83,668 pairs, 66.9% of them with an empty S2/S3 address.

**Bugs found — six of them cost real score, and four were invisible to unit tests.**

| bug | symptom | how it was caught | status |
|---|---|---|---|
| Polars `write_csv` emits `""` for empty strings | with the spec's unquoted format, a singleton round-trips as a literal 2-character ID; the predict-empty baseline scored **0.000 instead of 0.0558** and nothing in the file looks wrong | scoring the baseline | **FIXED** — `quote_style="never"` on every submission write, enforced repo-wide, 0 quote characters verified before each upload |
| train/val fold-hash mismatch | `features.py` filtered the fold with DuckDB's `hash()`, `scoring.py` defines it with Polars' `hash(seed=13)`; the "val" set overlapped real val by 20% and **a fifth of val leaked into training** — macro-F_0.5 0.214 against AUC 0.9985 | a gap too large to be anything but wiring | **FIXED** — fold derived in `scoring.py` and registered with DuckDB; never re-implement the split |
| tokenizer class `[^\p{L}\p{N}]` | deleted Devanagari combining marks, shattering one name into 15 fragments | pre-run inspection of the index | **FIXED** — `[^\p{L}\p{N}\p{M}]`, aligned in `shard.py` too |
| anusvara `M`→`n` applied to Latin text | mixed-script fields sent ASCII tokens through Indic transliteration: `ERNAKULAM`→`ernakulan`, `Mumbai`→`numbai` | a cross-record recall measurement, **not** stage 5's own per-field tests | **FIXED** — worth +1.2pp overall, +4–5pp on Indic scripts |
| `blocking.py` `del gt_path.fetchone()[0]` | botched edit, syntax-valid, so it survived a parse check; per-block recall reporting raised `NameError` on any split with labels | first train-side recall run | **FIXED** |
| union phase scanned the 373 M-row block glob twice per partition | wasted ~half the candidate stage; an earlier single 405 M-row group-by also overflowed 35 GiB of spill | profiling the stage | **FIXED** — one pass, partitioned on `hash(s1) % 8` |
| `PARQUET_DIR` defaulted to the 50 k-row **sample** | silently measured recall against 172,727 sample GT pairs instead of 7,638,365 (it agreed to 0.03pp, which hid it) | cross-checking pair counts | **worked around** — every run sets `PARQUET_DIR=data/parquet_full` explicitly |
| `normalize.py` classifies house numbers as postcodes | `00899 JEFFERSON RD` → `postcode='00899'`, so B1 partly joins on house numbers and `a_pcmatch`/`a_pcboth` are partly noise | Exp 1 miss inspection | **OPEN** — not currently costing recall (postcode equality holds for 0 of the 831 k misses), but it is wrong and pollutes two features |

Two design decisions were also reversed by measurement. **Key-space sharding was built,
measured and demoted**: no raw field gives a shard key that agrees between an S1 record and
its true match (leading trigram agrees only 77.3% of the time, because S1 is 100% Latin
while S2/S3 carry 8 Indic scripts), so it yields 81.7% GT self-consistency and skews
against India — sharding is by 5% of S1 entity IDs instead, at 100.00% self-consistency and
rival-retention ratio 1.0000. And **transliteration turned out to be a feature-quality win
rather than a blocking prerequisite**: for the 551,240 GT pairs with an Indic-script side,
0.0% of the S1 records have a name token with df ≤ 1,000 in the S2+S3 index (Latin control:
51.4%), while 82.2% have an address token that rare — so the address is the route for those
pairs, and the name field matters for scoring, not for reaching them.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
