# Experiments

Autonomous overnight loop, 2026-09-26. One variable per experiment. Every result is
val macro-F_0.5 on the clean disjoint val fold (`scoring.assign_fold`, seed 13,
440,875 S1 entities) unless stated otherwise. Blocking recall is measured over all
7,638,365 training GT pairs.

**Baseline carried in:** leaderboard 0.8397, val macro-F_0.5 0.847264, blocking recall
86.85% @cap30 (uncapped union 89.12%), CV AUC 0.99853, tau=0.25 delta=0.35.
Snapshot: `best/exp00_baseline_0.8397/`.

## Exp 0 — prerequisite fix (not an experiment)

`src/blocking.py` line 592 was corrupted by a botched edit: the per-block recall query
read

```python
hit = con.execute("""...""")
del gt_path.fetchone()[0]
```

so `candidates` raised `NameError: gt_path` in its report section on any split that has
ground truth — i.e. every train run. Syntax-valid, so it survived a parse check. Fixed
to `.fetchone()[0]` on the `hit` assignment. No behavioural change to the candidate set
itself; the crash was after all parquet output was written.

## Exp 1 — Inspect the misses

**Hypothesis:** the 10.9pp the uncapped union never generates has a structure the
current blocks cannot express, and naming it reorders the rest of the backlog.

**Change:** none. Diagnosis. `tools/exp01_misses.py` anti-joins the full ground truth
against the uncapped block union; `tools/exp01b_levers.py` measures what each candidate
rule would recover.

**Result:** 831,258 of 7,638,365 GT pairs (10.88%) are never generated — the 89.12%
uncapped union, reconfirmed. The misses are not exotic. **Only 0.20% (1,660 pairs) share
no token at all.** The other 99.8% are visible to the data and invisible to the rules.

### The single decisive number

The DF of the rarest *shared* token, over all 831k misses:

| rarest shared token DF | name field | addr field |
|---|---|---|
| none shared | 26.9% | 23.8% |
| **<= 200** | **0.0%** | **0.0%** |
| 201 – 1k | 9.3% | 5.4% |
| 1k – 5k | 11.1% | 8.3% |
| 5k – 20k | 18.4% | 23.9% |
| 20k – 100k | 33.6% | 23.2% |
| > 100k | 0.7% | 15.5% |

Not one miss has a shared token below df=200 — mechanically guaranteed, since B5/B6
would have caught it, but it confirms the caps are *exactly* the binding constraint and
nothing else is. The mass sits at **5k–100k**, which is where a generated-from-a-word-pool
corpus puts everything.

Cumulative recovery from widening the single-token cap alone:

| single-token cap | misses reached | +pp on GT recall |
|---|---|---|
| df <= 200 (current) | 0 | +0.00 |
| **df <= 1,000 (Exp 2)** | **121,600 (14.6%)** | **+1.59** |
| df <= 5,000 | 282,504 (34.0%) | +3.70 |
| df <= 20,000 | 625,710 (75.3%) | +8.19 |
| df <= 100,000 | 789,631 (95.0%) | +10.34 |

**This prices Exp 2 before running it: +1.59pp is its ceiling**, for roughly 4x the pair
volume. Noted, still run, because the pair count and the F_0.5 effect were asked for.

### What the records actually look like

50 misses printed in full (`logs/exp01_misses.log`). The generator's noise operators are
few and repetitive:

1. **Every token is common.** `Surgical Metro Physicians` / `Surgical Metro  Physicians`
   at `5441 W Mount Drive, Nashville, NC` / `5441 1/2 WEST MOUNT DR, NASHVILLE, NC`.
   Normalized name strings are *identical*; shared address tokens are
   `{drive, mount, nashville, nc}`. Nothing is rare, so nothing blocks. This is the
   modal miss.
2. **The address is deleted outright** — `business_address` is NULL on the S2/S3 side
   in 8 of the 50. Address-first blocking has nothing to work with and the name is
   template words. Only a name-set rule can reach these.
3. **A typo lands on the one distinctive token** — `saffira`/`saffria`,
   `creston`/`crestton`, `ganguly`/`ganeuly`, `walpole`/`wallpole`,
   `security`/`secauity`, `upland`/`upand`, `96th`/`96nd`. The rare token is precisely
   the one the corruption eats.
4. **House numbers are zero-padded or ranged**: `5110`/`005110`, `7`/`07`,
   `899`/`00899`, `1223`/`1223-1225`. String equality fails; the number is right there.
5. **Address truncated to city + state** (common on Indic rows): `523, Tower B-4, Spaze
   I-Tech Park, Sector-49, Sohna Road, Gurgaon, Haryana` -> `523, GURGAON, Haryana`.
6. **Indic transliteration is phonetic and one-way**: `सूर्य`->`sury` vs S1 `surya`,
   `फ्यूचर`->`phyuchar` vs `future`, `ईस्टर्न`->`istarn` vs `eastern`. The name field is
   unusable for these pairs no matter the cap — the address carries them or nothing does.
7. **Name mangled into a domain**: `Pacific Foundation, LLC` -> `PACIFICFOUNDATION.COM`,
   `Peridial Abony LLC` -> `peridialabony.com`. Tokens are *concatenated*, so token
   equality fails entirely while a character n-gram would match.
8. **Junk affixes**: `-- Prime Téchnologies`, `Smt Indian Constructions`,
   `Dermatology Partners [Có]`, `##1401 96ND ST`.

### Two things worth fixing regardless of any block change

- **`normalize.py` is classifying house numbers as postcodes.** `00899 JEFFERSON RD`
  yields `postcode='00899'`, `1401 96th Street` yields `postcode='30404'` on its
  counterpart. B1 is joining on house numbers. It is not currently *costing* recall —
  postcode equality holds for 0 of the 831k misses — but B1's 7.2M pairs are partly
  spurious and the feature `a_pcmatch` is partly noise.
- **Street numbers are compared raw.** 51.97% of misses share a street number already;
  zero-stripping and range-splitting should lift that. Quantified in Exp 1b.

### Verdict

KEEP (diagnosis). **It reorders the backlog.** The finding is that rare-token blocking is
the wrong instrument for this corpus: names and addresses are drawn from small pools, so
individual tokens are never rare and *the combination* is what identifies a record.
Widening a single-token DF cap chases 1.59pp. The three rules the data points at are

- **a product-DF pair rule** — admit a shared token pair when `df1 * df2` is small
  rather than when each df is small. The current `df <= 5,000` on each side forbids a
  20k x 20k pair whose expected intersection is 39 records, while admitting 5k x 5k
  whose expected intersection is 2.4. The budget should be on the product.
- **an exact token-set key** on name and on address — free, one equi-join, and the only
  rule that reaches the deleted-address rows.
- **normalized street numbers** — leading zeros stripped, ranges split.

Quantified in Exp 1b below; those become Exp 7-9.

**Cost:** 487s candidate regeneration (union phase is 95% of it), 232s miss analysis.

**Notes:** two pre-existing bugs surfaced. `blocking.py` could not report per-block
recall at all (Exp 0). And `PARQUET_DIR` defaults to `data/parquet`, the 50k-row
**sample** — the first run silently measured recall against 172,727 sample GT pairs
instead of 7,638,365. It agreed to within 0.03pp (84.98% capped, 89.21% uncapped), which
is reassuring about the sample but means every run in this log must set
`PARQUET_DIR=data/parquet_full` explicitly. The baseline numbers in `NOTES.md` were
produced with it set.

### Exp 1b — pricing each lever against the 831,258 misses

`tools/exp01b_levers.py`. Recovery is an upper bound: it says the rule *can* see the
pair, not what pair volume comes with it.

| lever | misses reached | +pp GT recall |
|---|---|---|
| L1 identical **name** token set (exact key) | 313,380 (37.7%) | **+4.10** |
| L2 identical **addr** token set (exact key) | 81,109 (9.8%) | +1.06 |
| L3 L1 or L2 | 371,282 (44.7%) | +4.86 |
| L4 >= 2 shared tokens, **any** DF, combined field | 780,443 (93.9%) | +10.22 |
| L5 >= 3 shared tokens, any DF | 622,650 (74.9%) | +8.15 |
| L6 street number match, zeros stripped + ranges split | 495,545 (59.6%) | +6.49 |
| L6b — of which the zero/range fix alone adds | 63,570 (7.6%) | +0.83 |
| L7 normalized number **and** >= 1 shared token | 494,317 (59.5%) | +6.47 |

**Product-DF pair rule** — admit a shared token pair on `df1 * df2 <= B`, i.e. on the
*expected* intersection size `df1*df2/N`, instead of capping each df separately:

| budget | misses reached | +pp GT recall |
|---|---|---|
| N * 5 (expect <= 5 records/key) | 145,240 (17.5%) | +1.90 |
| N * 20 | 313,350 (37.7%) | +4.10 |
| **N * 50** | **453,273 (54.5%)** | **+5.93** |
| N * 200 | 624,817 (75.2%) | +8.18 |
| N * 1,000 | 712,444 (85.7%) | +9.33 |
| N * 5,000 | 751,444 (90.4%) | +9.84 |

A triple rule is strictly worse per unit of budget (`df1*df2*df3 <= 50*N^2` reaches only
70.4%, below the pair rule at N*200), so pairs are the right arity.

**Combined package** — `product-DF pair at N*50` OR `name-set key` OR `addr-set key` OR
`normalized number + >=1 shared token`:

```
745,746 of 831,258 misses reachable (89.7%)   ->   +9.76pp
implied uncapped union recall  98.88%   (from 89.12%)
```

**The residue is 83,668 pairs, and 66.9% of them have an empty S2/S3 address.** Nothing
address-based can ever reach those; they are a name-only problem, and the name is
template words plus a typo.

This is the plan for Exp 7-9. Ranked by recall per unit of engineering:

1. **B8/B9 exact token-set keys** — +4.86pp, one equi-join each, essentially free, and
   the only rule that reaches deleted-address rows.
2. **B3/B4 -> product-DF budget** — +5.93pp at N*50, replacing a per-token cap that is
   simultaneously too tight for mid-DF pairs and too loose for rare-x-common ones.
3. **Street-number normalization** — +0.83pp for a regex, and it makes B2 and L7 work.

---

## Measurement protocol for Exp 2 onward

Regenerating candidates over all 2.2M S1 records costs 487s, and 95% of that is the
union group-by. Blocking is **exactly per-S1 independent** — the DF tables are built from
S2+S3 only, and ranking and capping partition by S1 — so restricting the S1 side to the
entities actually evaluated gives byte-identical candidates for them. Added
`blocking.py --s1-ids`, and `tools/make_eval_ids.py` writes the set:

```
val fold (440,875)  U  train fold sampled at 0.10 (176,194)  =  617,069 S1 ids (28.0%)
```

Verified against the full run — the subset reproduces every published number:

| | full S1 (2.2M) | eval subset (617k) |
|---|---|---|
| uncapped union recall | 89.12% | 89.15% |
| recall @ cap 30 (pre-rerank) | 84.98% | 84.97% |
| recall @ cap 30 (jaccard-idf rerank) | 86.85% | 86.87% |
| B7 recall alone | 64.80% | 64.80% |
| candidate stage wall clock | 487s | **92s** |

Every per-block recall agrees to within 0.1pp. The whole pipeline is now **~23 min**
(cand 92s, rerank 125s, feat-train 236s, feat-val 486s, train 322s, sweep 129s), which
fits the 45-minute box with room for a slower blocking stage.

### Baseline re-anchored under this protocol

| | NOTES.md baseline | this protocol |
|---|---|---|
| train pairs | 5,016,119 | 4,267,885 |
| CV AUC | 0.998527 | 0.998501 |
| **val macro-F_0.5** | **0.847264** | **0.847028** (tau=0.20, delta=0.35) |

The −0.00024 is the smaller training sample: the original run used `--s1-frac` ~0.115,
this protocol fixes it at 0.10. **0.847028 is the number every experiment below is
compared against.** Cross-protocol comparison against 0.847264 would be invalid.

---

## Exp 2 — widen B5/B6 single-token blocks from df<=200 to df<=1,000

**Hypothesis:** the single-token blocks are the tightest cap in the set, so relaxing
them recovers pairs no other block can see.

**Change:** `DF_CAP_SINGLE` 200 -> 1,000. One variable; everything else at baseline.

**Result:**

| | baseline | Exp 2 | delta |
|---|---|---|---|
| B5_addr_single pairs | 20,331,180 | 190,242,772 | **9.4x** |
| B6_name_single pairs | 19,181,226 | 82,930,639 | 4.3x |
| union pairs (uncapped) | 104,745,584 | 322,610,823 | 3.08x |
| **uncapped union recall** | 89.15% | **90.74%** | **+1.59pp** |
| recall @30, blocking-IDF order | 84.97% | 85.06% | +0.09pp |
| **recall @30, after jaccard re-rank** | **86.87%** | **86.81%** | **−0.06pp** |
| recall @100, after re-rank | 87.38% | 87.39% | +0.01pp |
| S1 with zero candidates | 2.2% | 2.0% | −0.2pp |
| CV AUC | 0.998501 | **0.998670** | +0.000169 |
| **val macro-F_0.5** | **0.847028** | **0.846530** | **−0.000498** |
| candidate stage wall clock | 92s | 330s | 3.6x |

**Verdict: REVERT.**

**Notes — this is the most useful result so far, and it is a negative one.**

1. **Exp 1 priced it exactly.** The predicted ceiling was +1.59pp on uncapped union
   recall. Measured: +1.59pp. The miss diagnostic is trustworthy as a planning
   instrument, which means the Exp 1b numbers for the other levers can be trusted too.

2. **The extra recall does not survive the cap.** +1.59pp uncapped became **−0.06pp at
   cap 30 after re-ranking**. Three times the pairs, and the shortlist got slightly
   *worse*. The newly generated true pairs are, by construction, pairs whose only shared
   token is common — so their IDF-weighted Jaccard is low and the re-ranker puts them
   below 30. Meanwhile the ~218M extra false pairs give the ranker more chances to make
   a mistake in the top 30.

3. **Generation is no longer the binding constraint at cap 30 — ranking is.** The
   uncapped union already holds 89.15% and the pipeline delivers 86.87% of it; the gap
   between what is generated and what survives to the model is now 2.3pp and grows as
   generation widens. Adding blocks without a ranker that can surface their output is
   not just wasted work, it is negative.

4. **CV AUC went up while F_0.5 went down** (0.998501 -> 0.998670, val
   0.847028 -> 0.846530). The new pairs are mostly *easy* negatives — they share one
   common token and nothing else — so they inflate AUC by padding the trivially-separable
   mass while the hard decision at the top of each shortlist gets no better. Exactly the
   reason CLAUDE.md says to tune against macro-F_0.5 and never against AUC. Worth
   remembering as evidence, not just as a rule.

**Cost:** 29 min end to end (cand 330s, rerank 162s, feat 791s, train 426s, sweep ~130s).

**Reordering consequence:** the Exp 1b blocks (B8/B9 set keys, B10 product-DF pairs) are
built and ready but must not be measured at cap 30 alone — they would likely fail the
same way. Exp 3, the cap sweep, now decides whether there is headroom to exploit at all,
and it runs next rather than after.

---

## Exp 3 — re-rank cap sweep

**Hypothesis:** cap 30 was chosen for cost, not measured against F_0.5. If recall is the
binding constraint, a larger cap should pay.

**Change:** `--cap` only. Blocks and df caps at baseline. Full pipeline re-run per cap,
so the model is fitted on the shortlist it is scored on (filtering a cap-100 feature set
down to 30 would leave the context features — `c_rank`, `c_ratio_mean`, `c_shortlist` —
computed over the wrong shortlist).

**Result:**

| cap | blocking recall @cap | CV AUC | **val macro-F_0.5** | best tau/delta | feature pairs (val) |
|---|---|---|---|---|---|
| 20 | 86.42% | 0.99787 | 0.845706 | 0.20 / 0.35 | 8.0M |
| **30** | **86.87%** | **0.998501** | **0.847028** | **0.20 / 0.35** | **11.9M** |
| 50 | 87.19% | 0.99899 | 0.846777 | 0.30 / 0.35 | 19.6M |

Caps 75 and 100 were **not run**. The recall curve after re-ranking is
`@30 86.87 / @50 87.19 / @75 87.32 / @100 87.38` — 50 already costs 1.6x the features
for +0.32pp recall and *lost* F_0.5, and 75 and 100 add only +0.13pp and +0.19pp more.
Two points either side of 30 both being worse makes 30 an interior optimum, and the
remaining recall on the table (0.5pp) is smaller than the loss already measured going
from 30 to 50. Spending 60+ minutes to confirm a third and fourth declining point was
not worth the slot.

**Verdict: KEEP cap 30** — unchanged from baseline. The default was already right.

**Notes:** the sweep is an interior optimum, not a plateau, and the two failure modes
are different at each end.

- **Cap 20 loses recall faster than it gains precision.** −0.45pp recall costs
  −0.0013 F_0.5. Even at 2x precision weighting, dropping a true match hurts more than
  the false ones it avoids, because the mean entity has 3.46 matches — trimming the
  shortlist trims real answers, not just noise.
- **Cap 50 gains recall that the decision layer cannot convert.** +0.32pp recall,
  and F_0.5 still falls. `tau` has to rise from 0.20 to 0.30 to hold the extra 20
  candidates per entity back, and it cannot do it selectively.
- **CV AUC rises monotonically with the cap — 0.99787 -> 0.998501 -> 0.99899 — while
  F_0.5 peaks in the middle.** The second independent confirmation in this log that AUC
  is anti-correlated with the objective here. A bigger shortlist adds easy negatives,
  which is free AUC and negative F_0.5.

**Cost:** 32 min (cap 50) + 26 min (cap 20). Cap 30 was already measured as the baseline.

---

## Exp 7 — exact token-set key blocks (B8 name-set, B9 addr-set)

**Hypothesis (from Exp 1b, L1+L2 = +4.86pp).** The modal miss is a pair whose token sets
are *identical* but composed entirely of common words, so no rare-token rule can see it.
Hashing the whole sorted token set into one key makes the *combination* the blocking key,
which is exactly what identifies a record in a corpus generated from small word pools.

**Critically, this should behave differently from Exp 2.** Exp 2's extra pairs shared only
one common token, so their IDF-weighted Jaccard was low and the re-ranker buried them
below the cap. A set-key pair has an identical token set on one whole field, so its
Jaccard is high and it ranks near the *top*. The Exp 2 failure mode predicts this one
succeeds.

**Change:** `ENABLE_SETKEY=1`. Two new blocks, key df <= 1,000, minimum 2 tokens per key.
Everything else at baseline (cap 30, df caps unchanged).

**Result:**

| | baseline | Exp 7 | delta |
|---|---|---|---|
| B8_name_set | — | 22,257,647 pairs, **50.39%** recall alone | new |
| B9_addr_set | — | 3,594,088 pairs, **23.90%** recall alone | new |
| union pairs (uncapped) | 104,745,584 | 127,782,287 | **+22% only** |
| **uncapped union recall** | 89.15% | **93.86%** | **+4.71pp** |
| recall @30, blocking-IDF order | 84.97% | 88.68% | +3.71pp |
| **recall @30, after jaccard re-rank** | **86.87%** | **90.46%** | **+3.59pp** |
| recall @10, after re-rank | 84.93% | 87.68% | +2.75pp |
| **S1 with zero candidates** | **13,638 (2.2%)** | **231 (0.0%)** | **−13,407** |
| reduction ratio | 0.9999976550 | 0.9999974091 | — |
| CV AUC | 0.998501 | 0.998262 | −0.000239 |
| **val macro-F_0.5** | **0.847028** | **0.866908** | **+0.019880** |
| best tau / delta | 0.20 / 0.35 | 0.25 / 0.35 | — |
| one-to-one worth | +0.001730 | +0.002205 | — |
| candidate stage wall clock | 92s | 116s | +26% |

Per country, capped: US 87.23% -> **91.06%**, India 81.65% -> **85.13%**.
Per script, capped: Latin 86.27% -> **90.33%**, Devanagari 66.05% -> 66.03% (unchanged —
transliterated Indic names do not produce identical token sets, as Exp 1 predicted).

**Verdict: KEEP.** New best: **val macro-F_0.5 = 0.866908**.

**Notes.**

1. **Exp 1b's prediction held again.** Predicted ceiling +4.86pp uncapped; measured
   +4.71pp (97% of it). Two for two on the miss diagnostic as a planning instrument.

2. **This recall survives the cap, and that was predictable.** +4.71pp uncapped became
   **+3.59pp at cap 30** — 76% conversion, against Exp 2's *negative* conversion. The
   difference is entirely about where the re-ranker puts the new pairs, which is why the
   Exp 2 failure was worth understanding rather than just reverting.

3. **The zero-candidate population essentially disappears: 13,638 -> 231.** These are the
   rows whose S2/S3 address was deleted outright (66.9% of the Exp 1 residue). Nothing
   address-based could ever reach them and the name was template words; the name-set key
   reaches them in one equi-join. Each such entity was contributing 0 to a macro-average,
   so this alone is worth roughly 13,400/440,875 = 3.0pp of macro-F_0.5 headroom.

4. **CV AUC went DOWN (0.998501 -> 0.998262) while F_0.5 went UP by 0.0199.** The third
   and cleanest confirmation of the AUC/F_0.5 divergence in this log, and this time in the
   opposite direction from Exp 2/3: the new pairs are *hard* — near-duplicate names at
   different addresses — so they make the classifier's job harder while making the
   shortlist far more likely to contain the right answer. **Anyone tuning this pipeline on
   AUC would have rejected the single best change of the night.**

5. **It is nearly free.** +22% pairs, +26% candidate-stage runtime, two extra equi-joins.

**Cost:** 32 min end to end (cand 131s, rerank 140s, feat 1,149s, train 349s, sweep 146s).

---

# Status at hand-off

## Scoreboard (val macro-F_0.5, clean disjoint val fold, 440,875 S1 entities)

| exp | change | blocking recall @30 | val macro-F_0.5 | verdict |
|---|---|---|---|---|
| 00 | baseline, re-anchored under this protocol | 86.87% | 0.847028 | reference |
| 1 | diagnosis only | — | — | KEEP (reordered the backlog) |
| 2 | B5/B6 df<=200 -> 1,000 | 86.81% | 0.846530 | **REVERT** |
| 3 | cap 20 | 86.42% | 0.845706 | **REVERT** |
| 3 | cap 50 | 87.19% | 0.846777 | **REVERT** |
| 3 | cap 30 (unchanged) | 86.87% | 0.847028 | **KEEP** |
| **7** | **B8/B9 exact token-set keys** | **90.46%** | **0.866908** | **KEEP — new best** |

Net gain for the run: **+0.019880 val macro-F_0.5**, all of it from Exp 7.
Original leaderboard 0.8397 at val 0.847264; val has tracked the leaderboard within
0.008, so Exp 7 projects to roughly **0.859 +/- 0.008** on the leaderboard.

## The one idea that mattered

Rare-token blocking is the wrong instrument for this corpus. Names and addresses are
generated from small word pools, so **no single token is ever rare — the combination is
what identifies a record.** Widening a rarity cap chases 1.6pp and loses at the cap
(Exp 2). Making the whole token set the key gains 4.7pp and keeps 3.6pp of it (Exp 7).

Corollary worth carrying: **AUC is anti-correlated with the objective here.** Exp 2 and
Exp 3 both raised CV AUC while lowering F_0.5; Exp 7 *lowered* AUC by 0.00024 and raised
F_0.5 by 0.0199. Every threshold and every block must be judged on macro-F_0.5.

## NOT DONE — next actions, highest value first

1. **Regenerate the submission for Exp 7 (the only blocking item).** `output/` still
   holds the exp00 baseline TSVs — valid, validated, leaderboard 0.8397, so the repo is
   not broken, but the 0.8669 config has never been run test-side. Exact commands in
   `best/exp07_setkey_0.8669/CONFIG.md`. ~1.5-2 h, dominated by test feature generation
   over ~45M pairs. **Do this before anything else** — it is worth ~+0.019 on the board.

2. **Exp 8 — B10 product-DF pair block.** Built, tested for syntax, `ENABLE_PRODPAIR=1`,
   never run. Exp 1b prices it at **+5.93pp** uncapped at budget `df1*df2 <= 50N`
   (`PROD_BUDGET=50`, `K_PROD=8`). It admits a shared token pair on its *expected*
   intersection `df1*df2/N` instead of capping each df — the current rule forbids a
   20k x 20k pair (expected intersection 39 records) while admitting 5k x 5k (expected
   2.4). Risk: these pairs share two mid-DF tokens, so their Jaccard is middling and they
   may be buried like Exp 2's rather than surfaced like Exp 7's. Measure at cap 30 and,
   if recall rises but F_0.5 does not, that is the ranker's fault, not the block's.

3. **Exp 9 — street-number normalization.** Built, `ENABLE_NUMNORM=1`, never run.
   Exp 1b: **+0.83pp** for a regex (zeros stripped, ranges split). `5110`/`005110`,
   `7`/`07`, `899`/`00899`, `1223`/`1223-1225` currently fail string equality. Note it
   only rewrites numbers inside `blocking.py`; if it wins, push it into `normalize.py` so
   the `a_numov` / `a_numfirst` features see it too.

4. **Re-run the cap sweep (Exp 3) on top of Exp 7.** Cap 30 was optimal at 86.87% recall;
   at 90.46% with a much better-ordered shortlist (@10 is now 87.68%, above the old @30)
   the optimum may have moved *down*. Cheap and likely worth a little.

5. **Backlog items not started:** Exp 4 (train on 20M pairs — low expected value, Exp 2/3
   showed the classifier is not the constraint and its 10% train sample yields only 4.3M
   pairs, so this needs `--s1-frac` ~0.5 first), Exp 5 (per-country tau/delta with a
   leave-one-country-out check for France), Exp 6 (delta scaled by shortlist size).
   Exp 5 and 6 touch only `decide.py`, so each is a ~3-minute sweep once the current
   best's features and model are on disk — which they are right now, for Exp 7.

6. **Two `normalize.py` bugs found in Exp 1, not yet fixed.** House numbers are being
   classified as postcodes (`00899 JEFFERSON RD` -> `postcode='00899'`), so B1 joins on
   house numbers and `a_pcmatch` is partly noise. Not currently costing recall —
   postcode equality holds for 0 of the 831k misses — but it is wrong.

## Repo state

- `output/` — exp00 baseline, valid, passes `utils/validate_submission.py`.
- `data/model/`, `data/features/`, `data/cand_exp/` — **Exp 7** artifacts (the new best).
  So the model on disk is *better* than the submission on disk; item 1 closes that gap.
- `data/cand/` — full-S1 baseline candidates + re-ranked set, kept for a baseline
  test-side run if ever needed.
- `best/exp00_baseline_0.8397/` — untouched. `best/exp07_setkey_0.8669/` — src + config
  + logs, TSVs pending per item 1.
- Tags: `exp00-0.8397`, `exp07-0.8669`.
- `tools/` — `run_exp.sh` (one-command experiment), `make_eval_ids.py`,
  `exp01_misses.py`, `exp01b_levers.py`. `logs/` has every run's raw output.
