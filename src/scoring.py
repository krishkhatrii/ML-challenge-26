"""Stage 3 — the metric and the train/val split. Written before any model code.

macro-F_0.5, exactly as the challenge specifies: computed per Source-1 entity, then
averaged over **all** S1 entities, singletons included.

    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

Substituting P = tp/|pred| and R = tp/|truth| cancels the products:

    F_0.5 = 1.25 * tp / (0.25 * |truth| + |pred|)

Both forms are implemented and `--selftest` asserts they agree on random cases. The
closed form is what runs on 2.2M rows: no division by zero when tp > 0, no per-entity
Python loop. The one case it cannot express is |truth| = |pred| = 0, which is scored
1.0 by rule (a correctly predicted singleton).

Usage:
    python src/scoring.py                          # self-tests (default)
    python src/scoring.py --split                  # train/val split sizes
    python src/scoring.py --pred output/matching_results.tsv
    python src/scoring.py --pred p.tsv --truth-parquet data/parquet/train_ground_truth.parquet

Env:
    PARQUET_DIR   where the Parquet lives          (default: data/parquet)
    VAL_FRAC      validation fraction              (default: 0.2)
    SPLIT_SEED    split seed                        (default: 13)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent

TRUTH_ID = "source1_entity_id"
TRUTH_IDS = "matched_entity_ids"
PRED_IDS = "matched_entity_ids"
CAND_IDS = "candidate_entity_ids"

DEFAULT_VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.2"))
DEFAULT_SPLIT_SEED = int(os.environ.get("SPLIT_SEED", "13"))


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


# ------------------------------------------------------------------- the metric


def f05(pred, truth) -> float:
    """macro-F_0.5 contribution of a single S1 entity. Reference implementation.

    Follows the specified formula literally via precision and recall, so it can be
    diffed against the vectorised closed form.
    """
    p, t = set(pred), set(truth)
    if not t and not p:
        return 1.0  # correctly predicted singleton — worth a full 1.0
    if not t or not p:
        return 0.0  # one side empty and the other not
    tp = len(p & t)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(p), tp / len(t)
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def f05_closed(tp: int, n_truth: int, n_pred: int) -> float:
    """The algebraically identical closed form used by the vectorised path."""
    if n_truth == 0 and n_pred == 0:
        return 1.0
    denom = 0.25 * n_truth + n_pred
    return 0.0 if denom == 0 else 1.25 * tp / denom


def id_list_expr(col: str) -> pl.Expr:
    """Comma-separated ID string -> deduplicated list. Empty/NULL -> empty list.

    Deduplication is not cosmetic: a repeated ID would inflate |pred| and depress
    precision. The submission rules forbid duplicates, so scoring them as an error
    would double-punish something the validator already catches.
    """
    return (
        pl.when(pl.col(col).is_null() | (pl.col(col).str.strip_chars() == ""))
        .then(pl.lit([], dtype=pl.List(pl.String)))
        .otherwise(
            pl.col(col)
            .str.split(",")
            .list.eval(pl.element().str.strip_chars().filter(pl.element().str.strip_chars() != ""))
            .list.unique()
        )
        .alias(col)
    )


def macro_f05(truth: pl.DataFrame, pred: pl.DataFrame, truth_col: str, pred_col: str) -> dict:
    """macro-F_0.5 over every S1 entity in `truth`.

    `truth` defines the universe. An S1 entity absent from `pred` scores as an empty
    prediction — silently dropping it would quietly inflate the score, which is the
    single easiest way to fool yourself in this challenge.
    """
    t = truth.select(pl.col(TRUTH_ID), id_list_expr(truth_col).alias("t"))
    p = pred.select(pl.col(TRUTH_ID), id_list_expr(pred_col).alias("p"))

    extra = p.join(t, on=TRUTH_ID, how="anti").height
    joined = t.join(p, on=TRUTH_ID, how="left").with_columns(
        pl.col("p").fill_null(pl.lit([], dtype=pl.List(pl.String)))
    )

    scored = joined.with_columns(
        n_t=pl.col("t").list.len().cast(pl.Int64),
        n_p=pl.col("p").list.len().cast(pl.Int64),
        tp=pl.col("t").list.set_intersection(pl.col("p")).list.len().cast(pl.Int64),
    ).with_columns(
        f=pl.when((pl.col("n_t") == 0) & (pl.col("n_p") == 0))
        .then(pl.lit(1.0))
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_t") + pl.col("n_p")))
        .fill_nan(0.0)
    )

    agg = scored.select(
        macro_f05=pl.col("f").mean(),
        entities=pl.len(),
        singletons=(pl.col("n_t") == 0).sum(),
        singletons_correct=((pl.col("n_t") == 0) & (pl.col("n_p") == 0)).sum(),
        predicted_empty=(pl.col("n_p") == 0).sum(),
        tp=pl.col("tp").sum(),
        total_pred=pl.col("n_p").sum(),
        total_truth=pl.col("n_t").sum(),
    ).to_dicts()[0]

    agg["missing_from_pred"] = int(joined.height - p.join(t, on=TRUTH_ID, how="semi").height)
    agg["pred_not_in_truth"] = extra
    agg["micro_precision"] = agg["tp"] / agg["total_pred"] if agg["total_pred"] else 0.0
    agg["micro_recall"] = agg["tp"] / agg["total_truth"] if agg["total_truth"] else 0.0
    agg["per_entity"] = scored
    return agg


# -------------------------------------------------------------------- the split


def fold_expr(col: str, val_frac: float, seed: int) -> pl.Expr:
    """Deterministic train/val assignment, hashed from the S1 entity ID.

    Hash-based rather than shuffled: stable under row reordering and under the data
    growing, reproducible across runs and machines, and computable independently by any
    later stage without persisting a split file. Because one ground-truth row *is* one
    S1 entity, hashing the S1 ID is exactly the grouped split the plan requires — every
    pair generated for an entity lands in the same fold by construction.

    Depends on Polars' seeded hash, so `requirements.txt` pins Polars.
    """
    frac = (pl.col(col).hash(seed=seed) % 1_000_000) / 1_000_000
    return pl.when(frac < val_frac).then(pl.lit("val")).otherwise(pl.lit("train")).alias("fold")


def assign_fold(
    df: pl.DataFrame,
    col: str = TRUTH_ID,
    val_frac: float = DEFAULT_VAL_FRAC,
    seed: int = DEFAULT_SPLIT_SEED,
) -> pl.DataFrame:
    return df.with_columns(fold_expr(col, val_frac, seed))


def split_report(gt: pl.DataFrame, val_frac: float, seed: int) -> None:
    df = assign_fold(gt, val_frac=val_frac, seed=seed).with_columns(
        n_t=id_list_expr(TRUTH_IDS).list.len()
    )
    total = df.height
    print(f"S1-entity-grouped split   val_frac={val_frac}  seed={seed}")
    print()
    rows = (
        df.group_by("fold")
        .agg(
            entities=pl.len(),
            singletons=(pl.col("n_t") == 0).sum(),
            matched_ids=pl.col("n_t").sum(),
            mean_matches=pl.col("n_t").mean(),
        )
        .sort("fold")
    )
    print(f"  {'fold':<7}{'entities':>12}{'share':>9}{'singletons':>12}{'singleton %':>13}{'matched IDs':>14}{'mean':>7}")
    print("  " + "-" * 74)
    for r in rows.iter_rows(named=True):
        print(
            f"  {r['fold']:<7}{r['entities']:>12,}{100 * r['entities'] / total:>8.2f}%"
            f"{r['singletons']:>12,}{100 * r['singletons'] / r['entities']:>12.2f}%"
            f"{r['matched_ids']:>14,}{r['mean_matches']:>7.2f}"
        )
    # An entity straddling folds would silently leak labels between train and val.
    overlap = (
        df.filter(pl.col("fold") == "train")
        .join(df.filter(pl.col("fold") == "val"), on=TRUTH_ID, how="semi")
        .height
    )
    print()
    print(f"  entities in both folds : {overlap}  (must be 0)")
    assert overlap == 0, "split leaked an S1 entity across folds"


# --------------------------------------------------------------------- selftest


def selftest() -> int:
    import random

    print("=" * 78)
    print("SELFTEST — macro-F_0.5")
    print("=" * 78)
    print()

    cases = [
        ("README worked example", ["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"], 0.714),
        ("perfect match", ["S2-1", "S3-2"], ["S2-1", "S3-2"], 1.0),
        ("correct singleton", [], [], 1.0),
        ("false merge on singleton", ["S2-1"], [], 0.0),
        ("missed everything", [], ["S2-1"], 0.0),
        ("no overlap", ["S2-9"], ["S2-1"], 0.0),
        ("duplicates deduped", ["S2-1", "S2-1"], ["S2-1"], 1.0),
    ]
    print(f"  {'case':<28}{'pred':>6}{'truth':>7}{'F_0.5':>9}{'expected':>10}  ok")
    print("  " + "-" * 68)
    ok = True
    for label, p, t, expected in cases:
        got = f05(p, t)
        good = abs(got - expected) < 5e-4
        ok &= good
        print(f"  {label:<28}{len(set(p)):>6}{len(set(t)):>7}{got:>9.4f}{expected:>10.3f}  {'OK' if good else 'FAIL'}")

    # The README example spelled out step by step, to prove the formula not just the number.
    tp, n_p, n_t = 2, 3, 2
    prec, rec = tp / n_p, tp / n_t
    manual = (1.25 * prec * rec) / (0.25 * prec + rec)
    print()
    print(f"  README arithmetic: P={prec:.4f} R={rec:.1f} -> F_0.5={manual:.6f} (README says 0.714)")
    assert abs(manual - 0.714285714) < 1e-6

    # Reference vs closed form on random cases.
    random.seed(7)
    ids = [f"S2-{i}" for i in range(12)]
    worst = 0.0
    for _ in range(4000):
        t = set(random.sample(ids, random.randint(0, 6)))
        p = set(random.sample(ids, random.randint(0, 6)))
        worst = max(worst, abs(f05(p, t) - f05_closed(len(p & t), len(t), len(p))))
    print(f"  reference vs closed form: max abs diff over 4000 random cases = {worst:.2e}")
    assert worst < 1e-12
    ok &= worst < 1e-12

    # Precision must outweigh recall 2x.
    half_prec = f05(["S2-1", "S2-2"], ["S2-1"])          # P=0.5 R=1.0
    half_rec = f05(["S2-1"], ["S2-1", "S2-2"])           # P=1.0 R=0.5
    print(f"  precision-heavy check: P=0.5,R=1.0 -> {half_prec:.4f}   P=1.0,R=0.5 -> {half_rec:.4f}")
    assert half_rec > half_prec, "F_0.5 must reward precision over recall"
    ok &= half_rec > half_prec

    # Vectorised path must agree with the reference on a fixture, including an
    # entity present in truth but absent from predictions.
    truth = pl.DataFrame({
        TRUTH_ID: ["S1-a", "S1-b", "S1-c", "S1-d", "S1-e"],
        TRUTH_IDS: ["S2-00047,S3-00812", "", "S2-1,S2-2,S3-3", "S2-9", "S2-5"],
    })
    pred = pl.DataFrame({
        TRUTH_ID: ["S1-a", "S1-b", "S1-c", "S1-d"],
        PRED_IDS: ["S2-00047,S2-00193,S3-00812", "", "S2-1,S2-2", "", ],
    })
    res = macro_f05(truth, pred, TRUTH_IDS, PRED_IDS)
    manual_scores = [
        f05(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"]),
        f05([], []),
        f05(["S2-1", "S2-2"], ["S2-1", "S2-2", "S3-3"]),
        f05([], ["S2-9"]),
        f05([], ["S2-5"]),  # S1-e missing from pred entirely
    ]
    expect = sum(manual_scores) / len(manual_scores)
    print()
    print(f"  vectorised macro = {res['macro_f05']:.6f}   reference macro = {expect:.6f}")
    print(f"  entities={res['entities']}  missing_from_pred={res['missing_from_pred']}  (S1-e)")
    assert abs(res["macro_f05"] - expect) < 1e-12
    assert res["missing_from_pred"] == 1
    ok &= abs(res["macro_f05"] - expect) < 1e-12

    # Split: reproducible, disjoint, roughly the requested size.
    gt = pl.DataFrame({
        TRUTH_ID: [f"S1-{i}" for i in range(20_000)],
        TRUTH_IDS: ["S2-1"] * 20_000,
    })
    a = assign_fold(gt)["fold"].to_list()
    b = assign_fold(gt)["fold"].to_list()
    val_share = a.count("val") / len(a)
    print()
    print(f"  split reproducible across calls: {a == b}")
    print(f"  val share = {val_share:.4f} (target {DEFAULT_VAL_FRAC})")
    assert a == b
    assert abs(val_share - DEFAULT_VAL_FRAC) < 0.01
    ok &= a == b

    print()
    print("=" * 78)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    print("=" * 78)
    return 0 if ok else 1


# ------------------------------------------------------------------------- main


def read_id_file(path: Path, col: str) -> pl.DataFrame:
    """Read a submission-format TSV: unquoted, tab-separated, comma-separated ID lists.

    Read with quote_char=None to match the required output format exactly, then strip
    any wrapping double quotes defensively. Polars' own `write_csv` emits `""` for an
    empty string, so a submission generated with default settings would arrive with
    every singleton row holding a literal two-character `""` ID -- scoring near 0 with
    no visible cause. Stage 9 must write with `quote_style="never"`; this catches it if
    it does not.
    """
    df = pl.read_csv(path, separator="\t", quote_char=None, schema_overrides={TRUTH_ID: pl.String, col: pl.String})
    for need in (TRUTH_ID, col):
        if need not in df.columns:
            raise SystemExit(f"{path}: missing column {need!r} (found {df.columns})")
    quoted = df.select((pl.col(col).str.contains('"', literal=True)).sum()).item() or 0
    if quoted:
        print(f"  WARNING: {quoted:,} rows had quoted {col!r} values; stripping quotes.")
        print('           Write submissions with quote_style="never" -- see src/scoring.py.')
        df = df.with_columns(pl.col(col).str.replace_all('"', "", literal=True))
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="macro-F_0.5 and the S1-grouped train/val split")
    ap.add_argument("--pred", help="predictions TSV to score")
    ap.add_argument("--pred-col", default=PRED_IDS)
    ap.add_argument("--truth-parquet", default=None, help="ground-truth Parquet (default: PARQUET_DIR)")
    ap.add_argument("--fold", choices=["train", "val", "all"], default="all", help="score one fold only")
    ap.add_argument("--split", action="store_true", help="report split sizes")
    ap.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    ap.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    args = ap.parse_args()

    if not args.pred and not args.split:
        return selftest()

    pq = resolve(args.truth_parquet) if args.truth_parquet else resolve(
        os.environ.get("PARQUET_DIR", "data/parquet")
    ) / "train_ground_truth.parquet"
    if not pq.exists():
        raise SystemExit(f"{pq} not found. Run src/convert.py first.")
    gt = pl.read_parquet(pq)

    if args.split:
        print(f"ground truth: {pq}")
        print()
        split_report(gt, args.val_frac, args.seed)
        if not args.pred:
            return 0
        print()

    truth = gt
    if args.fold != "all":
        truth = assign_fold(gt, val_frac=args.val_frac, seed=args.seed).filter(
            pl.col("fold") == args.fold
        ).drop("fold")

    pred = read_id_file(resolve(args.pred), args.pred_col)
    res = macro_f05(truth, pred, TRUTH_IDS, args.pred_col)

    print(f"predictions : {args.pred}")
    print(f"ground truth: {pq}  (fold={args.fold})")
    print()
    print(f"  macro-F_0.5          {res['macro_f05']:.6f}   <-- leaderboard metric")
    print()
    print(f"  S1 entities scored   {res['entities']:,}")
    print(f"  missing from pred    {res['missing_from_pred']:,}  (scored as empty)")
    print(f"  pred rows not in GT  {res['pred_not_in_truth']:,}  (ignored)")
    print(f"  true singletons      {res['singletons']:,}  ({100 * res['singletons'] / res['entities']:.2f}%)")
    print(f"  ...correctly empty   {res['singletons_correct']:,}")
    print(f"  predicted empty      {res['predicted_empty']:,}")
    print()
    print(f"  micro precision      {res['micro_precision']:.4f}")
    print(f"  micro recall         {res['micro_recall']:.4f}")
    print(f"  matched IDs: pred {res['total_pred']:,} / truth {res['total_truth']:,} / correct {res['tp']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
