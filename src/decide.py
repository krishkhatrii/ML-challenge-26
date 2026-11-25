"""Stage 9 — the decision layer. This is where F_0.5 is actually won.

Sweeps the acceptance threshold tau and the margin delta jointly against macro-F_0.5 on
the validation split, then applies:

* **Floor rule** — if nothing clears tau, still emit the top-1. Recon measured a 5.58%
  singleton rate, so emitting nothing scores 0.0 on the 94.4% of entities that do have a
  match. A wrong guess costs no more than silence there, while a right one gains
  everything, so abstention is never the better bet.
* **Margin rule** — accept the 2nd, 3rd, ... only within delta of the best score. The
  median entity has 3 true matches, so delta is tuned to admit several, not to suppress.
* **One-to-one conflict resolution** — confirmed on the full training data: 0 violations
  across 7,638,365 GT pairs. A contested S2/S3 id goes to the highest-scoring S1 and is
  dropped elsewhere. Pure precision gain at zero recall cost.

Submissions are written with `quote_style="never"` (see docs/REPRODUCE.md, section 3.4).

Usage:
    python src/decide.py --sweep --feat-glob 'data/features/train_val_part*.parquet'
    python src/decide.py --apply --feat-glob 'data/features/test_all_part*.parquet' \
        --split test --out output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import DEFAULT_SPLIT_SEED, DEFAULT_VAL_FRAC, TRUTH_ID, TRUTH_IDS, assign_fold, macro_f05  # noqa: E402


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def score_pairs(feat_glob: str, model_dir: Path) -> pl.DataFrame:
    booster = lgb.Booster(model_file=str(model_dir / "lgbm.txt"))
    feats = json.loads((model_dir / "features.json").read_text())
    con = duckdb.connect(config={"memory_limit": "7GB", "threads": 8,
                                 "preserve_insertion_order": False})
    df = pl.from_arrow(con.execute(f"SELECT * FROM read_parquet('{feat_glob}')").arrow())
    X = df.select(feats).to_numpy().astype(np.float32)
    p = booster.predict(X)
    return df.select("s1", "s23").with_columns(p=pl.Series("p", p))


def decide(scored: pl.DataFrame, tau: float, delta: float, one_to_one: bool = True) -> pl.DataFrame:
    """Apply tau, the margin rule, the floor rule and one-to-one. Returns (s1, s23)."""
    d = scored.with_columns(best=pl.col("p").max().over("s1"))
    keep = d.filter((pl.col("p") >= tau) & (pl.col("p") >= pl.col("best") - delta))
    # Floor rule: entities where nothing cleared tau still emit their top-1.
    lost = d.join(keep.select("s1").unique(), on="s1", how="anti")
    floor = lost.filter(pl.col("p") == pl.col("best")).unique(subset=["s1"], keep="first")
    out = pl.concat([keep, floor]).select("s1", "s23", "p")
    if one_to_one:
        # Greedy: highest-scoring claim wins a contested S2/S3 id.
        out = out.sort(["p", "s1"], descending=[True, False]).unique(subset=["s23"], keep="first")
    return out.select("s1", "s23")


def to_submission(pred: pl.DataFrame, universe: pl.DataFrame) -> pl.DataFrame:
    # Keep only the id column: the ground-truth frame carries its own matched_entity_ids,
    # which would collide with the one being built here.
    universe = universe.select(TRUTH_ID)
    agg = pred.group_by("s1").agg(matched=pl.col("s23").str.join(","))
    return (universe.join(agg, left_on=TRUTH_ID, right_on="s1", how="left")
            .with_columns(pl.col("matched").fill_null(""))
            .rename({"matched": TRUTH_IDS})
            .select(TRUTH_ID, TRUTH_IDS))


def main() -> int:
    ap = argparse.ArgumentParser(description="stage 9: decision layer")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--feat-glob", required=True)
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--model", default="data/model")
    ap.add_argument("--parquet-dir", default=os.environ.get("PARQUET_DIR", "data/parquet"))
    ap.add_argument("--cand-dir", default=os.environ.get("CAND_DIR", "data/cand"))
    ap.add_argument("--out", default="output")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--cap", type=int, default=30)
    args = ap.parse_args()

    model_dir = resolve(args.model)
    glob = str(resolve(args.feat_glob)) if not args.feat_glob.startswith("/") else args.feat_glob
    scored = score_pairs(glob, model_dir)
    print(f"scored {scored.height:,} pairs over {scored['s1'].n_unique():,} S1 entities")

    if args.sweep:
        gt = pl.read_parquet(resolve(args.parquet_dir) / "train_ground_truth.parquet")
        val = assign_fold(gt).filter(pl.col("fold") == "val").drop("fold")
        print(f"val universe: {val.height:,} S1 entities\n")
        best = (None, None, -1.0)
        taus = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
        deltas = [0.05, 0.10, 0.20, 0.35, 0.50, 1.00]
        print(f"  {'tau':>6}" + "".join(f"{'d=' + str(d):>9}" for d in deltas))
        print("  " + "-" * (6 + 9 * len(deltas)))
        for tau in taus:
            row = []
            for delta in deltas:
                pred = decide(scored, tau, delta)
                sub = to_submission(pred, val)
                f = macro_f05(val, sub, TRUTH_IDS, TRUTH_IDS)["macro_f05"]
                row.append(f)
                if f > best[2]:
                    best = (tau, delta, f)
            print(f"  {tau:>6.2f}" + "".join(f"{v:>9.4f}" for v in row))
        print(f"\n  BEST tau={best[0]} delta={best[1]}  macro-F_0.5 = {best[2]:.6f}")

        # what the one-to-one rule is worth on its own
        pred_no = decide(scored, best[0], best[1], one_to_one=False)
        f_no = macro_f05(val, to_submission(pred_no, val), TRUTH_IDS, TRUTH_IDS)["macro_f05"]
        print(f"  without one-to-one resolution: {f_no:.6f}  (delta {best[2]-f_no:+.6f})")

        (model_dir / "decision.json").write_text(json.dumps(
            {"tau": best[0], "delta": best[1], "val_macro_f05": best[2],
             "val_macro_f05_no_one_to_one": f_no}, indent=2) + "\n")
        print(f"  saved {model_dir/'decision.json'}")
        return 0

    if args.apply:
        cfg = json.loads((model_dir / "decision.json").read_text())
        tau = args.tau if args.tau is not None else cfg["tau"]
        delta = args.delta if args.delta is not None else cfg["delta"]
        print(f"applying tau={tau} delta={delta}")
        pred = decide(scored, tau, delta)
        universe = (pl.read_parquet(resolve(args.parquet_dir) / f"{args.split}_source1.parquet")
                    .select(pl.col("entity_id").alias(TRUTH_ID)))
        sub = to_submission(pred, universe)
        out_dir = resolve(args.out); out_dir.mkdir(parents=True, exist_ok=True)
        # quote_style="never": polars would otherwise write "" for empty singleton rows,
        # which the scorer reads back as a literal two-character id. See docs/REPRODUCE.md, section 3.4.
        sub.write_csv(out_dir / "matching_results.tsv", separator="\t", quote_style="never")

        cand = pl.read_parquet(resolve(args.cand_dir) / f"{args.split}_reranked_cap{args.cap}.parquet")
        cands = to_submission(cand.select("s1", "s23"), universe).rename(
            {TRUTH_IDS: "candidate_entity_ids"})
        cands.write_csv(out_dir / "candidate_pairs.tsv", separator="\t", quote_style="never")

        n_match = sub.filter(pl.col(TRUTH_IDS) != "").height
        print(f"  {sub.height:,} rows, {n_match:,} with >=1 match "
              f"({100*n_match/sub.height:.1f}%), {pred.height:,} matched ids")
        print(f"  wrote {out_dir/'matching_results.tsv'} and {out_dir/'candidate_pairs.tsv'}")
        return 0

    ap.error("pass --sweep or --apply")


if __name__ == "__main__":
    raise SystemExit(main())
