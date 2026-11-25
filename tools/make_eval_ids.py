"""Write the S1 id set the experiment loop evaluates on.

  val fold (scoring.assign_fold, seed 13)          -> where macro-F_0.5 is measured
  UNION  train fold sampled at --train-frac        -> what the model is fitted on

Blocking is per-S1 independent, so restricting candidate generation to this set gives
byte-identical candidates for these entities at ~3.5x less work.
"""
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from scoring import TRUTH_ID, assign_fold  # noqa: E402

TRAIN_FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
out = ROOT / "data/eval_ids.parquet"

ids = (pl.read_parquet(ROOT / "data/normalized/train_source1.parquet")
         .select(pl.col("entity_id").alias(TRUTH_ID)))
f = assign_fold(ids)
val = f.filter(pl.col("fold") == "val").select(TRUTH_ID)
# identical rule to features.py --s1-frac, so the two stages select the same entities
tr = (f.filter(pl.col("fold") == "train").select(TRUTH_ID)
       .filter((pl.col(TRUTH_ID).hash(seed=97) % 1_000_000) < int(TRAIN_FRAC * 1e6)))
keep = pl.concat([val, tr]).unique().rename({TRUTH_ID: "entity_id"})
keep.write_parquet(out, compression="zstd")
print(f"val {val.height:,} + train@{TRAIN_FRAC} {tr.height:,} = {keep.height:,} S1 ids"
      f"  ({100*keep.height/ids.height:.1f}% of S1)  -> {out}")
