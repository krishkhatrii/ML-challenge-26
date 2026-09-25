"""Stage 8 — LightGBM binary classifier over candidate pairs.

Positives are ground-truth matches; negatives are blocked candidates that are not
matches, which makes them naturally hard negatives -- they already survived blocking.
Cross-validation is grouped by S1 entity so no entity straddles folds.

Usage:
    python src/train.py --feat-glob 'data/features/train_train_part*.parquet'
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
SEED = 13

PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.05, num_leaves=127,
    min_data_in_leaf=200, feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, num_threads=8, verbosity=-1, seed=SEED,
)


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def main() -> int:
    ap = argparse.ArgumentParser(description="stage 8: LightGBM pair classifier")
    ap.add_argument("--feat-glob", default="data/features/train_train_part*.parquet")
    ap.add_argument("--parquet-dir", default=os.environ.get("PARQUET_DIR", "data/parquet"))
    ap.add_argument("--max-pairs", type=int, default=5_000_000)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--out", default="data/model")
    args = ap.parse_args()

    out_dir = resolve(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    glob = str(resolve(args.feat_glob)) if not args.feat_glob.startswith("/") else args.feat_glob
    con = duckdb.connect(config={"memory_limit": "7GB", "threads": 8,
                                 "preserve_insertion_order": False})
    con.execute(f"""CREATE VIEW gp AS SELECT source1_entity_id s1, trim(m) s23
        FROM read_parquet('{resolve(args.parquet_dir) / 'train_ground_truth.parquet'}'),
             unnest(str_split(matched_entity_ids, ',')) t(m)
        WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids) <> ''""")
    total = con.execute(f"SELECT count(*) FROM read_parquet('{glob}')").fetchone()[0]
    frac = min(1.0, args.max_pairs / max(total, 1))
    print(f"feature pairs available {total:,}; sampling {frac:.3f} -> ~{int(total*frac):,}")

    # Stratified: keep every positive, subsample negatives. Positives are ~4% of the
    # candidate set, so uniform sampling would throw away most of the signal.
    t0 = time.perf_counter()
    df = pl.from_arrow(con.execute(f"""
        SELECT f.*, CASE WHEN g.s1 IS NOT NULL THEN 1 ELSE 0 END AS y
        FROM read_parquet('{glob}') f
        LEFT JOIN gp g ON g.s1 = f.s1 AND g.s23 = f.s23
        WHERE g.s1 IS NOT NULL OR (hash(f.s1 || f.s23) % 1000000) < {int(frac*1e6)}
    """).arrow())
    feats = [c for c in df.columns if c not in ("s1", "s23", "y")]
    y = df["y"].to_numpy()
    X = df.select(feats).to_numpy().astype(np.float32)
    groups = df["s1"].to_numpy()
    print(f"  loaded {len(y):,} pairs, {y.sum():,} positives ({100*y.mean():.2f}%), "
          f"{len(feats)} features, {time.perf_counter()-t0:.0f}s")

    gkf = GroupKFold(n_splits=args.folds)
    aucs, imps, models = [], [], []
    oof = np.zeros(len(y))
    for k, (tr, va) in enumerate(gkf.split(X, y, groups), 1):
        ds_tr = lgb.Dataset(X[tr], label=y[tr], feature_name=feats)
        ds_va = lgb.Dataset(X[va], label=y[va], feature_name=feats, reference=ds_tr)
        m = lgb.train(PARAMS, ds_tr, num_boost_round=args.rounds, valid_sets=[ds_va],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        p = m.predict(X[va], num_iteration=m.best_iteration)
        oof[va] = p
        auc = roc_auc_score(y[va], p)
        aucs.append(auc); models.append(m)
        imps.append(m.feature_importance(importance_type="gain"))
        print(f"  fold {k}: AUC {auc:.5f}  best_iter {m.best_iteration}")

    print(f"\n  CV AUC  mean {np.mean(aucs):.5f}  std {np.std(aucs):.5f}")
    print(f"  OOF AUC {roc_auc_score(y, oof):.5f}")

    gain = np.mean(imps, axis=0); order = np.argsort(gain)[::-1]
    print(f"\n  {'feature':<18}{'gain %':>10}")
    print("  " + "-" * 28)
    for i in order[:20]:
        print(f"  {feats[i]:<18}{100*gain[i]/gain.sum():>9.2f}%")

    best = models[int(np.argmax(aucs))]
    best.save_model(str(out_dir / "lgbm.txt"))
    (out_dir / "features.json").write_text(json.dumps(feats, indent=2) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps({
        "cv_auc_mean": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs)),
        "oof_auc": float(roc_auc_score(y, oof)), "n_pairs": int(len(y)),
        "n_pos": int(y.sum()), "features": feats,
        "gain": {feats[i]: float(gain[i]) for i in order},
    }, indent=2) + "\n")
    print(f"\n  saved {out_dir/'lgbm.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
