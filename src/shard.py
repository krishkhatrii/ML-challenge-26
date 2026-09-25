"""Stage 4 — the 5% dev shard.

The full pipeline is ~7 hours, so roughly one run per day. Iterate on a shard instead.

The shard must preserve **distractor density**: how many plausible-but-wrong S2/S3
records compete for each S1 entity. Random sampling destroys it — blocking looks
artificially easy, the threshold tunes too low, and the full run underperforms what
validation promised.

Two modes, because measurement showed the originally planned approach cannot hold:

`s1` (default, recommended)
    Hash 5% of **S1 entity IDs**; keep the S2/S3 corpora **whole**. Every distractor
    that exists in the full run is present, and every true match is present, so
    distractor density and blocking recall are exact rather than approximated. The
    saving is on the S1 side, which is where the cost is: feature generation and
    inference scale with (S1 count x candidates per S1), so 5% of S1 is ~5% of the
    pair work. Only the blocking index build still pays full price, and that is the
    cheap half.

`keyspace` (as originally planned in docs/PLAN.md)
    Hash a blocking key, keep ~5% of the key space, keep every record falling in those
    keys across all three sources. Kept for comparison, but **it loses true matches**:
    measured on 1,000,000 true pairs from the full training data, a leading-trigram key
    agrees between an S1 record and its true match only **77.3%** of the time (82.8%
    where both sides have a key at all). S1 is 100% Latin while ~5-10% of S2/S3 is in
    one of 8 Indic scripts, and the raw fields simply do not share a key across that
    boundary before stage 5 normalization exists. A shard built this way silently caps
    blocking recall near 77%, so the >= 97% gate could never be read honestly.
    `--repair` adds the missing matched records back, at the cost of inflating match
    density above the full run's.

Output filenames match `src/convert.py`, so later stages just point `PARQUET_DIR` at
the shard directory. Train only — the shard exists to iterate on train/val.

Usage:
    python src/shard.py                                  # s1 mode, 5%
    python src/shard.py --frac 0.05 --mode keyspace      # the planned approach
    python src/shard.py --mode keyspace --repair
    PARQUET_DIR=data/parquet_full python src/shard.py --out data/shard_full

Env:
    PARQUET_DIR   input Parquet directory   (default: data/parquet)
    SHARD_DIR     output directory          (default: data/shard)
    SHARD_SALT    hash salt                 (default: dev-shard-v1)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ("train_source1", "train_source2", "train_source3")
GT = "train_ground_truth"
COLS = "entity_id, business_name, business_address, country"

# Blocking keys, as raw-field approximations of what stage 5 will do properly.
# Unicode-aware: \p{L}\p{N}\p{M} keeps Indic letters AND their combining marks. Dropping
# \p{M} deletes Devanagari vowel signs and shatters a name into single-character
# fragments -- the same bug found and fixed in src/blocking.py. Kept consistent so the
# shard key and the index tokenizer agree.
KEY_SQL = {
    "trigram": r"lower(regexp_replace(business_name, '[^\p{L}\p{N}\p{M}]', '', 'g'))[1:3]",
    "ntok": r"lower(regexp_extract(regexp_replace(business_name, '[^\p{L}\p{N}\p{M} ]', '', 'g'), '[^ ]+', 0))",
    "digit": r"coalesce(regexp_extract(business_address, '[0-9]+', 0), '')",
}
MOD = 1_000_000


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def keep_pred(key_expr: str, salt: str, frac: float) -> str:
    """Deterministic key-space membership. Stable across runs; DuckDB is pinned."""
    return f"(hash({key_expr} || '|' || '{salt}') % {MOD}) < {int(frac * MOD)}"


def n(v) -> str:
    return f"{v:,}"


def pct(a: int, b: int) -> str:
    return "n/a" if not b else f"{100.0 * a / b:.2f}%"


def main() -> int:
    ap = argparse.ArgumentParser(description="hash-based dev shard")
    ap.add_argument("--mode", choices=["s1", "keyspace"], default="s1")
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--key", choices=sorted(KEY_SQL), default="trigram")
    ap.add_argument("--salt", default=os.environ.get("SHARD_SALT", "dev-shard-v1"))
    ap.add_argument("--repair", action="store_true",
                    help="keyspace mode: add back matched records the key space dropped")
    ap.add_argument("--out", default=os.environ.get("SHARD_DIR", "data/shard"))
    ap.add_argument("--in-dir", default=os.environ.get("PARQUET_DIR", "data/parquet"))
    args = ap.parse_args()

    src_dir, out_dir = resolve(args.in_dir), resolve(args.out)
    missing = [t for t in (*SOURCES, GT) if not (src_dir / f"{t}.parquet").exists()]
    if missing:
        raise SystemExit(f"{src_dir} is missing {missing}. Run src/convert.py first.")
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(config={"memory_limit": "8GB", "threads": 8})
    for t in (*SOURCES, GT):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{src_dir / (t + '.parquet')}')")

    kexpr = KEY_SQL[args.key]
    print("=" * 78)
    print(f"DEV SHARD — mode={args.mode}  frac={args.frac}  key={args.key}  salt={args.salt!r}")
    print("=" * 78)
    print(f"  in   {src_dir}")
    print(f"  out  {out_dir}")
    print()

    full = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in SOURCES}

    # The shard's S1 universe is train_source1; ground truth must cover it. In
    # dataset_sample/ it does not -- the sample takes the first 50k rows of each file
    # INDEPENDENTLY, so the two files describe almost disjoint entity sets.
    orphan_gt = con.execute(f"""SELECT count(*) FROM {GT} g
        ANTI JOIN train_source1 s ON s.entity_id = g.source1_entity_id""").fetchone()[0]
    gt_all = con.execute(f"SELECT count(*) FROM {GT}").fetchone()[0]
    if orphan_gt:
        print(f"  WARNING: {n(orphan_gt)} of {n(gt_all)} ground-truth rows ({pct(orphan_gt, gt_all)})")
        print("           reference an S1 entity absent from train_source1. Expected on")
        print("           dataset_sample/ (files sampled independently); on the full dataset")
        print("           this must be 0. Shard diagnostics below are meaningless if nonzero.")
        print()

    # ---------------------------------------------------------------- select rows
    if args.mode == "s1":
        pred = keep_pred("entity_id", args.salt, args.frac)
        con.execute(f"CREATE TABLE keep1 AS SELECT {COLS} FROM train_source1 WHERE {pred}")
        con.execute(f"CREATE VIEW keep2 AS SELECT {COLS} FROM train_source2")
        con.execute(f"CREATE VIEW keep3 AS SELECT {COLS} FROM train_source3")
    else:
        pred = keep_pred(kexpr, args.salt, args.frac)
        for i, t in enumerate(SOURCES, 1):
            con.execute(f"CREATE TABLE keep{i} AS SELECT {COLS} FROM {t} WHERE {pred}")

    con.execute(f"""CREATE TABLE keepgt AS
        SELECT g.* FROM {GT} g SEMI JOIN keep1 k ON k.entity_id = g.source1_entity_id""")
    con.execute("""CREATE TABLE gp AS
        SELECT source1_entity_id AS s1, trim(m) AS mid
        FROM keepgt, unnest(str_split(matched_entity_ids, ',')) AS t(m)
        WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids) <> ''""")

    # keyspace mode drops true matches whose key disagrees with their S1's key.
    if args.mode == "keyspace" and args.repair:
        con.execute("""CREATE TABLE missing_ids AS
            SELECT mid FROM gp
            EXCEPT (SELECT entity_id FROM keep2 UNION ALL SELECT entity_id FROM keep3)""")
        for i, tbl in ((2, "train_source2"), (3, "train_source3")):
            con.execute(f"""INSERT INTO keep{i}
                SELECT {COLS} FROM {tbl}
                WHERE entity_id IN (SELECT mid FROM missing_ids)""")

    kept = {t: con.execute(f"SELECT count(*) FROM keep{i}").fetchone()[0]
            for i, t in enumerate(SOURCES, 1)}

    # ------------------------------------------------------------------- report
    print("Rows kept")
    print(f"  {'file':<18}{'full':>14}{'shard':>14}{'kept':>9}")
    print("  " + "-" * 55)
    for t in SOURCES:
        print(f"  {t:<18}{n(full[t]):>14}{n(kept[t]):>14}{pct(kept[t], full[t]):>9}")
    gt_rows = con.execute("SELECT count(*) FROM keepgt").fetchone()[0]
    gt_full = con.execute(f"SELECT count(*) FROM {GT}").fetchone()[0]
    print(f"  {GT:<18}{n(gt_full):>14}{n(gt_rows):>14}{pct(gt_rows, gt_full):>9}")

    print()
    print("Distractor density  (S2/S3 records competing for each kept S1 entity)")
    # The quantity that governs threshold tuning is how many plausible rivals a single
    # S1 entity faces -- i.e. how long its blocking posting list is -- not the corpus
    # size ratio. Measure it directly: for each kept S1 entity, how many S2/S3 records
    # share its blocking key, in the shard versus in the full data.
    con.execute(f"CREATE TABLE s1keys AS SELECT entity_id, {kexpr} AS k FROM keep1")
    con.execute(f"""CREATE TABLE kf AS SELECT {kexpr} AS k, count(*) c FROM
        (SELECT * FROM train_source2 UNION ALL SELECT * FROM train_source3) GROUP BY 1""")
    con.execute(f"""CREATE TABLE ks AS SELECT {kexpr} AS k, count(*) c FROM
        (SELECT * FROM keep2 UNION ALL SELECT * FROM keep3) GROUP BY 1""")
    # NOTE: ks is grouped by the *shard* rows' own key, recomputed identically.
    d_full, d_shard = con.execute("""
        SELECT avg(coalesce(f.c, 0)), avg(coalesce(s.c, 0))
        FROM s1keys LEFT JOIN kf f USING (k) LEFT JOIN ks s USING (k)""").fetchone()
    d_full, d_shard = float(d_full or 0), float(d_shard or 0)
    ratio = d_shard / d_full if d_full else 0.0
    print(f"  mean rivals per S1, full data : {d_full:>12,.1f}")
    print(f"  mean rivals per S1, shard     : {d_shard:>12,.1f}")
    print(f"  retention ratio               : {ratio:>12.4f}   (1.0000 = identical competition)")
    corpus = (kept["train_source2"] + kept["train_source3"]) / (full["train_source2"] + full["train_source3"])
    print(f"  S2/S3 corpus retained         : {100 * corpus:>11.2f}%")
    if ratio < 0.9:
        print(f"  WARNING: each S1 entity faces {1 / ratio if ratio else float('inf'):.1f}x fewer rivals than in the")
        print("           full run. Blocking will look easier than it is and the decision")
        print("           threshold will tune too low.")

    print()
    print("Ground-truth self-consistency  (every matched ID present in the sharded sources)")
    total_m, present_m = con.execute("""
        SELECT count(*), count(*) FILTER (WHERE s.entity_id IS NOT NULL)
        FROM gp LEFT JOIN (SELECT entity_id FROM keep2 UNION ALL SELECT entity_id FROM keep3) s
        ON s.entity_id = gp.mid""").fetchone()
    print(f"  matched IDs referenced : {n(total_m)}")
    print(f"  present in shard       : {n(present_m)}  ({pct(present_m, total_m)})")
    print(f"  MISSING                : {n(total_m - present_m)}")
    consistent = total_m == present_m
    print(f"  self-consistent        : {'YES' if consistent else 'NO'}")
    if not consistent:
        cap = present_m / total_m
        print(f"  Blocking recall on this shard is capped at {100 * cap:.1f}% — the >= 97% gate")
        print("  in stage 6 cannot be evaluated. Use --mode s1, or --repair.")

    # Why keyspace mode leaks: the key simply disagrees across the script boundary.
    if args.mode == "keyspace":
        both, agree = con.execute(f"""
            SELECT count(*) FILTER (WHERE ka <> '' AND kb <> '' AND ka IS NOT NULL AND kb IS NOT NULL),
                   count(*) FILTER (WHERE ka = kb AND ka <> '' AND ka IS NOT NULL)
            FROM (SELECT (SELECT {kexpr} FROM train_source1 WHERE entity_id = gp.s1) AS ka,
                         (SELECT {kexpr} FROM (SELECT * FROM train_source2 UNION ALL SELECT * FROM train_source3)
                          WHERE entity_id = gp.mid) AS kb
                  FROM gp USING SAMPLE reservoir(20000 ROWS) REPEATABLE (42))""").fetchone()
        print()
        print(f"Key agreement ({args.key}, 20k sampled true pairs)")
        print(f"  both sides have a key : {n(both)}")
        print(f"  keys agree            : {n(agree)}  ({pct(agree, both)})")

    print()
    print("Kept S1 entities by country  (skew check)")
    rows = con.execute("""
        SELECT f.country, f.c AS full_c, coalesce(k.c, 0) AS kept_c FROM
          (SELECT country, count(*) c FROM train_source1 GROUP BY 1) f
          LEFT JOIN (SELECT country, count(*) c FROM keep1 GROUP BY 1) k USING (country)
        ORDER BY full_c DESC""").fetchall()
    print(f"  {'country':<10}{'full':>12}{'shard':>12}{'kept':>9}")
    print("  " + "-" * 43)
    for ctry, fc, kc in rows:
        print(f"  {str(ctry):<10}{n(fc):>12}{n(kc):>12}{pct(kc, fc):>9}")

    # -------------------------------------------------------------------- write
    print()
    print("Writing")
    for i, t in enumerate(SOURCES, 1):
        dst = out_dir / f"{t}.parquet"
        # s1 mode leaves S2/S3 untouched -- hardlink rather than duplicate ~400 MB.
        if args.mode == "s1" and i in (2, 3):
            src = src_dir / f"{t}.parquet"
            dst.unlink(missing_ok=True)
            try:
                os.link(src, dst)
                how = "hardlink (unchanged)"
            except OSError:
                shutil.copy2(src, dst)
                how = "copy (unchanged)"
        else:
            con.execute(f"COPY (SELECT {COLS} FROM keep{i}) TO '{dst}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            how = "written"
        print(f"  {dst.name:<28}{dst.stat().st_size / 1e6:>9.1f} MB  {how}")
    dst = out_dir / f"{GT}.parquet"
    con.execute(f"COPY (SELECT * FROM keepgt) TO '{dst}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    print(f"  {dst.name:<28}{dst.stat().st_size / 1e6:>9.1f} MB  written")

    cfg = {
        "mode": args.mode, "frac": args.frac, "key": args.key, "salt": args.salt,
        "repair": args.repair, "source_parquet": str(src_dir),
        "rows": {**{t: kept[t] for t in SOURCES}, GT: gt_rows},
        "rival_retention_ratio": round(ratio, 6),
        "rivals_per_s1_full": round(d_full, 2), "rivals_per_s1_shard": round(d_shard, 2),
        "orphan_gt_rows": orphan_gt,
        "gt_self_consistent": consistent,
        "gt_matched_present": present_m, "gt_matched_total": total_m,
    }
    (out_dir / "_shard.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"  _shard.json                            config + diagnostics")
    print()
    print(f"Use it:  PARQUET_DIR={args.out} python src/scoring.py --split")
    return 0 if consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
