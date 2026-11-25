"""Exp 1 — diagnose the GT pairs the UNCAPPED union fails to generate.

Reads the block parquets written by `blocking.py candidates` (data/cand/blk_*.parquet),
anti-joins ground truth against them, then:
  * aggregate structure of ALL misses (country, script, token sharing, DF of shared tokens)
  * 50 sampled misses printed in full (raw + normalized + tokens)

Diagnosis only. Writes nothing the pipeline consumes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
CAND = Path(os.environ.get("CAND_DIR", ROOT / "data/cand"))
NORM = ROOT / "data/normalized"
PQ = ROOT / "data/parquet_full"
SEED = 13
N_SAMPLE = int(os.environ.get("N_SAMPLE", 50))

con = duckdb.connect(config={"memory_limit": os.environ.get("DUCKDB_MEM", "9GB"),
                             "threads": 8, "preserve_insertion_order": False,
                             "temp_directory": str(CAND / "_diag_tmp")})

blk = str(CAND / "blk_*.parquet")
con.execute(f"CREATE VIEW s1 AS SELECT * FROM read_parquet('{NORM/'train_source1.parquet'}')")
con.execute(f"""CREATE VIEW s23 AS
    SELECT * FROM read_parquet('{NORM/'train_source2.parquet'}') UNION ALL
    SELECT * FROM read_parquet('{NORM/'train_source3.parquet'}')""")
n23 = con.execute("SELECT count(*) FROM s23").fetchone()[0]

print("building GT pair table ...", flush=True)
con.execute(f"""CREATE TABLE gp AS SELECT source1_entity_id AS s1, trim(m) AS s23
    FROM read_parquet('{PQ/'train_ground_truth.parquet'}'),
         unnest(str_split(matched_entity_ids, ',')) AS t(m)
    WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids) <> ''""")
n_gt = con.execute("SELECT count(*) FROM gp").fetchone()[0]

print("anti-joining against the uncapped union ...", flush=True)
con.execute(f"""CREATE TABLE miss AS
    SELECT g.s1, g.s23 FROM gp g
    ANTI JOIN read_parquet('{blk}') c ON c.s1 = g.s1 AND c.s23 = g.s23""")
n_miss = con.execute("SELECT count(*) FROM miss").fetchone()[0]
print(f"\nGT pairs {n_gt:,}   uncapped-union misses {n_miss:,}  ({100*n_miss/n_gt:.2f}%)\n", flush=True)

# ---------------------------------------------------------------- token DF tables
con.execute("""CREATE TABLE dfn AS SELECT token, count(*) df FROM
    (SELECT DISTINCT entity_id, token FROM
       (SELECT entity_id, unnest(name_tokens) token FROM s23) WHERE len(token)>=2) GROUP BY 1""")
con.execute("""CREATE TABLE dfa AS SELECT token, count(*) df FROM
    (SELECT DISTINCT entity_id, token FROM
       (SELECT entity_id, unnest(addr_tokens) token FROM s23) WHERE len(token)>=2) GROUP BY 1""")

# per-miss shared-token structure: how many shared tokens, and the DF of the rarest
con.execute("""CREATE TABLE mrec AS
    SELECT m.s1, m.s23,
           a.name_tokens ant, a.addr_tokens aat, b.name_tokens bnt, b.addr_tokens bat,
           a.country, b.name_script bscr, b.addr_script bascr,
           a.postcode apc, b.postcode bpc, a.addr_numbers anum, b.addr_numbers bnum
    FROM miss m JOIN s1 a ON a.entity_id=m.s1 JOIN s23 b ON b.entity_id=m.s23""")

con.execute("""CREATE TABLE mshare AS
  SELECT s1, s23, country, bscr,
     list_intersect(list_distinct(ant), list_distinct(bnt)) AS sn,
     list_intersect(list_distinct(aat), list_distinct(bat)) AS sa,
     len(list_distinct(ant)) an_k, len(list_distinct(bnt)) bn_k,
     len(list_distinct(aat)) aa_k, len(list_distinct(bat)) ba_k,
     apc, bpc, list_intersect(anum, bnum) AS snum
  FROM mrec""")

def q(sql):
    return con.execute(sql).fetchall()

print("=" * 90)
print("A. MISS STRUCTURE — do the two records share any token at all?")
print("=" * 90)
for lbl, col in (("name tokens shared", "sn"), ("addr tokens shared", "sa")):
    rows = q(f"""SELECT CASE WHEN len({col})=0 THEN '0' WHEN len({col})=1 THEN '1'
                             WHEN len({col})=2 THEN '2' ELSE '3+' END k,
                    count(*) FROM mshare GROUP BY 1 ORDER BY 1""")
    tot = sum(r[1] for r in rows)
    print(f"  {lbl:<22}" + "  ".join(f"{k}:{100*v/tot:5.1f}%" for k, v in rows))
rows = q("SELECT count(*) FILTER (WHERE len(sn)=0 AND len(sa)=0), count(*) FROM mshare")
print(f"  share NOTHING at all  {rows[0][0]:,} / {rows[0][1]:,}  ({100*rows[0][0]/rows[0][1]:.2f}%)")
rows = q("SELECT count(*) FILTER (WHERE len(snum)>0), count(*) FROM mshare")
print(f"  share a street number {rows[0][0]:,}  ({100*rows[0][0]/rows[0][1]:.2f}%)")
rows = q("SELECT count(*) FILTER (WHERE apc<>'' AND apc=bpc), count(*) FILTER (WHERE apc<>'' AND bpc<>''), count(*) FROM mshare")
print(f"  postcode equal        {rows[0][0]:,}   both-present {rows[0][1]:,} / {rows[0][2]:,}")

print()
print("=" * 90)
print("B. DF OF THE RAREST SHARED TOKEN (why the df caps missed it)")
print("=" * 90)
con.execute("""CREATE TABLE mdf AS
 SELECT s.s1, s.s23, s.country, s.bscr,
   (SELECT min(d.df) FROM (SELECT unnest(s.sn) t) u JOIN dfn d ON d.token=u.t) AS mn_df,
   (SELECT min(d.df) FROM (SELECT unnest(s.sa) t) u JOIN dfa d ON d.token=u.t) AS ma_df
 FROM mshare s""")
for lbl, col in (("name", "mn_df"), ("addr", "ma_df")):
    rows = q(f"""SELECT CASE WHEN {col} IS NULL THEN 'none shared'
                    WHEN {col}<=200 THEN '<=200' WHEN {col}<=1000 THEN '201-1k'
                    WHEN {col}<=5000 THEN '1k-5k' WHEN {col}<=20000 THEN '5k-20k'
                    WHEN {col}<=100000 THEN '20k-100k' ELSE '>100k' END k, count(*)
                 FROM mdf GROUP BY 1""")
    order = ['none shared', '<=200', '201-1k', '1k-5k', '5k-20k', '20k-100k', '>100k']
    d = dict(rows); tot = sum(d.values())
    print(f"  rarest shared {lbl} token DF:")
    for k in order:
        v = d.get(k, 0)
        print(f"      {k:<12}{v:>12,}  {100*v/tot:5.1f}%")

print()
print("=" * 90)
print("C. WOULD A WIDER SINGLE-TOKEN CAP REACH THEM? (cumulative recoverable share)")
print("=" * 90)
for cap in (200, 1000, 5000, 20000, 100000):
    r = q(f"""SELECT count(*) FROM mdf WHERE coalesce(mn_df, 1e18) <= {cap}
                                          OR coalesce(ma_df, 1e18) <= {cap}""")[0][0]
    print(f"  single shared token df<={cap:<7,} would reach {r:>12,} of {n_miss:,} misses"
          f"  ({100*r/n_miss:5.1f}%)  -> +{100*r/n_gt:4.2f}pp on GT recall")

print()
print("=" * 90)
print("D. MISSES BY COUNTRY AND SCRIPT (share of that stratum's GT lost)")
print("=" * 90)
con.execute("""CREATE TABLE gtx AS SELECT g.s1, g.s23, a.country,
    CASE WHEN b.name_script='latin' THEN 'Latin' ELSE b.name_script END scr
    FROM gp g JOIN s1 a ON a.entity_id=g.s1 JOIN s23 b ON b.entity_id=g.s23""")
for dim in ("country", "scr"):
    print(f"  by {dim}:")
    for k, tot, ms in q(f"""SELECT t.{dim}, count(*), count(*) FILTER (WHERE m.s1 IS NOT NULL)
            FROM gtx t LEFT JOIN miss m ON m.s1=t.s1 AND m.s23=t.s23
            GROUP BY 1 ORDER BY 2 DESC"""):
        print(f"      {str(k):<14}{tot:>12,} GT   {ms:>10,} missed  {100*ms/tot:5.2f}%")

print()
print("=" * 90)
print(f"E. {N_SAMPLE} SAMPLED MISSES, IN FULL")
print("=" * 90)
samp = con.execute(f"""
  SELECT m.s1, m.s23,
     a.business_name an_raw, a.business_address aa_raw, a.country ac,
     a.name_norm anm, a.addr_norm aam, a.name_tokens ant, a.addr_tokens aat,
     a.name_script ascr, a.addr_script aascr, a.postcode apc, a.addr_numbers anum,
     b.business_name bn_raw, b.business_address ba_raw, b.country bc,
     b.name_norm bnm, b.addr_norm bam, b.name_tokens bnt, b.addr_tokens bat,
     b.name_script bscr, b.addr_script bascr, b.postcode bpc, b.addr_numbers bnum
  FROM miss m JOIN s1 a ON a.entity_id=m.s1 JOIN s23 b ON b.entity_id=m.s23
  USING SAMPLE {N_SAMPLE} ROWS (reservoir, {SEED})""").arrow()
samp = pl.from_arrow(samp)
for i, r in enumerate(samp.iter_rows(named=True), 1):
    print(f"\n--- miss {i:>2} ---  {r['s1']}  vs  {r['s23']}   [{r['ac']} / {r['bc']}]"
          f"  scripts name {r['ascr']}|{r['bscr']}  addr {r['aascr']}|{r['bascr']}")
    print(f"  S1  name raw : {r['an_raw']!r}")
    print(f"      name norm: {r['anm']!r}")
    print(f"      name tok : {r['ant']}")
    print(f"      addr raw : {r['aa_raw']!r}")
    print(f"      addr norm: {r['aam']!r}")
    print(f"      addr tok : {r['aat']}   num={r['anum']} pc={r['apc']!r}")
    print(f"  S23 name raw : {r['bn_raw']!r}")
    print(f"      name norm: {r['bnm']!r}")
    print(f"      name tok : {r['bnt']}")
    print(f"      addr raw : {r['ba_raw']!r}")
    print(f"      addr norm: {r['bam']!r}")
    print(f"      addr tok : {r['bat']}   num={r['bnum']} pc={r['bpc']!r}")
    sn = set(r['ant']) & set(r['bnt']); sa = set(r['aat']) & set(r['bat'])
    print(f"  SHARED name tokens: {sorted(sn) or 'NONE'}")
    print(f"  SHARED addr tokens: {sorted(sa) or 'NONE'}")
print("\ndone.")
