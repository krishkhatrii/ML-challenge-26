"""Exp 1b — quantify each candidate lever against the 831,258 uncapped-union misses.

Reads data/cand/blk_*.parquet (the uncapped union) and measures, for each lever, how
many of the misses it would make reachable. Recovery is an UPPER bound on the recall
the lever can add; it says nothing about the pair volume that comes with it.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
CAND = ROOT / "data/cand"
NORM = ROOT / "data/normalized"
PQ = ROOT / "data/parquet_full"

con = duckdb.connect(config={"memory_limit": os.environ.get("DUCKDB_MEM", "9GB"),
                             "threads": 8, "preserve_insertion_order": False,
                             "temp_directory": str(CAND / "_diag_tmp")})
blk = str(CAND / "blk_*.parquet")
con.execute(f"CREATE VIEW s1 AS SELECT * FROM read_parquet('{NORM/'train_source1.parquet'}')")
con.execute(f"""CREATE VIEW s23 AS
    SELECT * FROM read_parquet('{NORM/'train_source2.parquet'}') UNION ALL
    SELECT * FROM read_parquet('{NORM/'train_source3.parquet'}')""")
N = con.execute("SELECT count(*) FROM s23").fetchone()[0]

print("GT + miss tables ...", flush=True)
con.execute(f"""CREATE TABLE gp AS SELECT source1_entity_id AS s1, trim(m) AS s23
    FROM read_parquet('{PQ/'train_ground_truth.parquet'}'),
         unnest(str_split(matched_entity_ids, ',')) AS t(m)
    WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids) <> ''""")
n_gt = con.execute("SELECT count(*) FROM gp").fetchone()[0]
con.execute(f"""CREATE TABLE miss AS SELECT g.s1, g.s23 FROM gp g
    ANTI JOIN read_parquet('{blk}') c ON c.s1 = g.s1 AND c.s23 = g.s23""")
n_m = con.execute("SELECT count(*) FROM miss").fetchone()[0]
print(f"  GT {n_gt:,}   misses {n_m:,}  ({100*n_m/n_gt:.2f}%)\n", flush=True)

# combined per-record token set (name + address in ONE namespace) and its DF
print("combined token DF ...", flush=True)
for side in ("s1", "s23"):
    con.execute(f"""CREATE TABLE ct_{side} AS SELECT entity_id,
        list_filter(list_distinct(list_concat(name_tokens, addr_tokens)),
                    x -> len(x) >= 2) AS tok FROM {side}""")
con.execute("""CREATE TABLE dfc AS SELECT token, count(*) df FROM
    (SELECT entity_id, unnest(tok) token FROM ct_s23) GROUP BY 1""")

# leading-zero-stripped / range-split street numbers
print("normalised street numbers ...", flush=True)
for side in ("s1", "s23"):
    con.execute(f"""CREATE TABLE nm_{side} AS SELECT entity_id,
        list_distinct(list_transform(
          list_filter(
            flatten(list_transform(addr_numbers, x -> str_split(regexp_replace(x, '[^0-9/-]', '', 'g'), '-'))),
            y -> y <> ''),
          z -> ltrim(z, '0'))) AS num
        FROM {side}""")

print("miss feature table ...", flush=True)
con.execute("""CREATE TABLE mf AS SELECT m.s1, m.s23,
    list_sort(list_distinct(a.name_tokens)) ans, list_sort(list_distinct(b.name_tokens)) bns,
    list_sort(list_distinct(a.addr_tokens)) aas, list_sort(list_distinct(b.addr_tokens)) bas,
    list_intersect(ca.tok, cb.tok) sc,
    list_intersect(na.num, nb.num) snum,
    len(list_intersect(a.addr_numbers, b.addr_numbers)) rawnum
  FROM miss m
  JOIN s1 a ON a.entity_id=m.s1 JOIN s23 b ON b.entity_id=m.s23
  JOIN ct_s1 ca ON ca.entity_id=m.s1 JOIN ct_s23 cb ON cb.entity_id=m.s23
  JOIN nm_s1 na ON na.entity_id=m.s1 JOIN nm_s23 nb ON nb.entity_id=m.s23""")

# DFs of the shared combined tokens, ascending
con.execute("""CREATE TABLE md AS SELECT s1, s23, ans, bns, aas, bas, sc, snum, rawnum,
    (SELECT list_sort(list(d.df)) FROM (SELECT unnest(sc) t) u JOIN dfc d ON d.token=u.t) AS dfs
  FROM mf""")

def pct(n):
    return f"{n:>10,}  {100*n/n_m:5.1f}% of misses  -> +{100*n/n_gt:5.2f}pp GT recall"

print()
print("=" * 92)
print(f"LEVER RECOVERY — upper bound on what each rule could add. misses = {n_m:,}")
print("=" * 92)

rows = [
 ("L1  identical name token SET (exact key)",
  "list_sort(ans) = list_sort(bns) AND len(ans) > 0"),
 ("L2  identical addr token SET (exact key)",
  "list_sort(aas) = list_sort(bas) AND len(aas) > 0"),
 ("L3  identical name OR addr token SET",
  "(list_sort(ans)=list_sort(bns) AND len(ans)>0) OR (list_sort(aas)=list_sort(bas) AND len(aas)>0)"),
 ("L4  >=2 shared tokens, ANY df (combined field)", "len(sc) >= 2"),
 ("L5  >=3 shared tokens, ANY df (combined field)", "len(sc) >= 3"),
 ("L6  street number match, leading zeros stripped + ranges split", "len(snum) >= 1"),
 ("L6b  ... and it did NOT match raw (what the fix alone buys)",
  "len(snum) >= 1 AND rawnum = 0"),
 ("L7  normalised number AND >=1 shared token", "len(snum) >= 1 AND len(sc) >= 1"),
]
for label, cond in rows:
    n = con.execute(f"SELECT count(*) FROM md WHERE {cond}").fetchone()[0]
    print(f"  {label:<62}{pct(n)}")

print()
print("=" * 92)
print("L8  PRODUCT-DF PAIR RULE — a shared token pair whose expected intersection is small")
print("=" * 92)
print("    current B3/B4 rule is df<=5,000 on EACH token, which forbids a 20k x 20k pair")
print(f"    (expected intersection 20k*20k/{N:,} = {20000*20000/N:.0f} records) while allowing")
print(f"    5k x 5k (expected {5000*5000/N:.1f}). The binding quantity is the product.\n")
print(f"  {'budget: df1*df2 <= N *':<34}{'misses reachable':>18}{'share':>9}{'+pp GT':>9}")
print("  " + "-" * 70)
for b in (5, 20, 50, 200, 1000, 5000):
    n = con.execute(f"""SELECT count(*) FROM md
        WHERE len(dfs) >= 2 AND dfs[1]::BIGINT * dfs[2] <= {b}::BIGINT * {N}""").fetchone()[0]
    print(f"  {('N * ' + str(b)):<34}{n:>18,}{100*n/n_m:>8.1f}%{100*n/n_gt:>8.2f}")

print()
print("=" * 92)
print("L9  TRIPLE rule — three shared tokens with a product budget")
print("=" * 92)
for b in (1, 5, 20, 50):
    n = con.execute(f"""SELECT count(*) FROM md
        WHERE len(dfs) >= 3 AND dfs[1]::HUGEINT * dfs[2] * dfs[3] <= {b}::HUGEINT * {N} * {N}""").fetchone()[0]
    print(f"  df1*df2*df3 <= {b} * N^2 {'':<12}{n:>18,}{100*n/n_m:>8.1f}%{100*n/n_gt:>8.2f}")

print()
print("=" * 92)
print("COMBINED: best realistic package")
print("=" * 92)
combo = ("(len(dfs) >= 2 AND dfs[1]::BIGINT*dfs[2] <= 50::BIGINT * %d)"
         " OR (list_sort(ans)=list_sort(bns) AND len(ans)>0)"
         " OR (list_sort(aas)=list_sort(bas) AND len(aas)>0)"
         " OR (len(snum) >= 1 AND len(sc) >= 1)") % N
n = con.execute(f"SELECT count(*) FROM md WHERE {combo}").fetchone()[0]
print(f"  L8(N*50) OR L1 OR L2 OR L7 {'':<28}{pct(n)}")
print(f"  implied uncapped union recall: {100*(n_gt - n_m + n)/n_gt:.2f}%")

print()
print("=" * 92)
print("RESIDUE — misses none of the above reaches, by country/script")
print("=" * 92)
con.execute(f"CREATE TABLE resid AS SELECT s1, s23 FROM md WHERE NOT ({combo})")
for k, v in con.execute("""SELECT a.country, count(*) FROM resid r
    JOIN s1 a ON a.entity_id=r.s1 GROUP BY 1 ORDER BY 2 DESC""").fetchall():
    print(f"  {str(k):<14}{v:>12,}")
print()
for k, v in con.execute("""SELECT count(*) FILTER (WHERE b.addr_norm=''), count(*)
    FROM resid r JOIN s23 b ON b.entity_id=r.s23""").fetchall()[:1]:
    pass
r = con.execute("""SELECT count(*) FILTER (WHERE b.addr_norm=''), count(*) FROM resid r
    JOIN s23 b ON b.entity_id=r.s23""").fetchone()
print(f"  residue with an EMPTY S2/S3 address: {r[0]:,} of {r[1]:,}  ({100*r[0]/max(r[1],1):.1f}%)")
print("\ndone.")
