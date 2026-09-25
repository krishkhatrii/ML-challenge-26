"""Stage 7 — pairwise features for every (S1, candidate) pair.

Chunked by S1 hash: every candidate of a given S1 lands in the same chunk, which the
context features require (rank within shortlist, gap to the best, and so on are only
computable with the whole shortlist in hand). Each chunk is written to Parquet and
freed, so the full pair set is never resident.

String metrics come from RapidFuzz `process.cpdist`, which is elementwise, C++, and
releases the GIL -- `workers=-1` genuinely uses all cores. Char-n-gram cosine is the one
metric RapidFuzz does not provide; it is computed with a sparse hashing vectorizer rather
than Python set arithmetic, which would dominate runtime at this scale.

Usage:
    python src/features.py --split train --fold val
    python src/features.py --split train --fold train --s1-frac 0.12
    python src/features.py --split test
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz, process
from sklearn.feature_extraction.text import HashingVectorizer

ROOT = Path(__file__).resolve().parent.parent
NGRAM_FEATURES = 2 ** 18
_VEC = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3),
                         n_features=NGRAM_FEATURES, norm="l2", alternate_sign=False)


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def cpd(a, b, scorer, scale=1.0):
    return (np.asarray(process.cpdist(a, b, scorer=scorer, workers=-1), dtype=np.float32) * scale)


def ngram_cos(a: list[str], b: list[str]) -> np.ndarray:
    """Row-wise cosine over L2-normalised char-3-gram hashes: a sparse elementwise
    multiply and row sum, instead of per-pair Python set arithmetic."""
    Xa, Xb = _VEC.transform(a), _VEC.transform(b)
    return np.asarray(Xa.multiply(Xb).sum(axis=1)).ravel().astype(np.float32)


def idf_overlap(at: list, bt: list, idf: dict) -> tuple[np.ndarray, np.ndarray]:
    """IDF-weighted token overlap and plain Jaccard over two token-list columns."""
    w = np.empty(len(at), dtype=np.float32)
    j = np.empty(len(at), dtype=np.float32)
    for i in range(len(at)):
        A = set(at[i] or ()); B = set(bt[i] or ())
        if not A or not B:
            w[i] = 0.0; j[i] = 0.0; continue
        inter = A & B
        if not inter:
            w[i] = 0.0; j[i] = 0.0; continue
        wi = 0.0
        for t in inter:
            wi += idf.get(t, 0.0)
        wu = 0.0
        for t in A | B:
            wu += idf.get(t, 0.0)
        w[i] = wi / wu if wu else 0.0
        j[i] = len(inter) / len(A | B)
    return w, j


def num_feats(an: list, bn: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numeric-token overlap, street-number (first number) match, and a both-present flag."""
    ov = np.zeros(len(an), dtype=np.float32)
    first = np.zeros(len(an), dtype=np.float32)
    both = np.zeros(len(an), dtype=np.float32)
    for i in range(len(an)):
        A = list(an[i] or ()); B = list(bn[i] or ())
        if not A or not B:
            continue
        both[i] = 1.0
        sa, sb = set(A), set(B)
        ov[i] = len(sa & sb) / len(sa | sb)
        first[i] = 1.0 if A[0] == B[0] else 0.0
    return ov, first, both


SCRIPTS = ("latin", "devanagari", "bengali", "gurmukhi", "gujarati", "oriya",
           "tamil", "telugu", "kannada", "malayalam")
SCRIPT_ID = {s: i for i, s in enumerate(SCRIPTS)}

FEATURES = [
    "n_jw", "n_lev", "n_tset", "n_tsort", "n_ngcos", "n_idfov", "n_jac",
    "a_jw", "a_lev", "a_tset", "a_tsort", "a_ngcos", "a_idfov", "a_jac",
    "a_numov", "a_numfirst", "a_numboth", "a_pcmatch", "a_pcboth",
    "a_null_s1", "a_null_s23",
    "m_country_eq", "m_is_s2", "m_script_match", "m_script_s23",
    "blk_score", "n_blocks", "jac_plain", "jac_idf",
    "c_rank", "c_gap_best", "c_gap_second", "c_ratio_mean", "c_n_above", "c_shortlist",
]


def build_chunk(df: pl.DataFrame, idf_n: dict, idf_a: dict) -> pl.DataFrame:
    an, bn = df["an"].to_list(), df["bn"].to_list()
    aa, ba = df["aa"].to_list(), df["ba"].to_list()
    out = {
        "n_jw": cpd(an, bn, distance.JaroWinkler.normalized_similarity),
        "n_lev": cpd(an, bn, distance.Levenshtein.normalized_similarity),
        "n_tset": cpd(an, bn, fuzz.token_set_ratio, 0.01),
        "n_tsort": cpd(an, bn, fuzz.token_sort_ratio, 0.01),
        "n_ngcos": ngram_cos(an, bn),
        "a_jw": cpd(aa, ba, distance.JaroWinkler.normalized_similarity),
        "a_lev": cpd(aa, ba, distance.Levenshtein.normalized_similarity),
        "a_tset": cpd(aa, ba, fuzz.token_set_ratio, 0.01),
        "a_tsort": cpd(aa, ba, fuzz.token_sort_ratio, 0.01),
        "a_ngcos": ngram_cos(aa, ba),
    }
    out["n_idfov"], out["n_jac"] = idf_overlap(df["ant"].to_list(), df["bnt"].to_list(), idf_n)
    out["a_idfov"], out["a_jac"] = idf_overlap(df["aat"].to_list(), df["bat"].to_list(), idf_a)
    out["a_numov"], out["a_numfirst"], out["a_numboth"] = num_feats(
        df["anum"].to_list(), df["bnum"].to_list())

    apc, bpc = df["apc"].to_numpy(), df["bpc"].to_numpy()
    out["a_pcboth"] = ((apc != "") & (bpc != "")).astype(np.float32)
    out["a_pcmatch"] = ((apc == bpc) & (apc != "")).astype(np.float32)
    out["a_null_s1"] = (df["aa"].to_numpy() == "").astype(np.float32)
    out["a_null_s23"] = (df["ba"].to_numpy() == "").astype(np.float32)
    out["m_country_eq"] = (df["ac"].to_numpy() == df["bc"].to_numpy()).astype(np.float32)
    out["m_is_s2"] = np.fromiter((s.startswith("S2-") for s in df["s23"].to_list()),
                                 dtype=np.float32, count=df.height)
    asc = df["ascr"].to_list(); bsc = df["bsc"].to_list()
    out["m_script_match"] = np.fromiter((1.0 if x == y else 0.0 for x, y in zip(asc, bsc)),
                                        dtype=np.float32, count=df.height)
    out["m_script_s23"] = np.fromiter((SCRIPT_ID.get(y, -1) for y in bsc),
                                      dtype=np.float32, count=df.height)

    feat = df.select("s1", "s23").with_columns(
        **{k: pl.Series(k, v) for k, v in out.items()},
        blk_score=df["blk_score"].cast(pl.Float32),
        n_blocks=df["n_blocks"].cast(pl.Float32),
        jac_plain=df["jac_plain"].fill_null(0.0).cast(pl.Float32),
        jac_idf=df["jac_idf"].fill_null(0.0).cast(pl.Float32),
        c_rank=df["rk"].cast(pl.Float32),
    )
    # Context: matching is competitive, not independent. A 0.7 means something different
    # when the runner-up scored 0.2 than when three others also scored 0.7.
    base = (out["n_jw"] + out["a_jw"]) / 2.0
    feat = feat.with_columns(_base=pl.Series("_base", base))
    g = feat.group_by("s1").agg(
        _best=pl.col("_base").max(), _mean=pl.col("_base").mean(),
        _second=pl.col("_base").sort(descending=True).get(1, null_on_oob=True),
        _n_above=(pl.col("_base") > 0.8).sum(), _shortlist=pl.len())
    feat = feat.join(g, on="s1", how="left").with_columns(
        c_gap_best=(pl.col("_base") - pl.col("_best")).cast(pl.Float32),
        c_gap_second=(pl.col("_base") - pl.col("_second").fill_null(0.0)).cast(pl.Float32),
        c_ratio_mean=(pl.col("_base") / pl.when(pl.col("_mean") > 0).then(pl.col("_mean")).otherwise(1.0)).cast(pl.Float32),
        c_n_above=pl.col("_n_above").cast(pl.Float32),
        c_shortlist=pl.col("_shortlist").cast(pl.Float32),
    ).drop("_base", "_best", "_mean", "_second", "_n_above", "_shortlist")
    return feat.select(["s1", "s23"] + FEATURES)


def main() -> int:
    ap = argparse.ArgumentParser(description="stage 7: pairwise features")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--fold", choices=["train", "val", "all"], default="all")
    ap.add_argument("--s1-frac", type=float, default=1.0, help="subsample S1 entities")
    ap.add_argument("--norm-dir", default=os.environ.get("NORM_DIR", "data/normalized"))
    ap.add_argument("--cand-dir", default=os.environ.get("CAND_DIR", "data/cand"))
    ap.add_argument("--parquet-dir", default=os.environ.get("PARQUET_DIR", "data/parquet"))
    ap.add_argument("--out", default=os.environ.get("FEAT_DIR", "data/features"))
    ap.add_argument("--cap", type=int, default=30)
    ap.add_argument("--chunk-s1", type=int, default=100_000)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    norm, cand, out_dir = resolve(args.norm_dir), resolve(args.cand_dir), resolve(args.out)
    tag = args.tag or f"{args.split}_{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cpath = cand / f"{args.split}_reranked_cap{args.cap}.parquet"
    if not cpath.exists():
        raise SystemExit(f"{cpath} not found. Run `blocking.py rerank` first.")

    con = duckdb.connect(config={"memory_limit": os.environ.get("DUCKDB_MEM", "7GB"),
                                 "threads": 8, "preserve_insertion_order": False,
                                 "temp_directory": str(out_dir / "_tmp")})
    con.execute(f"CREATE VIEW s1 AS SELECT * FROM read_parquet('{norm / (args.split + '_source1.parquet')}')")
    con.execute(f"""CREATE VIEW s23 AS
        SELECT * FROM read_parquet('{norm / (args.split + '_source2.parquet')}')
        UNION ALL SELECT * FROM read_parquet('{norm / (args.split + '_source3.parquet')}')""")
    n23 = con.execute("SELECT count(*) FROM s23").fetchone()[0]

    print(f"features: split={args.split} fold={args.fold} s1_frac={args.s1_frac} -> {out_dir}/{tag}_*")
    t0 = time.perf_counter()
    idf_n = dict(con.execute(f"""SELECT token, ln({n23}.0/count(*)) FROM
        (SELECT DISTINCT entity_id, unnest(name_tokens) token FROM s23) GROUP BY 1""").fetchall())
    idf_a = dict(con.execute(f"""SELECT token, ln({n23}.0/count(*)) FROM
        (SELECT DISTINCT entity_id, unnest(addr_tokens) token FROM s23) GROUP BY 1""").fetchall())
    print(f"  idf maps: {len(idf_n):,} name / {len(idf_a):,} addr tokens  ({time.perf_counter()-t0:.0f}s)")

    # The fold MUST be derived by src/scoring.py itself, not re-implemented here.
    # DuckDB's hash() and Polars' hash(seed=13) are different functions, so filtering
    # the fold in SQL produced a "val" set that overlapped the real val fold by only
    # 20% -- and put a fifth of the val entities into the training sample. Build the
    # id list in Polars and hand it to DuckDB as a table.
    join_sql = ""
    if args.fold != "all" or args.s1_frac < 1.0:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from scoring import TRUTH_ID, assign_fold
        ids = (pl.read_parquet(norm / f"{args.split}_source1.parquet")
                 .select(pl.col("entity_id").alias(TRUTH_ID)))
        if args.fold != "all":
            ids = assign_fold(ids).filter(pl.col("fold") == args.fold).drop("fold")
        if args.s1_frac < 1.0:
            ids = ids.filter(
                (pl.col(TRUTH_ID).hash(seed=97) % 1_000_000) < int(args.s1_frac * 1e6))
        keep_s1 = ids.rename({TRUTH_ID: "entity_id"})
        con.register("keep_s1", keep_s1)
        join_sql = " JOIN keep_s1 k ON k.entity_id = c.s1"
        print(f"  fold={args.fold} s1_frac={args.s1_frac}: {keep_s1.height:,} S1 entities selected")

    nchunk = max(1, con.execute(f"SELECT count(*) FROM s1").fetchone()[0] // args.chunk_s1)
    written, rows = [], 0
    for i in range(nchunk):
        q = f"""SELECT c.s1, c.s23, c.blk_score, c.n_blocks, c.jac_plain, c.jac_idf, c.rk,
            a.name_core an, a.addr_norm aa, a.name_tokens ant, a.addr_tokens aat,
            a.addr_numbers anum, a.postcode apc, a.country ac, a.name_script ascr,
            b.name_core bn, b.addr_norm ba, b.name_tokens bnt, b.addr_tokens bat,
            b.addr_numbers bnum, b.postcode bpc, b.country bc, b.name_script bsc
            FROM read_parquet('{cpath}') c{join_sql}
            JOIN s1 a ON a.entity_id = c.s1 JOIN s23 b ON b.entity_id = c.s23
            WHERE hash(c.s1) % {nchunk} = {i}"""
        df = pl.from_arrow(con.execute(q).arrow())
        if df.height == 0:
            continue
        feat = build_chunk(df, idf_n, idf_a)
        dst = out_dir / f"{tag}_part{i:03d}.parquet"
        feat.write_parquet(dst, compression="zstd")
        written.append(dst); rows += feat.height
        print(f"  chunk {i+1:>3}/{nchunk}  {feat.height:>10,} pairs  "
              f"{time.perf_counter()-t0:>7.0f}s cumulative", flush=True)
        del df, feat
    print(f"  DONE {rows:,} pairs in {len(written)} parts, {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
