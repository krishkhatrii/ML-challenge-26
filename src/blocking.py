"""Stage 6 (part 1) — build the inverted index over S2+S3 business names.

Index build only. Candidate generation is deliberately not here yet: no posting-list
unions, no rare-token lookups, no recall measurement. This exists to answer three
questions before the rest of stage 6 is designed — how long the build takes, how much
memory it peaks at, and how big the index is on disk.

Why an inverted index at all: instead of "which records are similar to this one?"
(10.3M comparisons), ask "which records share a rare token with this one?" — a keyed
lookup returning a few hundred records. 2.2M x 10.3M = 22 trillion pairs is not
reachable any other way.

**The tokenizer here is provisional.** `src/normalize.py` (stage 5) does not exist yet,
so there is no transliteration: an Indic-script name yields Indic tokens, which can
never match the Latin tokens of the S1 record it belongs to. That caps achievable recall
on ~5-10% of S2/S3 and is exactly what stage 5 fixes. Treat the numbers below as a cost
and capacity measurement, not as a recall result. `TOKENS_SQL` is the single place to
swap in the real normalizer.

Usage:
    python src/blocking.py                             # train split
    python src/blocking.py --split test
    PARQUET_DIR=data/parquet_full python src/blocking.py --out data/index

Env:
    PARQUET_DIR   input Parquet directory   (default: data/parquet)
    INDEX_DIR     output directory          (default: data/index)
    DUCKDB_MEM    DuckDB memory limit       (default: 10GB)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
MIN_TOKEN_LEN = 2

# Unicode-aware tokenizer. \p{M} is load-bearing: Devanagari and other Indic scripts
# write vowels as combining marks, so a [^\p{L}\p{N}] class deletes them and shatters
# "राम मार्केटिंग प्राइवेट लिमिटेड" into 15 single-character fragments instead of 4 tokens.
# That would inflate the posting count and corrupt every DF statistic below.
TOKEN_CLASS = r"\p{L}\p{N}\p{M}"
TOKENS_SQL = (
    r"regexp_split_to_array(trim(lower(regexp_replace(business_name, '[^" + TOKEN_CLASS + r"]+', ' ', 'g'))), ' ')"
)

SCRIPT_CASE = (
    r"CASE WHEN regexp_matches(token, '[\x{0900}-\x{097F}]') THEN 'Devanagari'"
    r" WHEN regexp_matches(token, '[\x{0980}-\x{0DFF}]') THEN 'Other Indic'"
    r" WHEN NOT regexp_matches(token, '[^\x{0020}-\x{024F}]') THEN 'Latin'"
    r" ELSE 'Other' END"
)


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def human(b: float) -> str:
    v = float(b)
    for u in ("B", "KB", "MB", "GB"):
        if v < 1024 or u == "GB":
            return f"{v:.1f} {u}"
        v /= 1024
    return f"{v:.1f} GB"


def n(v) -> str:
    return f"{v:,}"


# ------------------------------------------------------------------ memory probe


def peak_rss_bytes() -> int:
    """Peak RSS of this process. ru_maxrss is bytes on macOS, kilobytes on Linux."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if platform.system() == "Darwin" else raw * 1024


class RssSampler:
    """Samples RSS via `ps` so the report can show the shape of the curve, not just the
    peak. getrusage gives the authoritative maximum; this gives per-phase attribution."""

    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _read(self) -> int | None:
        try:
            out = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            return int(out) * 1024 if out else None  # ps reports KB
        except (subprocess.SubprocessError, ValueError, OSError):
            return None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            v = self._read()
            if v is not None:
                self.samples.append((time.perf_counter() - self._t0, v))

    def __enter__(self) -> RssSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def peak(self) -> int:
        return max((v for _, v in self.samples), default=0)

    def at(self, t: float) -> int:
        return max((v for ts, v in self.samples if ts <= t), default=0)


PHASES: list[tuple[str, float, int]] = []


@contextmanager
def phase(label: str, sampler: RssSampler):
    t0 = time.perf_counter()
    print(f"  [{label}] ...", flush=True)
    yield
    dt = time.perf_counter() - t0
    PHASES.append((label, dt, sampler.at(time.perf_counter() - sampler._t0)))
    print(f"  [{label}] {dt:.1f}s   rss now {human(sampler.peak())}", flush=True)


# ------------------------------------------------------------------------- build


def cmd_index(args) -> int:
    """Build the inverted index over S2+S3 names (measurement of build cost)."""
    src_dir, out_dir = resolve(args.in_dir), resolve(args.out)
    s2, s3 = f"{args.split}_source2", f"{args.split}_source3"
    for t in (s2, s3):
        if not (src_dir / f"{t}.parquet").exists():
            raise SystemExit(f"{src_dir / (t + '.parquet')} not found. Run src/convert.py first.")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_duckdb_tmp"
    tmp_dir.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"INVERTED INDEX BUILD — {args.split} S2+S3 business_name")
    print("=" * 78)
    print(f"  in    {src_dir}")
    print(f"  out   {out_dir}")
    print(f"  duckdb {duckdb.__version__}  threads 8  mem limit {os.environ.get('DUCKDB_MEM', '10GB')}")
    print(f"  NOTE  provisional tokenizer — no transliteration (stage 5 pending)")
    print()

    wall0 = time.perf_counter()
    with RssSampler() as sampler:
        con = duckdb.connect(config={
            "memory_limit": os.environ.get("DUCKDB_MEM", "10GB"),
            "threads": 8,
            "temp_directory": str(tmp_dir),
            # Nothing here depends on row order -- the index is written ORDER BY df --
            # and holding insertion order across a 35M-row explode costs real memory.
            "preserve_insertion_order": False,
        })
        con.execute(f"""CREATE VIEW src AS
            SELECT entity_id, business_name FROM read_parquet('{src_dir / (s2 + '.parquet')}')
            UNION ALL
            SELECT entity_id, business_name FROM read_parquet('{src_dir / (s3 + '.parquet')}')""")
        n_rec = con.execute("SELECT count(*) FROM src").fetchone()[0]
        print(f"  records to index: {n(n_rec)}")
        print()

        with phase("tokenize + explode -> postings", sampler):
            con.execute(f"""CREATE TABLE postings AS
                SELECT entity_id, token FROM (
                    SELECT entity_id, unnest({TOKENS_SQL}) AS token FROM src
                ) WHERE len(token) >= {args.min_token_len}""")
            n_post = con.execute("SELECT count(*) FROM postings").fetchone()[0]

        with phase("document frequency", sampler):
            con.execute("""CREATE TABLE df AS
                SELECT token, count(*) AS df FROM postings GROUP BY token""")
            n_tok = con.execute("SELECT count(*) FROM df").fetchone()[0]

        with phase("posting lists", sampler):
            con.execute("""CREATE TABLE idx AS
                SELECT token, count(*) AS df, list(entity_id) AS ids
                FROM postings GROUP BY token""")

        idx_path = out_dir / f"{args.split}_name_index.parquet"
        df_path = out_dir / f"{args.split}_name_df.parquet"
        with phase("write parquet", sampler):
            con.execute(f"COPY (SELECT * FROM idx ORDER BY df DESC) TO '{idx_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            con.execute(f"COPY (SELECT * FROM df ORDER BY df DESC) TO '{df_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")

        with phase("index statistics", sampler):
            q = con.execute("""SELECT min(df), quantile_cont(df,0.5), quantile_cont(df,0.9),
                quantile_cont(df,0.99), max(df), avg(df),
                count(*) FILTER (WHERE df=1), count(*) FILTER (WHERE df<=2) FROM df""").fetchone()
            top = con.execute("SELECT token, df FROM df ORDER BY df DESC LIMIT 15").fetchall()
            cutoffs = con.execute("""SELECT t.thr,
                    (SELECT count(*) FROM df WHERE df <= t.thr) AS tokens_kept,
                    (SELECT sum(df) FROM df WHERE df <= t.thr) AS postings_kept
                FROM (SELECT unnest([10,100,1000,10000,100000,1000000]) AS thr) t ORDER BY t.thr""").fetchall()
            scripts = con.execute(f"""SELECT {SCRIPT_CASE} AS s, count(*) AS tokens, sum(df) AS postings
                FROM df GROUP BY 1 ORDER BY postings DESC""").fetchall()

            # Reachability: a record is findable by name blocking only if it owns at
            # least one token rare enough to survive the IDF cutoff. Records whose
            # rarest token is still common have no usable key at all.
            con.execute("""CREATE TABLE rec_min AS
                SELECT p.entity_id, min(d.df) AS min_df
                FROM postings p JOIN df d USING (token) GROUP BY p.entity_id""")
            con.execute(rf"""CREATE TABLE rec_script AS
                SELECT r.entity_id, r.min_df,
                  CASE WHEN regexp_matches(s.business_name, '[\x{{0900}}-\x{{097F}}]') THEN 'Devanagari'
                       WHEN regexp_matches(s.business_name, '[\x{{0980}}-\x{{0DFF}}]') THEN 'Other Indic'
                       WHEN NOT regexp_matches(s.business_name, '[^\x{{0020}}-\x{{024F}}]') THEN 'Latin'
                       ELSE 'Other' END AS script
                FROM rec_min r JOIN src s USING (entity_id)""")
            reach = con.execute("""SELECT script, count(*) AS recs,
                    count(*) FILTER (WHERE min_df > 100)   AS no_100,
                    count(*) FILTER (WHERE min_df > 1000)  AS no_1k,
                    count(*) FILTER (WHERE min_df > 10000) AS no_10k
                FROM rec_script GROUP BY 1 ORDER BY recs DESC""").fetchall()

        wall = time.perf_counter() - wall0
        rusage_peak, sampled_peak = peak_rss_bytes(), sampler.peak()

    # ------------------------------------------------------------------ report
    print()
    print("-" * 78)
    print("BUILD TIME")
    print("-" * 78)
    print(f"  {'phase':<34}{'seconds':>10}{'share':>9}")
    print("  " + "-" * 51)
    for label, dt, _ in PHASES:
        print(f"  {label:<34}{dt:>10.1f}{100 * dt / wall:>8.1f}%")
    print("  " + "-" * 51)
    print(f"  {'TOTAL WALL CLOCK':<34}{wall:>10.1f}{100.0:>8.1f}%")
    print(f"  {'throughput':<34}{n_rec / wall:>10,.0f} records/s")

    print()
    print("-" * 78)
    print("PEAK MEMORY")
    print("-" * 78)
    print(f"  getrusage ru_maxrss (authoritative) : {human(rusage_peak)}")
    print(f"  ps sampler peak ({len(sampler.samples)} samples)        : {human(sampled_peak)}")
    print(f"  SageMaker ml.r7i.2xlarge budget      : 64.0 GB")
    print(f"  headroom                             : {64 - rusage_peak / 1024**3:.1f} GB "
          f"({100 * rusage_peak / (64 * 1024**3):.1f}% used)")

    print()
    print("-" * 78)
    print("INDEX SIZE ON DISK")
    print("-" * 78)
    tot = 0
    for p in (idx_path, df_path):
        tot += p.stat().st_size
        print(f"  {p.name:<34}{human(p.stat().st_size):>12}")
    src_bytes = sum((src_dir / f"{t}.parquet").stat().st_size for t in (s2, s3))
    print("  " + "-" * 46)
    print(f"  {'TOTAL':<34}{human(tot):>12}")
    print(f"  {'source parquet (S2+S3)':<34}{human(src_bytes):>12}")
    print(f"  {'index / source':<34}{tot / src_bytes:>11.2f}x")
    print(f"  {'bytes per posting':<34}{tot / n_post:>11.2f}")

    print()
    print("-" * 78)
    print("INDEX SHAPE")
    print("-" * 78)
    print(f"  records indexed        {n(n_rec)}")
    print(f"  postings               {n(n_post)}")
    print(f"  unique tokens          {n(n_tok)}")
    print(f"  mean tokens / record   {n_post / n_rec:.2f}")
    print(f"  mean postings / token  {n_post / n_tok:.2f}")
    print()
    print(f"  DF distribution: min {q[0]}  p50 {q[1]:.0f}  p90 {q[2]:.0f}  p99 {q[3]:.0f}  max {n(q[4])}  mean {q[5]:.1f}")
    print(f"  tokens with df=1       {n(q[6])}  ({100 * q[6] / n_tok:.1f}% of tokens — near-proof of a match)")
    print(f"  tokens with df<=2      {n(q[7])}  ({100 * q[7] / n_tok:.1f}%)")

    print()
    print("  Highest-DF tokens (these carry no signal and must be cut):")
    for tk, d in top:
        print(f"    {tk:<24}{n(d):>12}  ({100 * d / n_rec:.1f}% of records)")

    print()
    print("  IDF cutoff trade-off (input to the next step, not applied here):")
    print(f"    {'keep df <=':>12}{'tokens':>14}{'% tokens':>10}{'postings':>14}{'% postings':>12}")
    for thr, tk_kept, po_kept in cutoffs:
        po_kept = po_kept or 0
        print(f"    {n(thr):>12}{n(tk_kept):>14}{100 * tk_kept / n_tok:>9.1f}%{n(po_kept):>14}{100 * po_kept / n_post:>11.1f}%")

    print()
    print("  Token script mix (why stage 5 matters):")
    print(f"    {'script':<14}{'tokens':>12}{'% tokens':>10}{'postings':>14}{'% postings':>12}")
    for s, tk, po in scripts:
        print(f"    {s:<14}{n(tk):>12}{100 * tk / n_tok:>9.1f}%{n(po):>14}{100 * po / n_post:>11.1f}%")
    print()
    print("  Records with NO token rare enough to survive an IDF cutoff")
    print("  (a record with no rare token cannot be retrieved by name blocking at all):")
    print(f"    {'script':<14}{'records':>12}{'no tok df<=100':>18}{'no tok df<=1k':>18}{'no tok df<=10k':>18}")
    for sc, recs, a100, a1k, a10k in reach:
        cells = "".join(f"{n(v):>12} {100 * v / recs:>4.0f}%" for v in (a100, a1k, a10k))
        print(f"    {sc:<14}{n(recs):>12}{cells}")

    nonlatin = sum(po for s, _, po in scripts if s != "Latin")
    print()
    print(f"  {100 * nonlatin / n_post:.1f}% of postings are non-Latin. S1 is 100% Latin, so those")
    print("  postings are currently unreachable by any S1 query — stage 5 transliteration")
    print("  is a hard prerequisite for the >= 97% recall gate, not an optimisation.")

    (out_dir / f"_{args.split}_index_build.json").write_text(json.dumps({
        "split": args.split, "records": n_rec, "postings": n_post, "unique_tokens": n_tok,
        "wall_seconds": round(wall, 2), "phases": [{"phase": p, "seconds": round(d, 2)} for p, d, _ in PHASES],
        "peak_rss_bytes": rusage_peak, "peak_rss_gb": round(rusage_peak / 1024**3, 3),
        "index_bytes": tot, "source_bytes": src_bytes,
        "min_token_len": args.min_token_len, "token_class": TOKEN_CLASS,
        "transliteration": False, "duckdb": duckdb.__version__,
    }, indent=2) + "\n")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print()
    print(f"  wrote {out_dir / f'_{args.split}_index_build.json'}")
    return 0


# ===========================================================================
# Candidate generation
# ===========================================================================
#
# ADDRESS-FIRST, name secondary. This inverts the original stage 6 design, and the
# reason is in the data: business names are template-generated from small word pools.
# There are only 171 distinct Devanagari name tokens corpus-wide, and for every single
# ground-truth pair with an Indic match the S1 side's rarest name token still has
# df >= 8,928. Names carry little discriminative power; addresses carry most of it
# (82.2% of Indic-matched pairs have an S1 address token with df <= 1,000).
#
# Four block types, each emitting (s1, s23, block, score):
#   B1  postcode exact
#   B2  street-number + rare address token
#   B3  rare address token pair
#   B4  rare name token pair
# Score is the summed IDF of the key tokens. Per pair, the max score within each block
# type is summed across block types -- so a pair found several ways ranks higher without
# double-counting the same token inside one block.

# Caps chosen by measuring recall-per-pair for each block independently (see
# docs/PLAN.md stage 6). The cross-field composite B7 is the best value in the set:
# 64.13% recall alone for 49.6M pairs, versus 62.98% for 825.7M from the address pair
# block at a wide cap -- 16x the cost for less recall.
# Overridable from the environment so a sweep does not need a source edit. The
# defaults are the measured baseline operating point -- unset env == baseline.
def _envint(k: str, d: int) -> int:
    return int(os.environ.get(k, d))


DF_CAP_PAIR = _envint("DF_CAP_PAIR", 5_000)        # rarity for the within-field pair blocks
K_PAIR = _envint("K_PAIR", 8)                      # rarest tokens kept per record for pairs
DF_CAP_SINGLE = _envint("DF_CAP_SINGLE", 200)      # a single shared token this rare is evidence alone
DF_CAP_COMPOSITE = _envint("DF_CAP_COMPOSITE", 20_000)  # wider: the composite key is selective by construction
K_COMPOSITE = _envint("K_COMPOSITE", 4)
CAPS = (10, 20, 30, 50, 75, 100)
NPART = _envint("NPART", 8)          # S1-hash partitions for the union group-by
PRE_CAP = _envint("PRE_CAP", 100)    # candidates kept per S1 BEFORE re-ranking. Not a df-cap change:
                        # the blocks are unchanged, this only keeps more of what they
                        # already produced so the Jaccard re-rank has room to reorder.
DEFAULT_CAP = 30

BLOCKS = ["B1_postcode", "B2_number_addrtok", "B3_addr_pair", "B4_name_pair",
          "B5_addr_single", "B6_name_single", "B7_addr_x_name"]

# ---------------------------------------------------------------- Exp 1 blocks
# Exp 1 measured the 831,258 GT pairs the uncapped union never generates. Not one of
# them shares a token below df=200; the mass sits at df 5k-100k, because names and
# addresses are drawn from small word pools and no single token is ever rare. The three
# rules below address that directly. All are OFF by default so the baseline path is
# byte-identical with no environment set.

# B8/B9 -- the whole token set as one key. Measured +4.86pp recovery, and the only rule
# that reaches the rows whose S2/S3 address was deleted outright (66.9% of the residue).
ENABLE_SETKEY = os.environ.get("ENABLE_SETKEY", "0") == "1"
KEY_DF_CAP = _envint("KEY_DF_CAP", 1_000)   # on the KEY, not on its tokens
KEY_MIN_TOK = _envint("KEY_MIN_TOK", 2)     # a 1-token key is just an uncapped single block

# B10 -- pairs budgeted on the EXPECTED INTERSECTION df1*df2/N rather than on each df.
# The current df<=5,000-per-token rule forbids a 20k x 20k pair (expected intersection
# 39 records) while admitting 5k x 5k (expected 2.4). Measured +5.93pp at N*50.
ENABLE_PRODPAIR = os.environ.get("ENABLE_PRODPAIR", "0") == "1"
PROD_BUDGET = _envint("PROD_BUDGET", 50)    # in multiples of N: df1*df2 <= PROD_BUDGET*N
K_PROD = _envint("K_PROD", 8)               # rarest tokens per record entering the pairing

# Street numbers compared raw miss 5110/005110, 7/07, 1223/1223-1225. Measured +0.83pp.
ENABLE_NUMNORM = os.environ.get("ENABLE_NUMNORM", "0") == "1"

if ENABLE_SETKEY:
    BLOCKS += ["B8_name_set", "B9_addr_set"]
if ENABLE_PRODPAIR:
    BLOCKS += ["B10_prod_pair"]


def _numlist(col: str) -> str:
    """Street numbers, optionally normalised: ranges split on - and /, zeros stripped."""
    if not ENABLE_NUMNORM:
        return col
    return (f"list_distinct(list_filter(list_transform(flatten(list_transform({col}, "
            f"x -> str_split(regexp_replace(x, '[^0-9/-]', '', 'g'), '-'))), "
            f"z -> ltrim(z, '0')), y -> y <> ''))")


def _idf(df_expr: str, n: int) -> str:
    return f"ln({n}.0 / {df_expr})"


def cmd_candidates(args) -> int:
    norm_dir = resolve(args.norm_dir)
    out_dir = resolve(args.out)
    for s in ("1", "2", "3"):
        f = norm_dir / f"{args.split}_source{s}.parquet"
        if not f.exists():
            raise SystemExit(f"{f} not found. Run src/normalize.py --apply first.")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_duckdb_tmp"
    tmp_dir.mkdir(exist_ok=True)

    print("=" * 84)
    print(f"CANDIDATE GENERATION — {args.split}   cap={args.cap}")
    print(f"  pair blocks df<={DF_CAP_PAIR:,} K={K_PAIR} | single df<={DF_CAP_SINGLE} | "
          f"composite df<={DF_CAP_COMPOSITE:,} K={K_COMPOSITE}")
    print("=" * 84)
    print(f"  normalized  {norm_dir}")
    print(f"  out         {out_dir}")
    print("  design      ADDRESS-FIRST (names are template-generated; see module notes)")
    print()

    wall0 = time.perf_counter()
    with RssSampler() as sampler:
        con = duckdb.connect(config={
            "memory_limit": os.environ.get("DUCKDB_MEM", "9GB"),
            "threads": 8,
            "temp_directory": str(tmp_dir),
            "preserve_insertion_order": False,
        })
        # --s1-ids restricts the S1 side to a subset. Blocking is exactly per-S1
        # independent -- the DF tables are built from S2+S3 only, and ranking/capping
        # partition by S1 -- so the candidates produced for a subset are IDENTICAL to
        # those a full run produces for the same entities. Restricting to the evaluation
        # entities (val fold + the train-fold sample the model is fitted on) cuts this
        # stage ~3.5x with no approximation.
        s1_src = f"read_parquet('{norm_dir / (args.split + '_source1.parquet')}')"
        if getattr(args, "s1_ids", None):
            s1_src = (f"(SELECT f.* FROM {s1_src} f SEMI JOIN "
                      f"read_parquet('{resolve(args.s1_ids)}') k ON k.entity_id = f.entity_id)")
            print(f"  S1 restricted to {args.s1_ids}")
        con.execute(f"CREATE VIEW s1 AS SELECT * FROM {s1_src}")
        con.execute(f"""CREATE VIEW s23 AS
            SELECT * FROM read_parquet('{norm_dir / (args.split + '_source2.parquet')}')
            UNION ALL
            SELECT * FROM read_parquet('{norm_dir / (args.split + '_source3.parquet')}')""")
        n1 = con.execute("SELECT count(*) FROM s1").fetchone()[0]
        n23 = con.execute("SELECT count(*) FROM s23").fetchone()[0]
        print(f"  S1 {n1:,}   S2+S3 {n23:,}")
        print()

        with phase("token tables + DF", sampler):
            for side in ("s1", "s23"):
                con.execute(f"""CREATE TABLE {side}_nt AS SELECT DISTINCT entity_id, token FROM
                    (SELECT entity_id, unnest(name_tokens) token FROM {side}) WHERE len(token) >= 2""")
                con.execute(f"""CREATE TABLE {side}_at AS SELECT DISTINCT entity_id, token FROM
                    (SELECT entity_id, unnest(addr_tokens) token FROM {side}) WHERE len(token) >= 2""")
            con.execute("CREATE TABLE dfn AS SELECT token, count(*) df FROM s23_nt GROUP BY 1")
            con.execute("CREATE TABLE dfa AS SELECT token, count(*) df FROM s23_at GROUP BY 1")
            con.execute("CREATE TABLE dfp AS SELECT postcode, count(*) df FROM s23 WHERE postcode <> '' GROUP BY 1")

        with phase("rarest-token tables", sampler):
            for side in ("s1", "s23"):
                for kind, dft in (("nt", "dfn"), ("at", "dfa")):
                    for tag, cap, k in (("pair", DF_CAP_PAIR, K_PAIR),
                                        ("comp", DF_CAP_COMPOSITE, K_COMPOSITE)):
                        con.execute(f"""CREATE TABLE r{tag}_{side}_{kind} AS
                            SELECT entity_id, token, df FROM (
                              SELECT x.entity_id, x.token, d.df,
                                     row_number() OVER (PARTITION BY x.entity_id ORDER BY d.df, x.token) rk
                              FROM {side}_{kind} x JOIN {dft} d USING (token)
                              WHERE d.df <= {cap}
                            ) WHERE rk <= {k}""")
                    con.execute(f"""CREATE TABLE rsing_{side}_{kind} AS
                        SELECT x.entity_id, x.token, d.df FROM {side}_{kind} x
                        JOIN {dft} d USING (token) WHERE d.df <= {DF_CAP_SINGLE}""")

        paths = {}

        def emit(block: str, sql: str) -> None:
            paths[block] = out_dir / f"blk_{block}.parquet"
            con.execute(f"COPY ({sql}) TO '{paths[block]}' (FORMAT PARQUET, COMPRESSION ZSTD)")

        # ---- B1: postcode exact -------------------------------------------------
        with phase("B1 postcode exact", sampler):
            emit("B1_postcode", f"""
                SELECT a.entity_id AS s1, b.entity_id AS s23, 'B1_postcode' AS block,
                       {_idf('p.df', n23)} AS score
                FROM s1 a JOIN s23 b ON a.postcode = b.postcode
                JOIN dfp p ON p.postcode = a.postcode
                WHERE a.postcode <> ''""")

        # ---- B2: street number + rare address token -----------------------------
        with phase("B2 street-number + rare addr token", sampler):
            for side in ("s1", "s23"):
                con.execute(f"""CREATE TABLE b2_{side} AS SELECT DISTINCT
                    x.entity_id, x.num, r.token, r.df
                    FROM (SELECT entity_id, unnest({_numlist('addr_numbers')}) num FROM {side}) x
                    JOIN rpair_{side}_at r ON r.entity_id = x.entity_id""")
            emit("B2_number_addrtok", f"""
                SELECT a.entity_id AS s1, b.entity_id AS s23, 'B2_number_addrtok' AS block,
                       max({_idf('a.df', n23)}) AS score
                FROM b2_s1 a JOIN b2_s23 b ON a.num = b.num AND a.token = b.token
                GROUP BY 1, 2, 3""")

        # ---- B3 / B4: within-field rare token pairs ------------------------------
        for kind, block in (("at", "B3_addr_pair"), ("nt", "B4_name_pair")):
            with phase(f"{block}", sampler):
                for side in ("s1", "s23"):
                    con.execute(f"""CREATE TABLE pp_{side}_{kind} AS
                        SELECT a.entity_id, a.token t1, b.token t2, a.df d1, b.df d2
                        FROM rpair_{side}_{kind} a JOIN rpair_{side}_{kind} b
                          ON a.entity_id = b.entity_id AND a.token < b.token""")
                emit(block, f"""
                    SELECT a.entity_id AS s1, b.entity_id AS s23, '{block}' AS block,
                           max({_idf('a.d1', n23)} + {_idf('a.d2', n23)}) AS score
                    FROM pp_s1_{kind} a JOIN pp_s23_{kind} b ON a.t1 = b.t1 AND a.t2 = b.t2
                    GROUP BY 1, 2, 3""")
                con.execute(f"DROP TABLE pp_s1_{kind}"); con.execute(f"DROP TABLE pp_s23_{kind}")

        # ---- B5 / B6: a single very rare shared token ----------------------------
        # 30% of the v1 misses shared exactly ONE address token below the pair cap, and
        # 44% shared exactly one name token -- a pair rule can never reach those.
        for kind, block in (("at", "B5_addr_single"), ("nt", "B6_name_single")):
            with phase(f"{block} (df<={DF_CAP_SINGLE})", sampler):
                emit(block, f"""
                    SELECT a.entity_id AS s1, b.entity_id AS s23, '{block}' AS block,
                           max({_idf('a.df', n23)}) AS score
                    FROM rsing_s1_{kind} a JOIN rsing_s23_{kind} b ON a.token = b.token
                    GROUP BY 1, 2, 3""")

        # ---- B7: cross-field composite, one address token x one name token -------
        # The best recall-per-pair in the set. Catches pairs where neither field alone
        # has a token rare enough to block on, but the combination is selective.
        with phase("B7 addr-token x name-token composite", sampler):
            for side in ("s1", "s23"):
                con.execute(f"""CREATE TABLE x_{side} AS SELECT DISTINCT
                    a.entity_id, a.token at_, n.token nt_, a.df ad, n.df nd
                    FROM rcomp_{side}_at a JOIN rcomp_{side}_nt n ON n.entity_id = a.entity_id""")
            emit("B7_addr_x_name", f"""
                SELECT a.entity_id AS s1, b.entity_id AS s23, 'B7_addr_x_name' AS block,
                       max({_idf('a.ad', n23)} + {_idf('a.nd', n23)}) AS score
                FROM x_s1 a JOIN x_s23 b ON a.at_ = b.at_ AND a.nt_ = b.nt_
                GROUP BY 1, 2, 3""")
            con.execute("DROP TABLE x_s1"); con.execute("DROP TABLE x_s23")

        # ---- B8/B9: the whole token set as one key ------------------------------
        if ENABLE_SETKEY:
            for kind, fld, block in (("nt", "name_tokens", "B8_name_set"),
                                     ("at", "addr_tokens", "B9_addr_set")):
                with phase(f"{block} (key df<={KEY_DF_CAP}, >={KEY_MIN_TOK} tokens)", sampler):
                    for side in ("s1", "s23"):
                        con.execute(f"""CREATE OR REPLACE TABLE ks_{side}_{kind} AS
                            SELECT entity_id, array_to_string(t, '\x1f') AS k FROM (
                              SELECT entity_id,
                                     list_sort(list_distinct(list_filter({fld}, x -> len(x) >= 2))) AS t
                              FROM {side})
                            WHERE len(t) >= {KEY_MIN_TOK}""")
                    con.execute(f"""CREATE OR REPLACE TABLE kdf_{kind} AS
                        SELECT k, count(*) df FROM ks_s23_{kind} GROUP BY 1""")
                    emit(block, f"""
                        SELECT a.entity_id AS s1, b.entity_id AS s23, '{block}' AS block,
                               {_idf('d.df', n23)} AS score
                        FROM ks_s1_{kind} a JOIN ks_s23_{kind} b ON a.k = b.k
                        JOIN kdf_{kind} d ON d.k = a.k
                        WHERE d.df <= {KEY_DF_CAP}""")
                    for side in ("s1", "s23"):
                        con.execute(f"DROP TABLE IF EXISTS ks_{side}_{kind}")

        # ---- B10: product-DF pair over the combined name+addr+number namespace ---
        if ENABLE_PRODPAIR:
            with phase(f"B10 product pair (df1*df2 <= {PROD_BUDGET}N, K={K_PROD})", sampler):
                for side in ("s1", "s23"):
                    con.execute(f"""CREATE OR REPLACE TABLE ctok_{side} AS
                        SELECT DISTINCT entity_id, token FROM (
                          SELECT entity_id, unnest(list_concat(
                              list_concat(name_tokens, addr_tokens),
                              {_numlist('addr_numbers')})) token
                          FROM {side})
                        WHERE len(token) >= 2""")
                con.execute("""CREATE OR REPLACE TABLE dfc AS
                    SELECT token, count(*) df FROM ctok_s23 GROUP BY 1""")
                for side in ("s1", "s23"):
                    con.execute(f"""CREATE OR REPLACE TABLE rprod_{side} AS
                        SELECT entity_id, token, df FROM (
                          SELECT x.entity_id, x.token, d.df,
                                 row_number() OVER (PARTITION BY x.entity_id
                                                    ORDER BY d.df, x.token) rk
                          FROM ctok_{side} x JOIN dfc d USING (token)
                        ) WHERE rk <= {K_PROD}""")
                    con.execute(f"DROP TABLE IF EXISTS ctok_{side}")
                    con.execute(f"""CREATE OR REPLACE TABLE pprod_{side} AS
                        SELECT a.entity_id, a.token t1, b.token t2, a.df d1, b.df d2
                        FROM rprod_{side} a JOIN rprod_{side} b
                          ON a.entity_id = b.entity_id AND a.token < b.token
                        WHERE a.df::BIGINT * b.df <= {PROD_BUDGET}::BIGINT * {n23}""")
                emit("B10_prod_pair", f"""
                    SELECT a.entity_id AS s1, b.entity_id AS s23, 'B10_prod_pair' AS block,
                           max({_idf('a.d1', n23)} + {_idf('a.d2', n23)}) AS score
                    FROM pprod_s1 a JOIN pprod_s23 b ON a.t1 = b.t1 AND a.t2 = b.t2
                    GROUP BY 1, 2, 3""")
                for side in ("s1", "s23"):
                    con.execute(f"DROP TABLE IF EXISTS pprod_{side}")
                    con.execute(f"DROP TABLE IF EXISTS rprod_{side}")

        # ---- union, score, rank (partitioned by S1 hash) ----------------------
        # 405M block rows keyed on two entity-id strings overflowed 35 GiB of spill when
        # grouped in one pass. Partitioning on hash(s1) keeps each group-by bounded and
        # is exact: a given S1 lives in exactly one partition, so per-S1 ranking and
        # distinct counts are unaffected by the split.
        with phase(f"union + score + rank ({NPART} partitions)", sampler):
            glob = str(out_dir / "blk_*.parquet")
            n_cand = 0
            # One pass per partition: the grouped pair set is materialised, counted
            # (that count IS the uncapped distinct-pair total for the partition), then
            # ranked and capped. Counting it with a second scan of the 373M-row glob
            # doubled this phase's runtime for a number already in hand.
            for pt in range(NPART):
                dst = out_dir / f"ranked_part{pt}.parquet"
                con.execute(f"""CREATE OR REPLACE TEMP TABLE upair AS
                    SELECT s1, s23, sum(score) AS score, count(*) AS n_blocks
                    FROM (SELECT s1, s23, block, max(score) AS score
                          FROM read_parquet('{glob}')
                          WHERE hash(s1) % {NPART} = {pt}
                          GROUP BY 1, 2, 3)
                    GROUP BY 1, 2""")
                n_cand += con.execute("SELECT count(*) FROM upair").fetchone()[0]
                con.execute(f"""COPY (
                    SELECT s1, s23, score, n_blocks, rk FROM (
                      SELECT s1, s23, score, n_blocks,
                             row_number() OVER (PARTITION BY s1 ORDER BY score DESC, s23) AS rk
                      FROM upair)
                    WHERE rk <= {PRE_CAP}
                ) TO '{dst}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
                con.execute("DROP TABLE IF EXISTS upair")
            con.execute(f"""CREATE VIEW ranked AS
                SELECT * FROM read_parquet('{out_dir / 'ranked_part*.parquet'}')""")

        # ---- ground truth -------------------------------------------------------
        gt_path = resolve(args.parquet_dir) / f"{args.split}_ground_truth.parquet"
        has_gt = gt_path.exists()
        with phase("recall vs ground truth" if has_gt else "skip recall (no labels)", sampler):
            if not has_gt:
                con.execute("CREATE TABLE gtx (s1 VARCHAR, s23 VARCHAR, script VARCHAR, country VARCHAR)")
                con.execute("CREATE TABLE gt_union (s1 VARCHAR, s23 VARCHAR)")
                n_gt = 0
            if has_gt:
              con.execute(f"""CREATE TABLE gp AS SELECT source1_entity_id AS s1, trim(m) AS s23
                FROM read_parquet('{gt_path}'), unnest(str_split(matched_entity_ids, ',')) AS t(m)
                WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids) <> ''""")
              con.execute(f"""CREATE TABLE gtx AS SELECT g.s1, g.s23,
                  CASE WHEN b.name_script = 'latin' THEN 'Latin' ELSE b.name_script END AS script,
                  a.country AS country
                FROM gp g JOIN s23 b ON b.entity_id = g.s23 JOIN s1 a ON a.entity_id = g.s1""")
              n_gt = con.execute("SELECT count(*) FROM gtx").fetchone()[0]
            # GT pairs present anywhere in the uncapped union. Semi-join with the small
            # side (7.6M GT rows) as the hash build, streaming the 405M block rows.
              con.execute(f"""CREATE TABLE gt_union AS SELECT g.s1, g.s23 FROM gtx g
                SEMI JOIN read_parquet('{str(out_dir / "blk_*.parquet")}') c
                ON c.s1 = g.s1 AND c.s23 = g.s23""")
            del gt_path

        wall = time.perf_counter() - wall0
        peak = peak_rss_bytes()

        # ================================ report ================================
        gate = True
        recalls = {c: 0.0 for c in CAPS}
        if not has_gt:
            print()
            print("  (no ground truth for this split — recall sections skipped)")
        print()
        print("-" * 84)
        print("BLOCK SIZES AND PER-BLOCK RECALL (uncapped — where the recall comes from)")
        print("-" * 84)
        print(f"  {'block':<22}{'pairs':>16}{'GT pairs found':>17}{'recall':>10}")
        print("  " + "-" * 65)
        for b in (BLOCKS if has_gt else ()):
            pr = con.execute(f"SELECT count(*) FROM read_parquet('{paths[b]}')").fetchone()[0]
            hit = con.execute(f"""SELECT count(*) FROM gtx g SEMI JOIN
                (SELECT DISTINCT s1, s23 FROM read_parquet('{paths[b]}')) c
                ON c.s1 = g.s1 AND c.s23 = g.s23""").fetchone()[0]
            print(f"  {b:<22}{pr:>16,}{hit:>17,}{100 * hit / n_gt:>9.2f}%")
        print("  " + "-" * 65)
        print(f"  {'UNION (uncapped)':<22}{n_cand:>16,}", end="")
        un = con.execute("SELECT count(*) FROM gt_union").fetchone()[0] if has_gt else 0
        print(f"{un:>17,}{(100 * un / n_gt) if n_gt else 0:>9.2f}%")

        print()
        print("-" * 84)
        print("RECALL BY CANDIDATE CAP")
        print("-" * 84)
        print(f"  {'cap':>6}{'candidates':>16}{'GT found':>14}{'recall':>10}{'mean/S1':>10}{'p99/S1':>9}{'zero-cand S1':>15}")
        print("  " + "-" * 78)
        for cap in (CAPS if has_gt else ()):
            r = con.execute(f"""SELECT count(*) FROM gtx g SEMI JOIN
                (SELECT s1, s23 FROM ranked WHERE rk <= {cap}) c ON c.s1=g.s1 AND c.s23=g.s23""").fetchone()[0]
            tot = con.execute(f"SELECT count(*) FROM ranked WHERE rk <= {cap}").fetchone()[0]
            st = con.execute(f"""SELECT avg(k), quantile_cont(k, 0.99) FROM
                (SELECT s1, count(*) k FROM ranked WHERE rk <= {cap} GROUP BY 1)""").fetchone()
            withc = con.execute(f"SELECT count(DISTINCT s1) FROM ranked WHERE rk <= {cap}").fetchone()[0]
            recalls[cap] = r / n_gt
            print(f"  {cap:>6}{tot:>16,}{r:>14,}{100 * r / n_gt:>9.2f}%"
                  f"{st[0]:>10.1f}{st[1]:>9.0f}{n1 - withc:>10,} {100 * (n1 - withc) / n1:>3.1f}%")

        cap = args.cap
        if not has_gt:
            for c2 in CAPS:
                tot2 = con.execute(f"SELECT count(*) FROM ranked WHERE rk <= {c2}").fetchone()[0]
                print(f"  {c2:>6}{tot2:>16,}{'-':>14}{'-':>10}")
        print()
        print("-" * 84)
        print(f"PER-SCRIPT RECALL (script of the matched S2/S3 record)   cap={cap}")
        print("-" * 84)
        print(f"  {'script':<16}{'GT pairs':>14}{'uncapped':>12}{'capped':>12}")
        print("  " + "-" * 52)
        for sc, tot, unc, cpd in (con.execute(f"""
            SELECT g.script, count(*),
              count(*) FILTER (WHERE c.s1 IS NOT NULL),
              count(*) FILTER (WHERE r.s1 IS NOT NULL)
            FROM gtx g
            LEFT JOIN gt_union c ON c.s1=g.s1 AND c.s23=g.s23
            LEFT JOIN (SELECT s1, s23 FROM ranked WHERE rk <= {cap}) r ON r.s1=g.s1 AND r.s23=g.s23
            GROUP BY 1 ORDER BY count(*) DESC""").fetchall() if has_gt else ()):
            print(f"  {sc:<16}{tot:>14,}{100 * unc / tot:>11.2f}%{100 * cpd / tot:>11.2f}%")

        print()
        print("-" * 84)
        print(f"PER-COUNTRY RECALL (country of the S1 record)   cap={cap}")
        print("-" * 84)
        print(f"  {'country':<16}{'GT pairs':>14}{'uncapped':>12}{'capped':>12}")
        print("  " + "-" * 52)
        for ctry, tot, unc, cpd in (con.execute(f"""
            SELECT g.country, count(*),
              count(*) FILTER (WHERE c.s1 IS NOT NULL),
              count(*) FILTER (WHERE r.s1 IS NOT NULL)
            FROM gtx g
            LEFT JOIN gt_union c ON c.s1=g.s1 AND c.s23=g.s23
            LEFT JOIN (SELECT s1, s23 FROM ranked WHERE rk <= {cap}) r ON r.s1=g.s1 AND r.s23=g.s23
            GROUP BY 1 ORDER BY count(*) DESC""").fetchall() if has_gt else ()):
            print(f"  {str(ctry):<16}{tot:>14,}{100 * unc / tot:>11.2f}%{100 * cpd / tot:>11.2f}%")

        print()
        print("-" * 84)
        print("COST")
        print("-" * 84)
        kept = con.execute(f"SELECT count(*) FROM ranked WHERE rk <= {cap}").fetchone()[0]
        full = n1 * n23
        print(f"  full cross product        {full:,.0f}")
        print(f"  candidates kept (cap {cap})   {kept:,}")
        print(f"  reduction ratio           {1 - kept / full:.10f}")
        print(f"  pairs per S1 (mean)       {kept / n1:.1f}")
        print(f"  wall clock                {wall:.1f}s")
        print(f"  peak RSS                  {human(peak)}")

        final = out_dir / f"{args.split}_candidates_cap{cap}.parquet"
        con.execute(f"""COPY (SELECT s1, s23, score, n_blocks FROM ranked WHERE rk <= {cap}
                            ORDER BY s1, rk) TO '{final}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
        print(f"\n  wrote {final.name}  ({final.stat().st_size / 1e6:.1f} MB)")

        if has_gt:
            gate = recalls[cap] >= 0.97
            print()
            print("=" * 84)
            print(f"GATE (>= 97% overall recall at cap {cap}): "
                  f"{'PASS' if gate else 'FAIL'} — {100 * recalls[cap]:.2f}%")
            print("=" * 84)

        (out_dir / f"_{args.split}_candidates.json").write_text(json.dumps({
            "split": args.split, "df_cap_pair": DF_CAP_PAIR, "k_pair": K_PAIR,
            "df_cap_single": DF_CAP_SINGLE, "df_cap_composite": DF_CAP_COMPOSITE, "cap": cap,
            "setkey": ENABLE_SETKEY, "key_df_cap": KEY_DF_CAP, "key_min_tok": KEY_MIN_TOK,
            "prodpair": ENABLE_PRODPAIR, "prod_budget": PROD_BUDGET, "k_prod": K_PROD,
            "numnorm": ENABLE_NUMNORM, "blocks": list(BLOCKS),
            "s1": n1, "s23": n23, "gt_pairs": n_gt, "candidates_uncapped": n_cand,
            "candidates_capped": kept, "recall_by_cap": {str(k): v for k, v in recalls.items()},
            "reduction_ratio": 1 - kept / full, "wall_seconds": round(wall, 1),
            "peak_rss_gb": round(peak / 1024**3, 3), "gate_pass": gate,
        }, indent=2) + "\n")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0 if gate else 1



def cmd_rerank(args) -> int:
    """Re-rank the pre-capped candidates by token-set Jaccard.

    Two variants are scored and the better one wins:
      plain — |A n B| / |A u B| over the combined name+address token sets
      idf   — IDF-weighted: sum(idf over intersection) / sum(idf over union)

    Only the intersection is unnested. sum_idf(A u B) = sum_idf(A) + sum_idf(B) -
    sum_idf(A n B), and the per-record sums are precomputable, so the 130M-row pair set
    never explodes its full union.
    """
    norm_dir, out_dir = resolve(args.norm_dir), resolve(args.out)
    if not sorted(out_dir.glob("ranked_part*.parquet")):
        raise SystemExit(f"no ranked_part*.parquet in {out_dir}; run `candidates` first.")

    print("=" * 84)
    print(f"JACCARD RE-RANK — {args.split}")
    print("=" * 84)
    t0 = time.perf_counter()
    with RssSampler() as sampler:
        con = duckdb.connect(config={
            "memory_limit": os.environ.get("DUCKDB_MEM", "9GB"), "threads": 8,
            "temp_directory": str(out_dir / "_duckdb_tmp"), "preserve_insertion_order": False})
        r_s1 = f"read_parquet('{norm_dir / (args.split + chr(95) + 'source1.parquet')}')"
        if getattr(args, "s1_ids", None):
            r_s1 = (f"(SELECT f.* FROM {r_s1} f SEMI JOIN "
                    f"read_parquet('{resolve(args.s1_ids)}') k ON k.entity_id = f.entity_id)")
        con.execute(f"CREATE VIEW s1 AS SELECT * FROM {r_s1}")
        con.execute(f"""CREATE VIEW s23 AS
            SELECT * FROM read_parquet('{norm_dir / (args.split + chr(95) + 'source2.parquet')}')
            UNION ALL SELECT * FROM read_parquet('{norm_dir / (args.split + chr(95) + 'source3.parquet')}')""")
        n23 = con.execute("SELECT count(*) FROM s23").fetchone()[0]

        with phase("token sets + IDF", sampler):
            for side in ("s1", "s23"):
                con.execute(f"""CREATE TABLE tk_{side} AS SELECT entity_id,
                    list_distinct(list_concat(name_tokens, addr_tokens)) AS tok FROM {side}""")
            con.execute("""CREATE TABLE idf0 AS SELECT tok AS token, count(*) df
                FROM (SELECT unnest(tok) tok FROM tk_s23) GROUP BY 1""")
            con.execute(f"CREATE TABLE idf AS SELECT token, ln({n23}.0/df) AS w FROM idf0")
            for side in ("s1", "s23"):
                con.execute(f"""CREATE TABLE w_{side} AS
                    SELECT e.entity_id, coalesce(sum(i.w), 0.0) AS wsum
                    FROM (SELECT entity_id, unnest(tok) AS token FROM tk_{side}) e
                    LEFT JOIN idf i USING (token) GROUP BY 1""")

        # Partitioned exactly like the union: materialising 130M rows that carry list
        # columns overflowed 40 GiB of spill in one pass.
        with phase(f"jaccard variants ({NPART} partitions)", sampler):
            src = str(out_dir / "ranked_part*.parquet")
            for pt in range(NPART):
                con.execute(f"""CREATE OR REPLACE TEMP TABLE ip AS
                    SELECT r.s1, r.s23, unnest(list_intersect(a.tok, b.tok)) AS token
                    FROM read_parquet('{src}') r
                    JOIN tk_s1 a ON a.entity_id = r.s1 JOIN tk_s23 b ON b.entity_id = r.s23
                    WHERE hash(r.s1) % {NPART} = {pt}""")
                con.execute("""CREATE OR REPLACE TEMP TABLE wi AS
                    SELECT s1, s23, coalesce(sum(i.w), 0.0) AS w_inter
                    FROM ip LEFT JOIN idf i USING (token) GROUP BY 1, 2""")
                con.execute(f"""COPY (
                    SELECT r.s1, r.s23, r.score AS blk_score, r.n_blocks, r.rk AS blk_rank,
                      len(list_intersect(a.tok, b.tok))::DOUBLE
                        / nullif(len(list_distinct(list_concat(a.tok, b.tok))), 0) AS jac_plain,
                      coalesce(w.w_inter, 0.0)
                        / nullif(wa.wsum + wb.wsum - coalesce(w.w_inter, 0.0), 0) AS jac_idf
                    FROM read_parquet('{src}') r
                    JOIN tk_s1 a ON a.entity_id = r.s1 JOIN tk_s23 b ON b.entity_id = r.s23
                    JOIN w_s1 wa ON wa.entity_id = r.s1 JOIN w_s23 wb ON wb.entity_id = r.s23
                    LEFT JOIN wi w ON w.s1 = r.s1 AND w.s23 = r.s23
                    WHERE hash(r.s1) % {NPART} = {pt}
                ) TO '{out_dir / 'sc_part'}{pt}.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)""")
                con.execute("DROP TABLE IF EXISTS ip"); con.execute("DROP TABLE IF EXISTS wi")
            con.execute(f"""CREATE VIEW sc AS
                SELECT * FROM read_parquet('{out_dir / 'sc_part*.parquet'}')""")

        gt = resolve(args.parquet_dir) / f"{args.split}_ground_truth.parquet"
        if not gt.exists():
            # No labels: use the ordering that won on train.
            order = "jac_idf DESC, blk_score DESC, s23"
            print(f"\n  no ground truth for {args.split}; applying the train winner: "
                  f"jaccard IDF-weighted")
            dst = out_dir / f"{args.split}_reranked_cap{args.cap}.parquet"
            con.execute(f"""COPY (SELECT s1, s23, blk_score, n_blocks, blk_rank, jac_plain, jac_idf, rk
                FROM (SELECT s.*, row_number() OVER (PARTITION BY s.s1 ORDER BY {order}) AS rk FROM sc s)
                WHERE rk <= {args.cap} ORDER BY s1, rk) TO '{dst}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
            print(f"  wrote {dst.name} ({dst.stat().st_size/1e6:.1f} MB) in "
                  f"{time.perf_counter()-t0:.0f}s, peak RSS {human(peak_rss_bytes())}")
            return 0
        # denominator must follow the S1 restriction, or recall is measured against
        # ground truth for entities that were never a candidate-generation input.
        con.execute(f"""CREATE TABLE gp AS SELECT g.s1, g.s23 FROM (
              SELECT source1_entity_id s1, trim(m) s23
              FROM read_parquet('{gt}'), unnest(str_split(matched_entity_ids, ',')) t(m)
              WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids) <> ''
            ) g SEMI JOIN s1 a ON a.entity_id = g.s1""")
        n_gt = con.execute("SELECT count(*) FROM gp").fetchone()[0]

        variants = {
            "blocking IDF score (baseline)": "blk_score DESC, s23",
            "jaccard plain": "jac_plain DESC, blk_score DESC, s23",
            "jaccard IDF-weighted": "jac_idf DESC, blk_score DESC, s23",
        }
        print()
        print(f"  {'ranking':<32}" + "".join(f"{'@' + str(c):>10}" for c in CAPS))
        print("  " + "-" * (32 + 10 * len(CAPS)))
        best, best_r = None, -1.0
        for label, order in variants.items():
            con.execute(f"""CREATE OR REPLACE TABLE rk AS SELECT s1, s23,
                row_number() OVER (PARTITION BY s1 ORDER BY {order}) AS rk FROM sc""")
            row = []
            for cap in CAPS:
                h = con.execute(f"""SELECT count(*) FROM gp g SEMI JOIN
                    (SELECT s1, s23 FROM rk WHERE rk <= {cap}) c
                    ON c.s1=g.s1 AND c.s23=g.s23""").fetchone()[0]
                row.append(h / n_gt)
            print(f"  {label:<32}" + "".join(f"{100*v:>9.2f}%" for v in row))
            if row[CAPS.index(args.cap)] > best_r:
                best, best_r = (label, order), row[CAPS.index(args.cap)]
        print()
        print(f"  WINNER at cap {args.cap}: {best[0]}  ({100*best_r:.2f}%)")

        dst = out_dir / f"{args.split}_reranked_cap{args.cap}.parquet"
        con.execute(f"""COPY (SELECT s1, s23, blk_score, n_blocks, blk_rank, jac_plain, jac_idf, rk
            FROM (SELECT s.*, row_number() OVER (PARTITION BY s.s1 ORDER BY {best[1]}) AS rk FROM sc s)
            WHERE rk <= {args.cap} ORDER BY s1, rk) TO '{dst}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
        print(f"  wrote {dst.name} ({dst.stat().st_size/1e6:.1f} MB) in {time.perf_counter()-t0:.0f}s, "
              f"peak RSS {human(peak_rss_bytes())}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="stage 6: blocking")
    sub = ap.add_subparsers(dest="cmd")

    pi = sub.add_parser("index", help="build the inverted index over S2+S3 names")
    pi.add_argument("--split", choices=["train", "test"], default="train")
    pi.add_argument("--in-dir", default=os.environ.get("PARQUET_DIR", "data/parquet"))
    pi.add_argument("--out", default=os.environ.get("INDEX_DIR", "data/index"))
    pi.add_argument("--min-token-len", type=int, default=MIN_TOKEN_LEN)

    pc = sub.add_parser("candidates", help="generate and evaluate candidate pairs")
    pc.add_argument("--split", choices=["train", "test"], default="train")
    pc.add_argument("--norm-dir", default=os.environ.get("NORM_DIR", "data/normalized"))
    pc.add_argument("--parquet-dir", default=os.environ.get("PARQUET_DIR", "data/parquet"))
    pc.add_argument("--out", default=os.environ.get("CAND_DIR", "data/cand"))
    pc.add_argument("--cap", type=int, default=DEFAULT_CAP)
    pc.add_argument("--s1-ids", default=None,
                    help="parquet with an entity_id column; restrict the S1 side to it")

    pr = sub.add_parser("rerank", help="re-rank capped candidates by token-set Jaccard")
    pr.add_argument("--split", choices=["train", "test"], default="train")
    pr.add_argument("--norm-dir", default=os.environ.get("NORM_DIR", "data/normalized"))
    pr.add_argument("--parquet-dir", default=os.environ.get("PARQUET_DIR", "data/parquet"))
    pr.add_argument("--out", default=os.environ.get("CAND_DIR", "data/cand"))
    pr.add_argument("--cap", type=int, default=DEFAULT_CAP)
    pr.add_argument("--s1-ids", default=None,
                    help="parquet with an entity_id column; restrict the S1 side to it")

    args = ap.parse_args()
    if args.cmd == "rerank":
        return cmd_rerank(args)
    if args.cmd == "candidates":
        return cmd_candidates(args)
    if args.cmd == "index":
        return cmd_index(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
