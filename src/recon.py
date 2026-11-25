"""Stage 2 — data recon over the Parquet built by ``src/convert.py``.

Every aggregate runs in DuckDB; only small result frames come back to Python.
No source file is ever read with pandas.

The two numbers that change the pipeline design:

* the **singleton rate** (section 2) — with F_0.5 scored per S1 entity, a correct
  empty prediction is worth a full 1.0, so this drives the whole decision layer;
* whether an S2/S3 ID is ever claimed by **two** S1 entities (section 3). If it
  never is, a one-to-one constraint holds and stage 9 gets a free precision win.

Output is printed and appended to ``docs/NOTES.md`` under a ``## Recon`` heading.

Usage:
    python src/recon.py

Env:
    PARQUET_DIR   directory holding the Parquet files   (default: data/parquet)
    NOTES         report destination                    (default: docs/NOTES.md)
    SAMPLE_SEED   seed for the random-row sample        (default: 42)
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
WIDTH = 86

SOURCES = (
    "train_source1",
    "train_source2",
    "train_source3",
    "test_source1",
    "test_source2",
    "test_source3",
)
FIELDS = ("entity_id", "business_name", "business_address", "country")
TEXT_FIELDS = ("business_name", "business_address")

# Unicode blocks worth naming. The challenge brief mentions only Devanagari, but the
# data also carries several other Indic scripts -- see section 5.
BLOCKS = (
    ("Devanagari", "0900", "097F"),
    ("Bengali", "0980", "09FF"),
    ("Gurmukhi", "0A00", "0A7F"),
    ("Gujarati", "0A80", "0AFF"),
    ("Oriya", "0B00", "0B7F"),
    ("Tamil", "0B80", "0BFF"),
    ("Telugu", "0C00", "0C7F"),
    ("Kannada", "0C80", "0CFF"),
    ("Malayalam", "0D00", "0D7F"),
    ("Sinhala", "0D80", "0DFF"),
    ("Arabic", "0600", "06FF"),
    ("Cyrillic", "0400", "04FF"),
    ("Greek", "0370", "03FF"),
    ("CJK", "4E00", "9FFF"),
)

# "Latin" = nothing outside Latin/Latin-Extended plus common punctuation, general
# punctuation and currency symbols. French accented names must land here, not in
# "Other", or the France-in-test signal gets buried.
LATIN_RANGES = r"\x{0020}-\x{024F}\x{2000}-\x{206F}\x{20A0}-\x{20BF}"
DEVANAGARI_RE = r"[\x{0900}-\x{097F}]"


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


class Report:
    """Collects every line so stdout and the docs/NOTES.md append cannot drift apart."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, line: str = "") -> None:
        print(line)
        self.lines.append(line)

    def head(self, num: str, title: str) -> None:
        self()
        self("=" * WIDTH)
        self(f"{num}. {title}")
        self("=" * WIDTH)

    def sub(self, title: str) -> None:
        self()
        self(title)
        self("-" * len(title))

    def table(self, headers: list[str], rows: list[list[str]], right: set[int] | None = None) -> None:
        if not rows:
            self("  (none)")
            return
        right = right if right is not None else set(range(1, len(headers)))
        widths = [
            max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))
        ]
        def fmt(cells: list[str]) -> str:
            out = []
            for i, c in enumerate(cells):
                out.append(c.rjust(widths[i]) if i in right else c.ljust(widths[i]))
            return "  " + "  ".join(out).rstrip()
        self(fmt(headers))
        self("  " + "  ".join("-" * w for w in widths))
        for r in rows:
            self(fmt(r))


def n(v: int | float) -> str:
    return f"{v:,}" if isinstance(v, int) else f"{v:,.2f}"


def pct(part: int, whole: int) -> str:
    return "n/a" if not whole else f"{100.0 * part / whole:.2f}%"


def script_case(col: str) -> str:
    """SQL CASE classifying a text column as Latin / Devanagari / Other."""
    return (
        f"CASE WHEN {col} IS NULL THEN 'NULL'"
        f" WHEN regexp_matches({col}, '{DEVANAGARI_RE}') THEN 'Devanagari'"
        f" WHEN NOT regexp_matches({col}, '[^{LATIN_RANGES}]') THEN 'Latin'"
        f" ELSE 'Other' END"
    )


# --------------------------------------------------------------------------- setup


def connect(parquet_dir: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    wanted = list(SOURCES) + ["train_ground_truth"]
    missing = [w for w in wanted if not (parquet_dir / f"{w}.parquet").exists()]
    if missing:
        raise SystemExit(
            f"{parquet_dir} is missing {missing}.\nRun src/convert.py first."
        )
    for name in wanted:
        con.execute(
            f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{parquet_dir / (name + '.parquet')}')"
        )
    return con


def provenance(parquet_dir: Path) -> tuple[str, dict]:
    """Which DATA_DIR did this Parquet come from? Read back from convert.py's manifest."""
    try:
        manifest = json.loads((parquet_dir / "_manifest.json").read_text())
    except (OSError, json.JSONDecodeError):
        return "unknown", {}
    names = {Path(v["source"]).parents[1].name for v in manifest.values() if "source" in v}
    return (names.pop() if len(names) == 1 else "mixed"), manifest


# ------------------------------------------------------------------------ sections


def match_distribution(con, r: Report) -> None:
    r.head("1", "Match-count distribution per S1 entity (train ground truth)")
    # The NULL/blank guard is load-bearing: str_split('', ',') returns a 1-element
    # list, which would report every singleton as having exactly one match.
    con.execute(
        """
        CREATE VIEW gt_counts AS
        SELECT source1_entity_id,
               CASE WHEN matched_entity_ids IS NULL OR trim(matched_entity_ids) = ''
                    THEN 0 ELSE len(str_split(matched_entity_ids, ',')) END AS k
        FROM train_ground_truth
        """
    )
    total = con.execute("SELECT count(*) FROM gt_counts").fetchone()[0]
    rows = con.execute(
        """
        SELECT CASE WHEN k >= 4 THEN '4+' ELSE CAST(k AS VARCHAR) END AS bucket,
               count(*) AS entities
        FROM gt_counts GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    r.table(
        ["matches", "S1 entities", "share"],
        [[b, n(c), pct(c, total)] for b, c in rows],
    )
    stats = con.execute(
        "SELECT sum(k), avg(k), max(k), quantile_cont(k, 0.5), quantile_cont(k, 0.95) FROM gt_counts"
    ).fetchone()
    r()
    r(f"  S1 entities in ground truth : {n(total)}")
    r(f"  total matched IDs           : {n(int(stats[0]))}")
    r(f"  matches per entity          : mean {stats[1]:.2f}  median {stats[3]:.0f}  p95 {stats[4]:.0f}  max {stats[2]}")


def singleton_rate(con, r: Report) -> None:
    r.head("2", "Singleton rate")
    total, zero = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE k = 0) FROM gt_counts"
    ).fetchone()
    r()
    r(f"  {n(zero)} of {n(total)} S1 entities have NO match  ->  {pct(zero, total)}")
    r()
    r("  Each is worth a full 1.0 when predicted empty, and 0.0 if given any match.")
    r("  A do-nothing baseline that predicts empty for every S1 entity therefore")
    r(f"  scores macro-F_0.5 = {zero / total:.4f} on this split. Anything the pipeline")
    r("  emits must beat that, so stage 9 must be willing to predict nothing.")


def one_to_one(con, r: Report) -> None:
    r.head("3", "Is any S2/S3 ID claimed by more than one S1 entity?")
    con.execute(
        """
        CREATE VIEW gt_pairs AS
        SELECT source1_entity_id AS s1, trim(m) AS mid
        FROM train_ground_truth, unnest(str_split(matched_entity_ids, ',')) AS t(m)
        WHERE matched_entity_ids IS NOT NULL AND trim(matched_entity_ids) <> ''
        """
    )
    pairs, distinct = con.execute(
        "SELECT count(*), count(DISTINCT mid) FROM gt_pairs"
    ).fetchone()
    con.execute(
        """
        CREATE VIEW gt_conflicts AS
        SELECT mid, count(DISTINCT s1) AS n_s1, string_agg(DISTINCT s1, ', ') AS s1_ids
        FROM gt_pairs GROUP BY mid HAVING count(DISTINCT s1) > 1
        """
    )
    conflicts = con.execute("SELECT count(*) FROM gt_conflicts").fetchone()[0]
    r()
    r(f"  (S1, matched ID) pairs      : {n(pairs)}")
    r(f"  distinct matched S2/S3 IDs  : {n(distinct)}")
    r(f"  IDs claimed by >1 S1 entity : {n(conflicts)}  ({pct(conflicts, distinct)})")
    r()
    if conflicts == 0:
        r("  VERDICT: one-to-one holds. Every S2/S3 record belongs to at most one S1")
        r("  entity, so stage 9 can resolve conflicts greedily -- award a contested ID")
        r("  to the highest-scoring S1 and drop the rest. Pure precision gain, zero")
        r("  recall cost, and F_0.5 weights precision 2x.")
    else:
        r("  VERDICT: one-to-one does NOT hold. Do not add a uniqueness constraint in")
        r("  stage 9; a shared ID is legitimate. Examples:")
        rows = con.execute(
            "SELECT mid, n_s1, s1_ids FROM gt_conflicts ORDER BY n_s1 DESC, mid LIMIT 5"
        ).fetchall()
        r()
        r.table(
            ["matched ID", "# S1", "claimed by"],
            [[m, str(c), s] for m, c, s in rows],
            right={1},
        )


def country_counts(con, r: Report) -> None:
    r.head("4", "Row counts per country, per source")
    union = " UNION ALL ".join(
        f"SELECT '{s}' AS src, country, count(*) AS rows FROM {s} GROUP BY 2" for s in SOURCES
    )
    rows = con.execute(f"SELECT * FROM ({union}) ORDER BY src, rows DESC").fetchall()
    totals = dict(
        con.execute(
            " UNION ALL ".join(f"SELECT '{s}', count(*) FROM {s}" for s in SOURCES)
        ).fetchall()
    )
    r.table(
        ["source", "country", "rows", "share of source"],
        [[s, c if c is not None else "(NULL)", n(k), pct(k, totals[s])] for s, c, k in rows],
        right={2, 3},
    )
    seen = {s: {c for ss, c, _ in rows if ss == s} for s in SOURCES}
    train_only = set().union(*(seen[s] for s in SOURCES if s.startswith("train")))
    test_only = set().union(*(seen[s] for s in SOURCES if s.startswith("test")))
    r()
    r(f"  countries in train : {', '.join(sorted(map(str, train_only)))}")
    r(f"  countries in test  : {', '.join(sorted(map(str, test_only)))}")
    unseen = test_only - train_only
    r(f"  test-only (unseen) : {', '.join(sorted(map(str, unseen))) or '(none)'}")
    if unseen:
        r()
        r("  Country must stay an open set of string labels used only as an equality")
        r("  feature. One-hot encoding it leaves the model with nothing for the unseen")
        r("  country -- confidently wrong, and invisible until the private leaderboard.")


def script_breakdown(con, r: Report) -> None:
    r.head("5", "Script of business_name, per source")
    case = script_case("business_name")
    union = " UNION ALL ".join(
        f"SELECT '{s}' AS src, {case} AS script, count(*) AS rows FROM {s} GROUP BY 2"
        for s in SOURCES
    )
    rows = con.execute(f"SELECT * FROM ({union})").fetchall()
    totals: dict[str, int] = {}
    for s, _, k in rows:
        totals[s] = totals.get(s, 0) + k
    by = {(s, sc): k for s, sc, k in rows}
    order = ["Latin", "Devanagari", "Other", "NULL"]
    present = [o for o in order if any((s, o) in by for s in SOURCES)]
    r.table(
        ["source"] + present + ["total"],
        [
            [s] + [f"{n(by.get((s, o), 0))} ({pct(by.get((s, o), 0), totals[s])})" for o in present]
            + [n(totals[s])]
            for s in SOURCES
        ],
    )

    # "Other" is not junk -- break it down before stage 5 plans transliteration.
    r.sub("What is inside 'Other'? (detected Unicode blocks, per source)")
    parts = []
    for s in SOURCES:
        for label, lo, hi in BLOCKS:
            if label == "Devanagari":
                continue
            parts.append(
                f"SELECT '{s}' AS src, '{label}' AS blk, count(*) AS rows FROM {s} "
                f"WHERE {case} = 'Other' "
                f"AND regexp_matches(business_name, '[\\x{{{lo}}}-\\x{{{hi}}}]')"
            )
    hits = [
        (s, b, k)
        for s, b, k in con.execute(" UNION ALL ".join(parts)).fetchall()
        if k > 0
    ]
    if not hits:
        r("  (no named block matched -- inspect section 8 rows directly)")
    else:
        blocks_seen = [b for _, b, _ in dict.fromkeys((None, b, None) for _, b, _ in hits)]
        blocks_seen = sorted({b for _, b, _ in hits})
        hb = {(s, b): k for s, b, k in hits}
        other_tot = {s: by.get((s, "Other"), 0) for s in SOURCES}
        r.table(
            ["source"] + blocks_seen + ["'Other' total"],
            [
                [s] + [n(hb.get((s, b), 0)) for b in blocks_seen] + [n(other_tot[s])]
                for s in SOURCES
            ],
        )
        r()
        r("  NOTE: docs/PLAN.md stage 5 plans Devanagari transliteration only. These are")
        r("  additional Indic scripts, so the normalizer needs a per-script transliteration")
        r("  path (indic-transliteration covers all of them offline) and blocking recall")
        r("  must be measured per script -- non-Latin records fail silently otherwise.")


def missingness(con, r: Report) -> None:
    r.head("6", "NULL / empty / whitespace-only counts per field, per source")
    r()
    r("  'blank' = trims to empty but is not the empty string, which a naive NULL or")
    r("  = '' check would miss entirely.")
    for s in SOURCES:
        total = con.execute(f"SELECT count(*) FROM {s}").fetchone()[0]
        parts = [
            f"SELECT '{f}' AS field,"
            f" count(*) FILTER (WHERE {f} IS NULL) AS nulls,"
            f" count(*) FILTER (WHERE {f} = '') AS empties,"
            f" count(*) FILTER (WHERE {f} <> '' AND trim({f}) = '') AS blanks"
            f" FROM {s}"
            for f in FIELDS
        ]
        rows = con.execute(" UNION ALL ".join(parts)).fetchall()
        r.sub(f"{s}  ({n(total)} rows)")
        r.table(
            ["field", "NULL", "empty ''", "blank", "any missing", "share"],
            [
                [f, n(a), n(b), n(c), n(a + b + c), pct(a + b + c, total)]
                for f, a, b, c in rows
            ],
        )


def lengths(con, r: Report) -> None:
    r.head("7", "Field length distributions (characters), per source")
    for s in SOURCES:
        parts = [
            f"SELECT '{f}' AS field, min(length({f})), quantile_cont(length({f}), 0.5),"
            f" quantile_cont(length({f}), 0.9), quantile_cont(length({f}), 0.99),"
            f" max(length({f})), avg(length({f})) FROM {s}"
            for f in TEXT_FIELDS
        ]
        rows = con.execute(" UNION ALL ".join(parts)).fetchall()
        r.sub(s)
        r.table(
            ["field", "min", "p50", "p90", "p99", "max", "mean"],
            [
                [f, n(mn), f"{p50:.0f}", f"{p90:.0f}", f"{p99:.0f}", n(mx), f"{av:.1f}"]
                for f, mn, p50, p90, p99, mx, av in rows
            ],
        )


def random_rows(con, r: Report, seed: int) -> None:
    r.head("8", f"10 random full rows per source (reservoir sample, seed {seed})")
    for s in SOURCES:
        rows = con.execute(
            f"SELECT {', '.join(FIELDS)} FROM {s} "
            f"USING SAMPLE reservoir(10 ROWS) REPEATABLE ({seed})"
        ).fetchall()
        r.sub(s)
        for i, row in enumerate(rows, 1):
            r(f"  [{i:>2}] entity_id        {row[0]}")
            r(f"       business_name    {row[1]!r}")
            r(f"       business_address {row[2]!r}")
            r(f"       country          {row[3]}")
            r()


# ---------------------------------------------------------------------------- main


def main() -> int:
    parquet_dir = resolve(os.environ.get("PARQUET_DIR", "data/parquet"))
    notes_path = resolve(os.environ.get("NOTES", "docs/NOTES.md"))
    seed = int(os.environ.get("SAMPLE_SEED", "42"))

    con = connect(parquet_dir)
    label, manifest = provenance(parquet_dir)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    r = Report()
    r("=" * WIDTH)
    r("RECON")
    r("=" * WIDTH)
    r(f"  generated   {stamp}")
    r(f"  parquet     {parquet_dir}")
    r(f"  DATA_DIR    {label}")
    r(f"  duckdb      {duckdb.__version__}")
    if manifest:
        r(f"  rows        {n(sum(v.get('rows', 0) for v in manifest.values()))} across {len(manifest)} files")
    if label != "dataset":
        r()
        r("  *** SAMPLE DATA -- NOT THE FULL DATASET ***")
        r("  dataset_sample/ takes the first 50k rows of each file INDEPENDENTLY, so the")
        r("  ground-truth rows reference S2/S3 IDs that are mostly absent from the sampled")
        r("  source2/source3 files. Sections 1-3 read the ground-truth file alone and are")
        r("  structurally valid here; sections 4-8 describe the sample, not the population.")
        r("  Cross-file joins are meaningless on the sample and are not attempted.")

    match_distribution(con, r)
    singleton_rate(con, r)
    one_to_one(con, r)
    country_counts(con, r)
    script_breakdown(con, r)
    missingness(con, r)
    lengths(con, r)
    random_rows(con, r, seed)

    r()
    r("=" * WIDTH)
    r("END RECON")
    r("=" * WIDTH)

    existed = notes_path.exists()
    with notes_path.open("a", encoding="utf-8") as fh:
        if not existed:
            fh.write("# Notes\n\nRunning log of what was tried, what it scored, what changed.\n")
        fh.write(f"\n## Recon — DATA_DIR={label} ({stamp})\n\n")
        fh.write("Generated by `src/recon.py`. Fenced to preserve column alignment.\n\n")
        fh.write("```\n")
        fh.write("\n".join(r.lines).rstrip() + "\n")
        fh.write("```\n")
    print(f"\nappended {len(r.lines)} lines -> {notes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
