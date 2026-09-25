"""Stage 1 — convert the 7 challenge TSVs to Parquet.

Streams every file with Polars (``scan_csv`` -> ``sink_parquet``), so peak memory
stays flat no matter how large the input is. No source file is ever loaded into
RAM, and ``pandas`` is not involved.

Idempotent: a sidecar ``_manifest.json`` records the source path, byte size and
mtime behind each Parquet file. A file is rebuilt only when that stamp no longer
matches — so re-running is cheap, and switching ``DATA_DIR`` between the sample
and the full dataset correctly forces a rebuild (mtime alone would not: the real
TSVs are older than a freshly built sample Parquet).

Usage:
    python src/convert.py                              # DATA_DIR=dataset
    DATA_DIR=dataset_sample python src/convert.py       # 50k-row local sample
    FORCE=1 python src/convert.py                       # rebuild everything

Env:
    DATA_DIR      input directory holding train/ and test/   (default: dataset)
    PARQUET_DIR   output directory for the Parquet files     (default: data/parquet)
    FORCE         set to 1 to rebuild regardless of manifest
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent

# Explicit schemas: every field stays a string. Entity IDs look numeric after the
# prefix and PIN/ZIP codes have leading zeros — inference must not touch them.
SOURCE_SCHEMA = {
    "entity_id": pl.String,
    "business_name": pl.String,
    "business_address": pl.String,
    "country": pl.String,
}
GT_SCHEMA = {
    "source1_entity_id": pl.String,
    "matched_entity_ids": pl.String,
}

FILES: tuple[tuple[str, str, dict[str, pl.DataType]], ...] = (
    ("train/train_source1.tsv", "train_source1.parquet", SOURCE_SCHEMA),
    ("train/train_source2.tsv", "train_source2.parquet", SOURCE_SCHEMA),
    ("train/train_source3.tsv", "train_source3.parquet", SOURCE_SCHEMA),
    ("train/train_ground_truth.tsv", "train_ground_truth.parquet", GT_SCHEMA),
    ("test/test_source1.tsv", "test_source1.parquet", SOURCE_SCHEMA),
    ("test/test_source2.tsv", "test_source2.parquet", SOURCE_SCHEMA),
    ("test/test_source3.tsv", "test_source3.parquet", SOURCE_SCHEMA),
)


def resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def human(n: int) -> str:
    v = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024 or unit == "TB":
            return f"{v:.1f} {unit}" if unit != "B" else f"{int(v)} B"
        v /= 1024
    return f"{v:.1f} TB"


def stamp_of(src: Path) -> dict[str, int | str]:
    st = src.stat()
    return {"source": str(src), "source_bytes": st.st_size, "source_mtime_ns": st.st_mtime_ns}


def row_count(parquet: Path) -> int:
    """Row count straight from the Parquet footer — no data pages are read."""
    return int(pl.scan_parquet(parquet).select(pl.len()).collect().item())


def main() -> int:
    data_dir = resolve(os.environ.get("DATA_DIR", "dataset"))
    out_dir = resolve(os.environ.get("PARQUET_DIR", "data/parquet"))
    force = os.environ.get("FORCE", "").lower() not in ("", "0", "false", "no")

    missing = [rel for rel, _, _ in FILES if not (data_dir / rel).exists()]
    if missing:
        print(f"DATA_DIR={data_dir} is missing {len(missing)} expected file(s):", file=sys.stderr)
        for rel in missing:
            print(f"  {rel}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        manifest = {}

    print(f"DATA_DIR     {data_dir}")
    print(f"PARQUET_DIR  {out_dir}")
    print(f"polars       {pl.__version__}")
    print()
    header = f"{'file':<28}{'rows':>12}  {'tsv':>10}  {'parquet':>10}  {'ratio':>6}  {'secs':>6}  status"
    print(header)
    print("-" * len(header))

    total_tsv = total_pq = total_rows = 0
    for rel, out_name, schema in FILES:
        src = data_dir / rel
        out = out_dir / out_name
        stamp = stamp_of(src)
        cached = manifest.get(out_name, {})
        fresh = (
            not force
            and out.exists()
            and all(cached.get(k) == v for k, v in stamp.items())
            and isinstance(cached.get("rows"), int)
        )

        t0 = time.perf_counter()
        if fresh:
            rows, status = cached["rows"], "skip (up to date)"
        else:
            # Streaming: scan_csv builds a lazy plan, sink_parquet drains it to disk
            # in batches. quote_char=None because the challenge TSVs are unquoted —
            # a stray " must be treated as data, not as a field delimiter.
            (
                pl.scan_csv(
                    src,
                    separator="\t",
                    has_header=True,
                    schema=schema,
                    quote_char=None,
                )
                .sink_parquet(out, compression="zstd", statistics=True)
            )
            rows, status = row_count(out), "written"
        elapsed = time.perf_counter() - t0

        tsv_bytes, pq_bytes = src.stat().st_size, out.stat().st_size
        total_tsv += tsv_bytes
        total_pq += pq_bytes
        total_rows += rows
        manifest[out_name] = {**stamp, "rows": rows, "parquet_bytes": pq_bytes}

        print(
            f"{out_name:<28}{rows:>12,}  {human(tsv_bytes):>10}  {human(pq_bytes):>10}  "
            f"{tsv_bytes / pq_bytes:>5.1f}x  {elapsed:>6.1f}  {status}"
        )

    print("-" * len(header))
    print(
        f"{'total':<28}{total_rows:>12,}  {human(total_tsv):>10}  {human(total_pq):>10}  "
        f"{total_tsv / total_pq:>5.1f}x"
    )

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nmanifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
