"""Stage 5 — the one shared normalizer. Used by blocking, features, everything.

Scope is set by two measurements in `docs/PLAN.md` stage 6:

* **Transliteration is a feature-quality improvement, not a blocking prerequisite.**
  For S1 records whose true match is Indic-script, *zero percent* have a name token with
  df <= 1,000 in the S2+S3 index (p10 of the rarest token's DF is 8,928). Their names are
  generic templates -- "Balaji Investment Private Limited" -- so transliterating the
  Indic side recovers the same common words and buys no blocking power. What it does buy
  is a usable name-similarity signal in stage 7.
* **The address is the discriminative field for those records.** 82.2% of Indic-matched
  pairs have an S1 address token with df <= 1,000, versus 78.2% for Latin-matched pairs.
  So address normalization is on the critical path and gets equal care here.

Design notes:

* One canonical pure-Python implementation. Everything else calls it, so blocking and
  features cannot drift apart.
* Transliteration is cached per *token*, not per record. The corpus has only ~171
  distinct Devanagari tokens and ~1,363 other Indic tokens, so the cache converges after
  a few thousand calls and the cost effectively vanishes.
* ASCII strings skip transliteration and unidecode entirely -- ~92% of records.

Usage:
    python src/normalize.py                      # self-test on real strings
    python src/normalize.py --demo               # normalize sample rows from Parquet
    python src/normalize.py --apply --split train --out data/normalized
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import re
import time
import unicodedata
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from unidecode import unidecode

ROOT = Path(__file__).resolve().parent.parent

# Unicode block -> (label, sanscript scheme). Order matters only for lookup speed.
INDIC_BLOCKS: tuple[tuple[int, int, str, str], ...] = (
    (0x0900, 0x097F, "devanagari", sanscript.DEVANAGARI),
    (0x0980, 0x09FF, "bengali", sanscript.BENGALI),
    (0x0A00, 0x0A7F, "gurmukhi", sanscript.GURMUKHI),
    (0x0A80, 0x0AFF, "gujarati", sanscript.GUJARATI),
    (0x0B00, 0x0B7F, "oriya", sanscript.ORIYA),
    (0x0B80, 0x0BFF, "tamil", sanscript.TAMIL),
    (0x0C00, 0x0C7F, "telugu", sanscript.TELUGU),
    (0x0C80, 0x0CFF, "kannada", sanscript.KANNADA),
    (0x0D00, 0x0D7F, "malayalam", sanscript.MALAYALAM),
)
SCHEME_OF = {label: scheme for _, _, label, scheme in INDIC_BLOCKS}

# Legal suffixes, canonicalised to one spelling. French forms matter: France is ~15% of
# the test set and absent from training, so nothing tunes these for us.
LEGAL_CANON = {
    "corporation": "corp", "corp": "corp", "incorporated": "inc", "inc": "inc",
    "limited": "ltd", "ltd": "ltd", "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "llc": "llc", "llp": "llp", "plc": "plc", "company": "co", "co": "co",
    "sarl": "sarl", "sas": "sas", "sasu": "sas", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "sa": "sa",
}
LEGAL_TOKENS = frozenset(LEGAL_CANON) | frozenset(LEGAL_CANON.values())

# Address abbreviations expanded to a single canonical form so both sides converge.
# "st" -> "street" is the common case in this data; it costs us the rare "St <Saint>".
ADDR_CANON = {
    "rd": "road", "st": "street", "str": "street", "blvd": "boulevard", "blv": "boulevard",
    "ave": "avenue", "av": "avenue", "ln": "lane", "dr": "drive", "ct": "court",
    "hwy": "highway", "pkwy": "parkway", "sq": "square", "apt": "apartment",
    "ste": "suite", "fl": "floor", "bldg": "building", "nr": "near", "opp": "opposite",
    "nagar": "nagar", "marg": "marg", "sec": "sector",
    # French street words -- kept as themselves, listed so they are never treated as junk
    "rue": "rue", "avenue": "avenue", "boulevard": "boulevard", "place": "place",
    "impasse": "impasse", "chemin": "chemin", "allee": "allee", "quai": "quai",
    "cours": "cours", "voie": "voie",
}

_NONWORD = re.compile(r"[^0-9a-z]+")
_JUNK_EDGE = re.compile(r"^[^0-9A-Za-zऀ-෿]+|[^0-9A-Za-zऀ-෿]+$")
_AMP = re.compile(r"\s*&\s*")
_POSTCODE = re.compile(r"(?<!\d)(\d{6}|\d{5}(?:-\d{4})?)(?!\d)")
_HOUSENO = re.compile(r"(?<![0-9a-z])(\d+(?:[/-]\d+)*[a-z]?)(?![0-9a-z])")
_TRAILING_A = re.compile(r"(?<=[bcdfghjklmnpqrstvwxyz])a$")


def _keep_edge(ch: str) -> bool:
    """Alphanumeric, or a combining mark. Indic vowel signs are marks (category M*) and
    are not alphanumeric, so a plain \\W trim would eat the final matra off a word."""
    return ch.isalnum() or unicodedata.category(ch)[0] == "M"


def trim_junk(s: str) -> str:
    """Strip leading/trailing non-alphanumeric junk: '-- Holloway Peak', '<< Team Ecole'.

    Done before anything else -- leading junk poisons tokenisation and sort keys. Trimmed
    by Unicode category rather than an ASCII class, so 'Café' keeps its 'é'.
    """
    i, j = 0, len(s)
    while i < j and not _keep_edge(s[i]):
        i += 1
    while j > i and not _keep_edge(s[j - 1]):
        j -= 1
    return s[i:j]


def detect_script(s: str) -> str:
    """First Indic block present, else 'latin'. ASCII short-circuits."""
    if s.isascii():
        return "latin"
    for ch in s:
        o = ord(ch)
        if o < 0x0300:
            continue
        for lo, hi, label, _ in INDIC_BLOCKS:
            if lo <= o <= hi:
                return label
    return "latin"


@lru_cache(maxsize=2_000_000)
def translit_token(token: str, script: str) -> str:
    """Indic -> Latin for one token, cached.

    Names need almost no cache (only ~171 distinct Devanagari tokens corpus-wide), but
    Indic *addresses* carry ~1.2M distinct tokens -- street and locality names -- so the
    cache is sized for those. At 200k it thrashed: 1.24M misses against 11.1M hits.
    """
    # An Indic-script record routinely mixes scripts -- "5/177-H, ERNAKULAM, കേരളം" is
    # a Latin house number, a Latin city and a Malayalam state. Transliterating the
    # ASCII parts corrupts them: the anusvara rule below turns every capital M into n,
    # so ERNAKULAM -> ernakulan and Mumbai -> numbai, and the record then shares no
    # token with its Latin S1 counterpart. Pass ASCII through untouched, punctuation
    # included, so it tokenizes exactly as the pure-Latin path would.
    if token.isascii():
        return token.lower()
    scheme = SCHEME_OF.get(script)
    if scheme is None:
        return unidecode(token).lower()
    raw_it = transliterate(token, scheme, sanscript.ITRANS)
    # ITRANS encodes anusvara as 'M'; unidecode would render it 'm', giving "marketimga".
    # The Latin spelling these names actually use is 'n' -- "marketing", "consultants".
    raw_it = raw_it.replace("M", "n").replace(".m", "n").replace(".n", "n")
    out = unidecode(raw_it).lower()
    out = _NONWORD.sub("", out)
    # ITRANS renders the inherent vowel, so "लिमिटेड" -> "limiteda" and "राम" -> "rama".
    # Dropping a single trailing schwa after a consonant recovers the Latin form the S1
    # side actually uses. Only applied to tokens long enough that it cannot empty them.
    if len(out) > 3:
        out = _TRAILING_A.sub("", out)
    return out


def _fold(s: str, script: str) -> str:
    """Lowercase, strip accents, transliterate Indic. ASCII takes the fast path."""
    if s.isascii():
        return s.lower()
    if script != "latin":
        return " ".join(
            translit_token(t, script) for t in re.split(r"\s+", s) if t
        )
    return unidecode(unicodedata.normalize("NFKC", s)).lower()


def _tokens(s: str) -> list[str]:
    return [t for t in _NONWORD.split(s) if len(t) >= 2 or t.isdigit()]


@dataclass(slots=True)
class Normalized:
    script: str          # detected script of the raw field
    norm: str            # full normalized string
    core: str            # name with legal suffixes removed (names only)
    tokens: tuple[str, ...]
    core_tokens: tuple[str, ...]
    numbers: tuple[str, ...]   # house/street numbers (addresses)
    postcode: str              # PIN / ZIP, "" when absent

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("tokens", "core_tokens", "numbers"):
            d[k] = list(d[k])
        return d


def normalize_name(raw: str | None) -> Normalized:
    """Normalize a business name. Emits both the full form and the suffix-stripped core."""
    if not raw:
        return Normalized("latin", "", "", (), (), (), "")
    s = trim_junk(raw)
    script = detect_script(s)
    s = _AMP.sub(" and ", s)
    s = _fold(s, script)
    toks = [LEGAL_CANON.get(t, t) for t in _tokens(s)]
    core = [t for t in toks if t not in LEGAL_TOKENS]
    if not core:                          # a name that is nothing but legal suffixes
        core = toks
    return Normalized(script, " ".join(toks), " ".join(core), tuple(toks), tuple(core), (), "")


def normalize_address(raw: str | None) -> Normalized:
    """Normalize an address and pull out the structured parts that discriminate best."""
    if not raw:
        return Normalized("latin", "", "", (), (), (), "")
    s = trim_junk(raw)
    script = detect_script(s)
    s = _AMP.sub(" and ", s)
    s = _fold(s, script)
    post = ""
    m = _POSTCODE.search(s)
    if m:
        post = m.group(1)
    numbers = tuple(dict.fromkeys(_HOUSENO.findall(s)))
    toks = [ADDR_CANON.get(t, t) for t in _tokens(s)]
    words = tuple(t for t in toks if not t.isdigit())
    return Normalized(script, " ".join(toks), " ".join(words), tuple(toks), words, numbers, post)


# --------------------------------------------------------------------------- selftest


CASES_NAME = [
    "-- Holloway Peak Inc Seafood",
    "<< Team Ecole",
    "B+ Retail Inc",
    "Orelee's Barbershop",
    "Balaji Investment Private Limited",
    "राम मार्केटिंग प्राइवेट लिमिटेड",
    "ಶ್ಯಾಮ್ ಟೆಕ್ನಾಲಜಿ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್",
    "பாலாஜி இன்வெஸ்ட்மெண்ட் பிரைவேட் லிமிடெட்",
    "Café Bordeaux SARL",
    "Dupont & Fils S.A.S.",
    "ACME CORPORATION",
]
CASES_ADDR = [
    "1795 Westchester Drive, High Point, NC",
    "A/101, Vithoba Sadan, Near Radha Krishna Temple, Thane West, Thane, Maharashtra",
    "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi 110059",
    "175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine",
    "6/1 Pc Industrial Area, Bangalore North, Bangalore, Karnataka 560079",
    "17560 Ellis Rd, Tahlequah, OK 74464",
]


def selftest() -> int:
    print("=" * 84)
    print("NORMALIZE — names")
    print("=" * 84)
    for raw in CASES_NAME:
        r = normalize_name(raw)
        print(f"  raw    {raw!r}")
        print(f"  script {r.script:<12} norm {r.norm!r}")
        print(f"  core   {r.core!r}")
        print()

    print("=" * 84)
    print("NORMALIZE — addresses")
    print("=" * 84)
    for raw in CASES_ADDR:
        r = normalize_address(raw)
        print(f"  raw      {raw!r}")
        print(f"  norm     {r.norm!r}")
        print(f"  numbers  {list(r.numbers)}   postcode {r.postcode!r}   script {r.script}")
        print()

    ok = True

    def check(label: str, got, want) -> None:
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'} {label:<44} {got!r}")
        if not good:
            print(f"       expected {want!r}")

    print("=" * 84)
    print("ASSERTIONS")
    print("=" * 84)
    check("leading junk stripped", normalize_name("-- Holloway Peak Inc").norm, "holloway peak inc")
    check("angle junk stripped", normalize_name("<< Team Ecole").norm, "team ecole")
    check("ampersand expanded", normalize_name("Dupont & Fils").norm, "dupont and fils")
    check("legal suffix canonicalised", normalize_name("ACME CORPORATION").norm, "acme corp")
    check("core drops legal suffix", normalize_name("ACME CORPORATION").core, "acme")
    check("french suffix folded", normalize_name("Café Bordeaux SARL").core, "cafe bordeaux")
    check("all-suffix name keeps tokens", normalize_name("Limited").core, "ltd")
    check("accents folded", normalize_name("Café").norm, "cafe")
    check("devanagari script", normalize_name("राम मार्केटिंग").script, "devanagari")
    check("tamil script", normalize_name("பாலாஜி").script, "tamil")
    check("empty input", normalize_name("").norm, "")
    check("none input", normalize_name(None).norm, "")
    check("india PIN extracted", normalize_address("Delhi 110059").postcode, "110059")
    check("US ZIP extracted", normalize_address("Tahlequah, OK 74464").postcode, "74464")
    check("no postcode", normalize_address("High Point, NC").postcode, "")
    check("house number", list(normalize_address("A/101, Vithoba Sadan").numbers), ["101"])
    check("compound number", list(normalize_address("6/1 Pc Industrial Area").numbers), ["6/1"])
    check("rd expanded", normalize_address("17560 Ellis Rd").norm, "17560 ellis road")
    check("french street kept", "boulevard" in normalize_address("175 Boulevard du Roosevelt").tokens, True)

    # Transliteration should bring an Indic name close to its Latin counterpart.
    dev = normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड")
    print()
    print(f"  devanagari -> {dev.norm!r}   (S1 counterpart reads 'ram marketing pvt ltd')")
    check("schwa trimmed: ltd recognised", "ltd" in dev.tokens, True)

    print()
    print("=" * 84)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    print("=" * 84)
    return 0 if ok else 1


# ------------------------------------------------------------------------------- main


COLUMNS = ("name_norm", "name_core", "name_tokens", "name_script",
           "addr_norm", "addr_tokens", "addr_numbers", "postcode", "addr_script")


def normalize_rows(names: list, addrs: list) -> tuple[list, ...]:
    """Normalize one block of rows. The single source of truth for both code paths:
    serial and parallel call exactly this, so their output cannot diverge."""
    nm = [normalize_name(v) for v in names]
    ad = [normalize_address(v) for v in addrs]
    return (
        [r.norm for r in nm], [r.core for r in nm],
        [list(r.core_tokens) for r in nm], [r.script for r in nm],
        [r.norm for r in ad], [list(r.core_tokens) for r in ad],
        [list(r.numbers) for r in ad], [r.postcode for r in ad], [r.script for r in ad],
    )


def _worker(chunk: tuple[list, list]) -> tuple[list, ...]:
    return normalize_rows(chunk[0], chunk[1])


def _normalize_columns(names: list, addrs: list, pool, chunk_rows: int) -> tuple[list, ...]:
    """Normalize two columns, optionally across a process pool.

    Chunks are dispatched with ordered `imap`, so results reassemble in input order and
    the parallel output is identical to the serial output row for row -- the property
    --verify asserts. Transliteration is pure Python under the GIL, so only processes
    help; threads would not.
    """
    if pool is None or len(names) <= chunk_rows:
        return normalize_rows(names, addrs)
    # Generator, not a list: materialising every slice up front duplicates the whole
    # column before any work starts.
    chunks = ((names[i:i + chunk_rows], addrs[i:i + chunk_rows])
              for i in range(0, len(names), chunk_rows))
    out: tuple[list, ...] = tuple([] for _ in COLUMNS)
    for part in pool.imap(_worker, chunks):
        for dst, src in zip(out, part):
            dst.extend(src)
    return out


def apply_split(split: str, in_dir: Path, out_dir: Path, workers: int = 8,
                chunk_rows: int = 25_000, batch_rows: int = 250_000,
                sources: tuple[str, ...] = ("1", "2", "3")) -> list[Path]:
    """Normalize a split to Parquet, streaming in batches.

    Batching matters more than the pool did. Holding 5.3M rows of Python output -- nine
    columns, two of them lists-of-lists -- peaked at 8 GB and pushed the machine into
    swap, which erased the parallel speedup entirely (source3 ran at 19k rows/s with 8
    workers, exactly its single-core rate). Converting each batch to Arrow immediately
    and keeping only Arrow frames caps Python-object memory at one batch.
    """
    import polars as pl

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    pool = ctx.Pool(processes=workers) if workers > 1 else None
    try:
        for sfx in sources:
            src = f"{split}_source{sfx}"
            path = in_dir / f"{src}.parquet"
            if not path.exists():
                raise SystemExit(f"{path} not found. Run src/convert.py first.")
            t0 = time.perf_counter()
            df = pl.read_parquet(path)
            frames = []
            for off in range(0, df.height, batch_rows):
                part = df.slice(off, batch_rows)
                cols = _normalize_columns(
                    part["business_name"].to_list(), part["business_address"].to_list(),
                    pool, chunk_rows,
                )
                frames.append(part.with_columns(
                    **{name: pl.Series(name, vals) for name, vals in zip(COLUMNS, cols)}
                ))
                del cols
            out = pl.concat(frames) if len(frames) > 1 else frames[0]
            del frames
            dst = out_dir / f"{src}.parquet"
            out.write_parquet(dst, compression="zstd")
            del out, df
            written.append(dst)
            dt = time.perf_counter() - t0
            rows = _rows(dst)
            print(f"  {src:<18}{rows:>12,} rows  {dt:>7.1f}s  {rows / dt:>10,.0f} rows/s  "
                  f"{dst.stat().st_size / 1e6:>7.1f} MB")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return written


def _rows(path: Path) -> int:
    import polars as pl
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def verify_parallel(split: str, in_dir: Path, tmp: Path, workers: int) -> int:
    """Assert the parallel path reproduces the serial path exactly."""
    import polars as pl

    print(f"serial (workers=1) vs parallel (workers={workers}) on {in_dir}\n")
    a = apply_split(split, in_dir, tmp / "serial", workers=1)
    b = apply_split(split, in_dir, tmp / "par", workers=workers)
    ok = True
    print()
    for pa, pb in zip(a, b):
        ha, hb = sha256(pa), sha256(pb)
        same_bytes = ha == hb
        same_frame = pl.read_parquet(pa).equals(pl.read_parquet(pb))
        ok &= same_frame and same_bytes
        print(f"  {pa.name:<26} bytes {'IDENTICAL' if same_bytes else 'DIFFER':<10} "
              f"frame {'EQUAL' if same_frame else 'DIFFER'}")
        print(f"    sha256 {ha[:32]}...")
    print()
    print("PARALLEL OUTPUT VERIFIED IDENTICAL" if ok else "MISMATCH — do not use the parallel path")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="stage 5: shared normalizer")
    ap.add_argument("--apply", action="store_true", help="normalize Parquet and write it out")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--in-dir", default=os.environ.get("PARQUET_DIR", "data/parquet"))
    ap.add_argument("--out", default=os.environ.get("NORM_DIR", "data/normalized"))
    ap.add_argument("--demo", action="store_true", help="show normalized sample rows")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "8")))
    ap.add_argument("--chunk-rows", type=int, default=25_000)
    ap.add_argument("--batch-rows", type=int, default=250_000)
    ap.add_argument("--verify", action="store_true",
                    help="run serial and parallel, assert byte-identical output")
    args = ap.parse_args()

    in_dir = Path(args.in_dir) if Path(args.in_dir).is_absolute() else ROOT / args.in_dir
    out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out

    if args.verify:
        return verify_parallel(args.split, in_dir, out_dir, args.workers)

    if args.apply:
        print(f"normalizing {args.split} from {in_dir}  workers={args.workers} "
              f"chunk={args.chunk_rows:,}")
        t0 = time.perf_counter()
        apply_split(args.split, in_dir, out_dir, args.workers, args.chunk_rows, args.batch_rows)
        print(f"  total {time.perf_counter() - t0:.1f}s")
        if args.workers <= 1:
            print(f"  cache: {translit_token.cache_info()}")
        return 0

    if args.demo:
        import polars as pl
        in_dir = Path(args.in_dir) if Path(args.in_dir).is_absolute() else ROOT / args.in_dir
        df = pl.read_parquet(in_dir / f"{args.split}_source2.parquet").sample(8, seed=42)
        for row in df.iter_rows(named=True):
            nm, ad = normalize_name(row["business_name"]), normalize_address(row["business_address"])
            print(f"  {row['entity_id']}  [{nm.script}]")
            print(f"    name {row['business_name']!r}\n      -> {nm.norm!r}  core={nm.core!r}")
            print(f"    addr {row['business_address']!r}\n      -> {ad.norm!r}  no={list(ad.numbers)} pc={ad.postcode!r}")
            print()
        return 0

    return selftest()


if __name__ == "__main__":
    raise SystemExit(main())
