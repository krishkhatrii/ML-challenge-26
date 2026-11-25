#!/usr/bin/env bash
# End-to-end train-side evaluation for one experiment.
#   tools/run_exp.sh <tag> [cap] [max_pairs]
#
# Blocking-cap overrides are read from the environment by src/blocking.py:
#   DF_CAP_SINGLE DF_CAP_PAIR DF_CAP_COMPOSITE K_PAIR K_COMPOSITE PRE_CAP NPART
# Stage skips: SKIP_CAND=1 SKIP_RERANK=1 SKIP_FEAT=1 SKIP_TRAIN=1
# S1_IDS=<parquet> restricts candidate generation to those S1 entities (exact; see
# blocking.py). Unset = the full 2.2M S1 side.
set -u
cd "$(dirname "$0")/.."
TAG="${1:?tag}"; CAP="${2:-30}"; MAXP="${3:-5000000}"
PY=.venv/bin/python
L="logs/${TAG}"
mkdir -p logs
# data/parquet is the 50k-row SAMPLE. Every measurement here must use the full data.
export PARQUET_DIR="${PARQUET_DIR:-data/parquet_full}"
export CAND_DIR="${CAND_DIR:-data/cand}"
export FEAT_DIR="${FEAT_DIR:-data/features}"
IDS=""
[ -n "${S1_IDS:-}" ] && IDS="--s1-ids ${S1_IDS}"

echo "=== $TAG  cap=$CAP max_pairs=$MAXP  $(date -u +%H:%M:%S)Z ==="
echo "  PARQUET_DIR=$PARQUET_DIR  CAND_DIR=$CAND_DIR  S1_IDS=${S1_IDS:-<full>}"
env | grep -E '^(DF_CAP|K_PAIR|K_COMPOSITE|PRE_CAP|NPART)=' | sort | sed 's/^/  env /'

step () {
  local n="$1"; shift
  local t0=$SECONDS
  echo "--- $n  $(date -u +%H:%M:%S)Z"
  "$@" > "${L}_${n}.log" 2>&1
  echo "    $n rc=$?  $((SECONDS-t0))s  free=$(df -g . | tail -1 | awk '{print $4}')GB"
  return 0   # blocking.py exits 1 on gate-fail, which is expected
}

[ "${SKIP_CAND:-0}" = 1 ]   || step cand   $PY src/blocking.py candidates --split train --cap "$CAP" $IDS
[ "${SKIP_RERANK:-0}" = 1 ] || step rerank $PY src/blocking.py rerank     --split train --cap "$CAP" $IDS
if [ "${SKIP_FEAT:-0}" != 1 ]; then
  rm -rf "$FEAT_DIR"
  step feat_tr  $PY src/features.py --split train --fold train --s1-frac 0.10 --cap "$CAP"
  step feat_val $PY src/features.py --split train --fold val --cap "$CAP"
fi
[ "${SKIP_TRAIN:-0}" = 1 ] || step train $PY src/train.py --max-pairs "$MAXP" --feat-glob "$FEAT_DIR/train_train_part*.parquet"
step sweep $PY src/decide.py --sweep --feat-glob "$FEAT_DIR/train_val_part*.parquet"

echo
echo "=== $TAG done  $(date -u +%H:%M:%S)Z ==="
grep -hE "GATE|WINNER at cap" "${L}_cand.log" "${L}_rerank.log" 2>/dev/null
grep -hE "CV AUC" "${L}_train.log" 2>/dev/null
grep -hE "BEST tau" "${L}_sweep.log" 2>/dev/null
