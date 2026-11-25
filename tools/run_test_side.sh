#!/usr/bin/env bash
# Test-side submission regeneration for the current best config.
#   tools/run_test_side.sh <tag> [cap]
# Block switches come from the environment (ENABLE_SETKEY / ENABLE_PRODPAIR / ...),
# exactly as the train-side run that produced the config.
#
# Writes to a STAGING directory. output/ is only overwritten after the validator
# passes, by the caller -- never by this script.
set -u
cd "$(dirname "$0")/.."
TAG="${1:?tag}"; CAP="${2:-30}"
PY=.venv/bin/python
L="logs/${TAG}"
mkdir -p logs
export PARQUET_DIR="${PARQUET_DIR:-data/parquet_full}"
export CAND_DIR="${CAND_DIR:-data/cand_test_${TAG}}"
export FEAT_DIR="${FEAT_DIR:-data/features_test_${TAG}}"
STAGE="stage/${TAG}"
mkdir -p "$STAGE"

echo "=== TEST SIDE ${TAG}  cap=${CAP}  $(date -u +%H:%M:%S)Z ==="
echo "  PARQUET_DIR=$PARQUET_DIR  CAND_DIR=$CAND_DIR  FEAT_DIR=$FEAT_DIR  STAGE=$STAGE"
env | grep -E '^(ENABLE_|DF_CAP|K_PAIR|K_COMPOSITE|K_PROD|PROD_BUDGET|KEY_|PRE_CAP|NPART)=' | sort | sed 's/^/  env /'

step () {
  local n="$1"; shift
  local t0=$SECONDS
  echo "--- $n  $(date -u +%H:%M:%S)Z"
  "$@" > "${L}_${n}.log" 2>&1
  local rc=$?
  echo "    $n rc=$rc  $((SECONDS-t0))s  free=$(df -g . | tail -1 | awk '{print $4}')GB"
  if [ "$n" != "cand" ] && [ $rc -ne 0 ]; then
    echo "    ABORT: $n failed, output/ untouched"; exit 1
  fi
}

step cand   $PY src/blocking.py candidates --split test --cap "$CAP"
step rerank $PY src/blocking.py rerank     --split test --cap "$CAP"
step feat   $PY src/features.py --split test --fold all --cap "$CAP"
step apply  $PY src/decide.py --apply --split test --cap "$CAP" \
              --feat-glob "$FEAT_DIR/test_all_part*.parquet" --out "$STAGE"

echo "--- validate  $(date -u +%H:%M:%S)Z"
$PY utils/validate_submission.py \
    --matching "$STAGE/matching_results.tsv" \
    --candidate "$STAGE/candidate_pairs.tsv" \
    --test-dir dataset/test > "${L}_validate.log" 2>&1
VRC=$?
echo "    validate rc=$VRC"
tail -25 "${L}_validate.log"
echo "=== TEST SIDE ${TAG} done rc=$VRC  $(date -u +%H:%M:%S)Z ==="
exit $VRC
