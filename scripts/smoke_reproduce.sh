#!/usr/bin/env bash
# A cheap end-to-end smoke test: one task per harness, converted and
# validated.
#
# This is what to run first when something looks wrong.  It exercises the
# whole path -- the trial, the model call, the trajectory, the conversion,
# the independent validator -- in about ten minutes, instead of the days the
# full annotation takes.
#
# Prerequisites: `make setup images toolchain registry-clawgym`, and a
# reachable model endpoint in `.env`.
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ``.env`` first, then the environment wins over it: a value that was already
# set when this script started beats the file.  The rule lives in
# ``opencode_sft/load_env.sh``, so that every script here reads its
# configuration the same way.
source "$REPO/opencode_sft/load_env.sh"

PY3="${PY3:-$REPO/.venv/bin/python}"
PY="${PY:-$REPO/.venv312/bin/python}"
MODEL_BASE_URL="${MODEL_BASE_URL:-http://host.docker.internal:8000/v1}"
LITELLM_NAME="${LITELLM_NAME:-openai/glm5.3-flash}"
MODEL_NAME="${MODEL_NAME:-glm5.3-flash}"
RUN_ROOT="${RUN_ROOT:-$HOME/.rsi-harbor}"
SMOKE="$RUN_ROOT/smoke"
PROXY_PORT="${PROXY_PORT:-8911}"

REG=opencode_sft/registry_clawgym
[[ -d "$REG" ]] || { echo "the ClawGym registry is missing -- run 'make registry-clawgym'" >&2; exit 2; }

mkdir -p "$SMOKE/logs"

log() { printf '\n==> %s\n' "$*"; }

# --------------------------------------------------------------------------- #
# the two tasks: one from each half, so both harnesses get a real task
# --------------------------------------------------------------------------- #
log "picking one task from each half"
head -1 "$REG/manifest-mini.jsonl"    > "$SMOKE/mini.jsonl"
head -1 "$REG/manifest-opencode.jsonl" > "$SMOKE/oc.jsonl"
TASK_MINI=$($PY3 -c "import json,sys; print(json.load(open(sys.argv[1]))['task'])" "$SMOKE/mini.jsonl" 2>/dev/null || \
            $PY3 -c "import json,sys; print(json.loads(open(sys.argv[1]).readline())['task'])" "$SMOKE/mini.jsonl")
TASK_OC=$($PY3 -c "import json,sys; print(json.loads(open(sys.argv[1]).readline())['task'])" "$SMOKE/oc.jsonl")
echo "  mini:    $TASK_MINI"
echo "  opencode: $TASK_OC"

# --------------------------------------------------------------------------- #
# the recording proxy (the opencode stream talks to the model through it)
# --------------------------------------------------------------------------- #
if ! curl -sf -m 5 "http://127.0.0.1:$PROXY_PORT/v1/models" >/dev/null 2>&1; then
  log "starting the recording proxy on $PROXY_PORT"
  nohup $PY3 opencode_sft/record_proxy.py \
    --port "$PROXY_PORT" --host 0.0.0.0 \
    --upstream "${PROXY_UPSTREAM:-$MODEL_BASE_URL}" \
    --out "$SMOKE/logs/oc-requests.jsonl" \
    --compact --dedupe --blob-dir "$SMOKE/logs/oc-blobs" \
    > "$SMOKE/logs/record_proxy.log" 2>&1 &
  sleep 3
fi

# --------------------------------------------------------------------------- #
# the two trials
# --------------------------------------------------------------------------- #
log "mini-swe-agent: one trial"
$PY opencode_sft/run_mini_rollouts.py \
  --registry $REG \
  --manifest "$SMOKE/mini.jsonl" \
  --run-root "$SMOKE/mini" --families clawgym \
  --min-config opencode_sft/clawgym/mini-config-clawgym.yaml \
  --batch-size 1 --concurrency 1 --max-batches 1 --enable-verifier \
  --api-base "$MODEL_BASE_URL" --model-name "$LITELLM_NAME" --api-key "${API_KEY:-EMPTY}"

log "opencode: one trial"
$PY opencode_sft/run_rollouts.py \
  --registry $REG \
  --manifest "$SMOKE/oc.jsonl" \
  --run-root "$SMOKE/oc" --families clawgym \
  --batch-size 1 --concurrency 1 --max-batches 1 --enable-verifier \
  --proxy-check-url "http://127.0.0.1:$PROXY_PORT/v1/models" \
  --api-base "http://host.docker.internal:$PROXY_PORT/v1" \
  --api-key EMPTY --model-id "$MODEL_NAME" --provider-id rsi_glm

# --------------------------------------------------------------------------- #
# convert and validate
# --------------------------------------------------------------------------- #
log "converting"
$PY3 opencode_sft/clawgym/build_sft_dataset.py \
  --run-root "$SMOKE/mini" --registry $REG \
  --out "$SMOKE/mini-sft.jsonl" --min-reward 0.0 --model "$MODEL_NAME"
$PY3 opencode_sft/clawgym/build_opencode_sft_dataset.py \
  --run-root "$SMOKE/oc" --registry $REG \
  --requests "$SMOKE/logs/oc-requests.jsonl" --blob-dir "$SMOKE/logs/oc-blobs" \
  --out "$SMOKE/oc-sft.jsonl" --min-reward 0.0

log "validating"
$PY3 opencode_sft/clawgym/validate_sft.py --dataset "$SMOKE/mini-sft.jsonl" --registry $REG
$PY3 opencode_sft/clawgym/validate_sft.py --dataset "$SMOKE/oc-sft.jsonl" --registry $REG

log "the three keys"
for f in "$SMOKE/mini-sft.jsonl" "$SMOKE/oc-sft.jsonl"; do
  $PY3 - "$f" <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as handle:
    lines = [l for l in handle if l.strip()]
if not lines:
    print(f"  {path}: EMPTY")
    raise SystemExit(0)
row = json.loads(lines[0])
print(f"  {path}: {len(lines)} row(s)")
print("   ", {k: row.get(k) for k in ("dataset", "model", "harness", "model_id", "model_provider")})
PY
done

log "smoke test finished"
