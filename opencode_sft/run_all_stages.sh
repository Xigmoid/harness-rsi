#!/usr/bin/env bash
# Run the whole re-annotation, stage after stage, unattended.
#
# The runs take days, so the stages are chained: each one starts when the
# previous one has finished, and each one checks its own work before moving
# on.  Nothing here decides *what* to run -- that is the execution plan in
# ``opencode_sft/EXECUTION_PLAN.md``; this script only makes sure the
# machine keeps working while nobody is watching.
#
# Configuration
# -------------
# Everything that is specific to one machine comes from the environment (see
# ``.env.example``): the model endpoint, the run root, the two interpreters,
# the concurrency.  ``.env`` is read if it exists, so ``cp .env.example .env``
# and edit it once.
#
# The wait is on a **PID file**, never on ``pgrep -f``: a pattern that appears
# in the waiting script's own command line matches the waiting script, and
# the wait never ends.
#
# The two streams
# ---------------
#
# Two harnesses (Mini, Opencode), one model:
#
#   mini-glm   mini-swe-agent
#   oc-glm     opencode
#
# Each harness gets half of every dataset, and the half is a manifest file
# written by ``opencode_sft/split_tasks.py``.  The two halves never overlap,
# so every task is annotated exactly once.
#
# A second provider (MiniMax) was wired in here and then taken out again:
# the endpoint serves every task on its own now.  The machinery is still in
# the tree -- ``opencode_sft/minimax.py``, the ``--model-id`` / ``--api-base``
# switches of the drivers, the ``--minimax-share`` of the splitter -- because
# switching a stream over is a command line change, but it is **not** used by
# this script unless ``MINIMAX_SHARE`` is above zero.
#
# The recording proxy
# -------------------
#
# The opencode stream talks to the endpoint through the recording proxy, so
# that every request the model receives is on disk.  The proxy stores the
# repeated parts -- the system prompt, the tool schemas -- once, by digest,
# and the request log carries the digests.  The proxy is also where a real
# API key is injected: the job file only ever holds the placeholder.
#
# The supervisor
# --------------
#
# A stream that dies is started again: the drivers keep a ledger, and a
# driver that is restarted picks up where the ledger says it left off.  A
# stream that is *finished* writes a ``COMPLETE`` marker into its run root,
# and the supervisor only stops when every stream has one.
#
# Usage::
#
#     bash opencode_sft/run_all_stages.sh            # everything that is left
#     STAGES="clawgym-convert" bash opencode_sft/run_all_stages.sh
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
# ``.env`` first, then the environment wins over it.
if [[ -f "$REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO/.env"
  set +a
fi

#: The endpoint the agents talk to, **as seen from inside the container**.
MODEL_BASE_URL="${MODEL_BASE_URL:-http://host.docker.internal:8000/v1}"
#: The same endpoint as seen from the host, which is where the proxy runs.
#: They differ only when the endpoint is on the machine that also runs
#: Docker: the container reaches it at host.docker.internal, the host at
#: 127.0.0.1.
PROXY_UPSTREAM="${PROXY_UPSTREAM:-$MODEL_BASE_URL}"
#: The model id, as the endpoint knows it.
MODEL_NAME="${MODEL_NAME:-glm5.3-flash}"
#: The litellm name the mini harness uses for it.
LITELLM_NAME="${LITELLM_NAME:-openai/${MODEL_NAME##*/}}"
#: The key, or the placeholder for an endpoint that does not check one.
API_KEY="${API_KEY:-EMPTY}"
#: The port the recording proxy listens on, and the port the mini stream
#: talks to the proxy on when it is routed through it.
PROXY_PORT="${PROXY_PORT:-8910}"

#: Where the trials run.  It has to survive a reboot: macOS wipes /tmp on
#: every boot, and a multi-day run does not survive losing its ledger.  The
#: default is therefore under $HOME, not under /tmp.  (On macOS /tmp is
#: also the path Harbor refuses -- it is the sandbox that refuses it, and
#: /private/tmp is the real path of /tmp -- so a persistent path solves
#: that too.)
RUN_ROOT="${RUN_ROOT:-$HOME/.rsi-harbor}"
LOGS="$RUN_ROOT/logs"
PIDS="$RUN_ROOT/pids"
mkdir -p "$LOGS" "$PIDS"

PY="${PY:-$REPO/.venv312/bin/python}"
PY3="${PY3:-$REPO/.venv/bin/python}"

#: How many trials run at the same time, per harness.  The ceiling is
#: memory: an opencode trial is ~500 MB, a mini trial ~100 MB.
MINI_CONCURRENCY="${MINI_CONCURRENCY:-24}"
OPENCODE_CONCURRENCY="${OPENCODE_CONCURRENCY:-10}"

#: The share of each half that is annotated with MiniMax instead of the
#: endpoint above.  0 = every task goes to the main endpoint.
MINIMAX_SHARE="${MINIMAX_SHARE:-0.0}"

STAGES="${STAGES:-clawgym-split clawgym-run clawgym-convert clawgym-merge openthoughts-split openthoughts-run openthoughts-convert}"

log() { printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*" | tee -a "$LOGS/run_all_stages.log"; }

run_stage() {
  local name="$1"; shift
  if [[ " $STAGES " != *" $name "* ]]; then return 0; fi
  log "=== stage $name ==="
  "$@"
  local rc=$?
  log "=== stage $name finished with $rc ==="
  return $rc
}

# --------------------------------------------------------------------------- #
# the streams
# --------------------------------------------------------------------------- #
# Every stream is a function, so that the supervisor can start it again when
# it dies.  The functions are named after the streams with the dashes turned
# into underscores, because that is what a shell function may be called.
#
# The last command of every function is ``exec``: a backgrounded shell
# function forks a subshell, and ``$!`` is *that* subshell's pid, not the
# driver's.  Without ``exec`` the pid file points at a wrapper, and killing
# it leaves the driver running.

clawgym_mini_glm() {
  local reg=opencode_sft/registry_clawgym root=$RUN_ROOT
  exec $PY opencode_sft/run_mini_rollouts.py \
    --registry $reg \
    --manifest $reg/manifest-mini.jsonl \
    --run-root $root/clawgym --families clawgym \
    --min-config opencode_sft/clawgym/mini-config-clawgym.yaml \
    --batch-size 100 --concurrency $MINI_CONCURRENCY --enable-verifier \
    --shuffle-seed 20261024 --resync-ledger \
    --api-base "$MODEL_BASE_URL" \
    --model-name "$LITELLM_NAME" --api-key "$API_KEY"
}

clawgym_oc_glm() {
  local reg=opencode_sft/registry_clawgym root=$RUN_ROOT
  exec $PY opencode_sft/run_rollouts.py \
    --registry $reg \
    --manifest $reg/manifest-opencode.jsonl \
    --run-root $root/clawgym-opencode --families clawgym \
    --batch-size 50 --concurrency $OPENCODE_CONCURRENCY --enable-verifier \
    --shuffle-seed 20261024 --resync-ledger \
    --proxy-check-url "http://127.0.0.1:$PROXY_PORT/v1/models" \
    --api-base "http://host.docker.internal:$PROXY_PORT/v1" \
    --api-key EMPTY --model-id "$MODEL_NAME" --provider-id rsi_glm
}

ot_mini_glm() {
  local reg=opencode_sft/registry root=$RUN_ROOT
  exec $PY opencode_sft/run_mini_rollouts.py \
    --registry $reg \
    --manifest $reg/manifest-mini.jsonl \
    --run-root $root/openthoughts-mini \
    --families issue,tezos,superuser \
    --min-config opencode_sft/mini_sft/mini-config.yaml \
    --batch-size 100 --concurrency $MINI_CONCURRENCY --shuffle-seed 20261024 --resync-ledger \
    --api-base "$MODEL_BASE_URL" \
    --model-name "$LITELLM_NAME" --api-key "$API_KEY"
}

ot_oc_glm() {
  local reg=opencode_sft/registry root=$RUN_ROOT
  exec $PY opencode_sft/run_rollouts.py \
    --registry $reg \
    --manifest $reg/manifest-opencode.jsonl \
    --run-root $root/openthoughts-opencode \
    --families issue,tezos,superuser \
    --batch-size 50 --concurrency $OPENCODE_CONCURRENCY --shuffle-seed 20261024 --resync-ledger \
    --proxy-check-url "http://127.0.0.1:$PROXY_PORT/v1/models" \
    --api-base "http://host.docker.internal:$PROXY_PORT/v1" \
    --api-key EMPTY --model-id "$MODEL_NAME" --provider-id rsi_glm
}

# stream_function <name> -- the function that runs the stream
stream_function() {
  case "$1" in
    clawgym-mini-glm) echo clawgym_mini_glm ;;
    clawgym-oc-glm)   echo clawgym_oc_glm ;;
    ot-mini-glm)      echo ot_mini_glm ;;
    ot-oc-glm)        echo ot_oc_glm ;;
    *) return 1 ;;
  esac
}

# stream_done <name> -- the marker a finished stream leaves behind
stream_done() {
  case "$1" in
    clawgym-mini-glm) echo "$RUN_ROOT/clawgym/COMPLETE" ;;
    clawgym-oc-glm)   echo "$RUN_ROOT/clawgym-opencode/COMPLETE" ;;
    ot-mini-glm)      echo "$RUN_ROOT/openthoughts-mini/COMPLETE" ;;
    ot-oc-glm)        echo "$RUN_ROOT/openthoughts-opencode/COMPLETE" ;;
    *) return 1 ;;
  esac
}

stream_running() {
  local pidfile="$PIDS/$1.pid" pid state
  [[ -f "$pidfile" ]] || return 1
  pid="$(cat "$pidfile")"
  kill -0 "$pid" 2>/dev/null || return 1
  # A driver that has exited but has not been reaped is a *zombie*, and
  # ``kill -0`` succeeds on a zombie.  Reading the state tells the two
  # apart; without this the supervisor would believe a crashed driver is
  # still running.
  state="$(ps -o state= -p "$pid" 2>/dev/null | tr -d ' ')"
  if [[ "$state" == "Z" ]]; then
    wait "$pid" 2>/dev/null
    return 1
  fi
  return 0
}

# start_stream <name> -- start it, unless it is already running
start_stream() {
  local name="$1" fn
  if stream_running "$name"; then
    log "$name is already running (pid $(cat "$PIDS/$name.pid"))"
    return 0
  fi
  fn="$(stream_function "$name")"
  log "starting $name"
  "$fn" > "$LOGS/$name.log" 2>&1 &
  echo $! > "$PIDS/$name.pid"
  log "$name started as pid $!"
}

# wait_pid <pidfile> <label> -- wait for the driver to exit
wait_pid() {
  local pidfile="$1" label="$2" pid
  if [[ ! -f "$pidfile" ]]; then
    log "$label: no pid file ($pidfile), skipping the wait"
    return 0
  fi
  pid="$(cat "$pidfile")"
  while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  log "$label: driver $pid has exited"
}

# supervise <name...> -- run the streams until every one of them is finished
#
# Every stream is polled, not waited on in turn: a stream that dies is
# started again *while the others keep running*.  Waiting in turn would
# leave a crashed stream idle for as long as the slowest one needs, and the
# whole run is days long.
supervise() {
  local name
  for name in "$@"; do start_stream "$name"; done
  while :; do
    sleep 60
    local unfinished=0
    for name in "$@"; do
      if ! stream_running "$name"; then
        if [[ -f "$(stream_done "$name")" ]]; then
          log "$name is finished"
        else
          log "$name is not running and is unfinished; starting it again"
          start_stream "$name"
        fi
      fi
      [[ -f "$(stream_done "$name")" ]] || unfinished=1
    done
    if [[ $unfinished -eq 0 ]]; then
      log "every stream is finished"
      return 0
    fi
  done
}

# --------------------------------------------------------------------------- #
# the recording proxy
# --------------------------------------------------------------------------- #
# One proxy for the endpoint, in blob mode: the system prompt and the tool
# schemas are stored once, by digest, and the request log carries the
# digests.
start_proxies() {
  if ! curl -sf -m 5 "http://127.0.0.1:$PROXY_PORT/v1/models" >/dev/null 2>&1; then
    log "starting the recording proxy on $PROXY_PORT"
    nohup $PY3 opencode_sft/record_proxy.py \
      --port "$PROXY_PORT" --host 0.0.0.0 \
      --upstream "$PROXY_UPSTREAM" \
      --out "$LOGS/oc-requests.jsonl" \
      --compact --dedupe --blob-dir "$LOGS/oc-blobs" \
      > "$LOGS/record_proxy.log" 2>&1 &
    sleep 3
  fi
}

# --------------------------------------------------------------------------- #
# stage 1: the split
# --------------------------------------------------------------------------- #
clawgym_split() {
  $PY3 opencode_sft/split_tasks.py \
    --registry opencode_sft/registry_clawgym \
    --minimax-share "$MINIMAX_SHARE" \
    --report opencode_sft/clawgym/split_report.json
}
run_stage clawgym-split clawgym_split

# --------------------------------------------------------------------------- #
# stage 2: the two ClawGym streams
# --------------------------------------------------------------------------- #
clawgym_run() {
  start_proxies
  supervise clawgym-mini-glm clawgym-oc-glm
}
run_stage clawgym-run clawgym_run

# --------------------------------------------------------------------------- #
# stage 3: convert and check the ClawGym halves
# --------------------------------------------------------------------------- #
clawgym_convert() {
  $PY3 opencode_sft/clawgym/build_sft_dataset.py \
    --run-root $RUN_ROOT/clawgym \
    --registry opencode_sft/registry_clawgym \
    --out opencode_sft/clawgym/dataset/clawgym-mini-glm-sft.jsonl \
    --min-reward "${MIN_REWARD:-0.0}" --model "$MODEL_NAME"
  $PY3 opencode_sft/clawgym/build_opencode_sft_dataset.py \
    --run-root $RUN_ROOT/clawgym-opencode \
    --registry opencode_sft/registry_clawgym \
    --requests "$LOGS/oc-requests.jsonl" --blob-dir "$LOGS/oc-blobs" \
    --out opencode_sft/clawgym/dataset/clawgym-opencode-glm-sft.jsonl \
    --min-reward "${MIN_REWARD:-0.0}"
  for f in clawgym-mini-glm clawgym-opencode-glm; do
    $PY3 opencode_sft/clawgym/validate_sft.py \
      --dataset opencode_sft/clawgym/dataset/$f-sft.jsonl \
      --registry opencode_sft/registry_clawgym
  done
}
run_stage clawgym-convert clawgym_convert

# --------------------------------------------------------------------------- #
# stage 4: merge the two ClawGym streams
# --------------------------------------------------------------------------- #
clawgym_merge() {
  $PY3 opencode_sft/clawgym/merge_sft.py \
    --dataset opencode_sft/clawgym/dataset/clawgym-mini-glm-sft.jsonl \
    --dataset opencode_sft/clawgym/dataset/clawgym-opencode-glm-sft.jsonl \
    --out opencode_sft/clawgym/dataset/clawgym-sft.jsonl
  $PY3 opencode_sft/clawgym/validate_sft.py \
    --dataset opencode_sft/clawgym/dataset/clawgym-sft.jsonl \
    --registry opencode_sft/registry_clawgym
}
run_stage clawgym-merge clawgym_merge

# --------------------------------------------------------------------------- #
# stage 5: the OpenThoughts families
# --------------------------------------------------------------------------- #
openthoughts_split() {
  $PY3 opencode_sft/split_tasks.py \
    --registry opencode_sft/registry --families issue,tezos,superuser \
    --minimax-share "$MINIMAX_SHARE" \
    --report opencode_sft/split_report_openthoughts.json
}
run_stage openthoughts-split openthoughts_split

openthoughts_run() {
  start_proxies
  supervise ot-mini-glm ot-oc-glm
}
run_stage openthoughts-run openthoughts_run

openthoughts_convert() {
  $PY3 opencode_sft/build_mini_sft_dataset.py \
    --run-root $RUN_ROOT/openthoughts-mini \
    --out opencode_sft/mini_sft/dataset/openthoughts-mini-glm-sft.jsonl
  $PY3 opencode_sft/build_sft_dataset.py \
    --run "$RUN_ROOT/openthoughts-opencode" \
    --requests "$LOGS/oc-requests.jsonl" --blob-dir "$LOGS/oc-blobs" \
    --manifest opencode_sft/registry/manifest.jsonl \
    --out opencode_sft/dataset/openthoughts-opencode-glm
}
run_stage openthoughts-convert openthoughts_convert

log "all requested stages are done"
