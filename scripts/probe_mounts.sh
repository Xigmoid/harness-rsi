#!/usr/bin/env bash
# Probe which host paths Docker Desktop can bind-mount on this machine.
#
# A single pending mount request wedges the whole Docker daemon on macOS, so
# every probe is run with a hard timeout and the daemon is restarted after a
# hang.  The result is a short list of paths the study may use for the Harbor
# job output (Harbor bind-mounts the trial's log directories into the
# container, so the output directory *has* to be mountable).
#
#   scripts/probe_mounts.sh
#
set -uo pipefail

run_with_timeout() {
  local secs="$1"; shift
  "$@" &
  local pid=$!
  local waited=0
  while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt "$secs" ]; do
    sleep 5; waited=$((waited + 5))
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    echo "HANG (${secs}s)"
    return 1
  fi
  wait "$pid" 2>/dev/null
  return 0
}

restart_docker() {
  docker ps -aq 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1
  docker desktop stop >/dev/null 2>&1
  sleep 4
  pkill -f "Docker Desktop" >/dev/null 2>&1
  pkill -f com.docker.backend >/dev/null 2>&1
  sleep 3
  open -a Docker >/dev/null 2>&1
  for _ in $(seq 1 40); do
    docker info >/dev/null 2>&1 && break
    sleep 3
  done
  sleep 5
}

probe() {
  local label="$1" src="$2"
  echo "--- $label : $src"
  if [ ! -e "$src" ]; then
    echo "SKIP (does not exist)"
    return
  fi
  local out
  out="$(run_with_timeout 90 docker run --rm -v "$src:/probe" ubuntu:24.04 ls /probe 2>&1 | head -3)"
  if [ "${out}" = "HANG (90s)" ]; then
    echo "HANG"
    restart_docker
  else
    echo "OK: $(echo "$out" | tr '\n' ' ' | cut -c1-90)"
  fi
}

echo "=== docker ready? ==="
for _ in $(seq 1 40); do docker info >/dev/null 2>&1 && break; sleep 3; done
docker info >/dev/null 2>&1 && echo "docker up" || { echo "docker down"; exit 1; }

mkdir -p /tmp/probe_dir && echo hi > /tmp/probe_dir/a.txt
mkdir -p "$HOME/probe_dir" && echo hi > "$HOME/probe_dir/a.txt"
mkdir -p "$HOME/Documents/probe_dir" && echo hi > "$HOME/Documents/probe_dir/a.txt"

probe "plain (no mount)"  ""
probe "/tmp"              "/tmp/probe_dir"
probe "/private/tmp"      "/private/tmp/probe_dir"
probe "HOME"              "$HOME/probe_dir"
probe "HOME/Documents"    "$HOME/Documents/probe_dir"
