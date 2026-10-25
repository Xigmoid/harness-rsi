#!/usr/bin/env bash
# Rebuild both task registries from the public HuggingFace datasets.
#
# Why this exists
# ---------------
# The two registries are 1.3 GB of task directories: 15,402 for ClawGym and
# 23,225 for OpenThoughts (issue / tezos / superuser).  They are **not** in
# git, because they are derived data -- both can be rebuilt from public
# datasets in a few minutes, and a researcher should not have to clone a
# gigabyte to get them.
#
#   RUC-AIBOX/ClawGym-Task
#       syn_task.jsonl -- 15,402 tasks, each with its prompt and its files
#
#   open-thoughts/OpenThoughts-Agent-SFT-31.6K
#       train.parquet -- 31,600 published rollouts over 29,765 tasks
#
# The OpenThoughts dataset ships *rollouts*, not tasks, so there is one more
# step: the task instructions have to be cut out of the terminus-2
# transcripts, byte-identically, before the environments can be built.
# ``extract_tasks.py`` does that.
#
# Usage::
#
#     bash scripts/build_registries.sh            # both
#     bash scripts/build_registries.sh clawgym    # only ClawGym
#     bash scripts/build_registries.sh openthoughts
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY3="${PY3:-$REPO/.venv/bin/python}"
CACHE="${CACHE:-$REPO/.cache}"

CLAWGYM_REPO="${CLAWGYM_REPO:-RUC-AIBOX/ClawGym-Task}"
OT_REPO="${OT_REPO:-open-thoughts/OpenThoughts-Agent-SFT-31.6K}"

which="${1:-all}"

log() { printf '\n==> %s\n' "$*"; }

# --------------------------------------------------------------------------- #
# ClawGym: 15,402 tasks
# --------------------------------------------------------------------------- #
build_clawgym() {
  log "ClawGym: downloading $CLAWGYM_REPO"
  mkdir -p "$CACHE/clawgym"
  $PY3 - "$CLAWGYM_REPO" "$CACHE/clawgym" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

repo, dest = sys.argv[1], Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)
path = hf_hub_download(
    repo_id=repo, filename="syn_task.jsonl", repo_type="dataset",
    local_dir=str(dest),
)
print("downloaded", path)
PY

  log "ClawGym: building the registry"
  $PY3 opencode_sft/clawgym/build_registry.py \
    --jsonl "$CACHE/clawgym/syn_task.jsonl" \
    --out opencode_sft/registry_clawgym \
    --clean

  log "ClawGym: writing the split manifests"
  $PY3 opencode_sft/split_tasks.py \
    --registry opencode_sft/registry_clawgym \
    --report opencode_sft/clawgym/split_report.json
}

# --------------------------------------------------------------------------- #
# OpenThoughts: 23,225 tasks in three families
# --------------------------------------------------------------------------- #
build_openthoughts() {
  log "OpenThoughts: extracting the tasks and their instructions from $OT_REPO"
  $PY3 opencode_sft/extract_tasks.py \
    --repo "$OT_REPO" \
    --out opencode_sft/tasks \
    --clean

  log "OpenThoughts: building the registry"
  # Only the three families that are re-annotated here.  The SWE-Smith
  # family needs the per-instance SWE-Smith images and mirror caches, and
  # it is not part of this re-annotation.
  $PY3 opencode_sft/build_registry.py \
    --instructions opencode_sft/tasks/instructions \
    --index opencode_sft/tasks/index.jsonl \
    --families issue,tezos,superuser \
    --out opencode_sft/registry \
    --clean

  log "OpenThoughts: one shared Docker network for every trial"
  # Every trial gets its own compose project, and every compose project its
  # own bridge network.  Docker's default address pool is sixteen networks
  # wide, so without this the pool is exhausted after a few dozen trials
  # and *every* new trial dies with "all predefined address pools have been
  # fully subnetted" -- which looks like an agent failure and is not one.
  $PY3 opencode_sft/add_shared_network_compose.py \
    --registry opencode_sft/registry \
    --families issue,tezos,superuser

  log "OpenThoughts: writing the split manifests"
  $PY3 opencode_sft/split_tasks.py \
    --registry opencode_sft/registry --families issue,tezos,superuser \
    --report opencode_sft/split_report_openthoughts.json
}

case "$which" in
  clawgym)      build_clawgym ;;
  openthoughts) build_openthoughts ;;
  all)          build_clawgym; build_openthoughts ;;
  *) echo "usage: $0 [clawgym|openthoughts|all]" >&2; exit 2 ;;
esac

log "done"
