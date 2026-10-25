# Reproduce the harness-agent SFT annotation, end to end.
#
#   make setup       -- the two Python environments
#   make images      -- the three Docker images
#   make toolchain   -- the mini-swe-agent bundle that is bind-mounted
#   make registry    -- both task registries, rebuilt from HuggingFace
#   make annotate    -- the rollouts (this is the part that takes days)
#   make dataset     -- convert, validate and merge
#
# Every target is idempotent: it skips the work that is already done, so
# `make annotate` can be re-run after an interruption and picks up where the
# ledger left off.
#
# Configuration lives in `.env` (copy `.env.example`); every variable can
# also be given on the command line.

SHELL := /bin/bash
REPO := $(shell pwd)

PY3 ?= $(REPO)/.venv/bin/python
PY  ?= $(REPO)/.venv312/bin/python

# The endpoint the agents talk to.  The value has to be reachable *from
# inside the trial container*: on macOS and Windows that is
# host.docker.internal, not localhost.
MODEL_BASE_URL ?= http://host.docker.internal:8000/v1
MODEL_NAME     ?= glm5.3-flash
RUN_ROOT       ?= $(HOME)/.rsi-harbor

# The families that are re-annotated.
OT_FAMILIES ?= issue,tezos,superuser

# Export only what the caller actually set.  The defaults above are
# documentation: the environment wins over ``.env`` (the rule lives in
# ``opencode_sft/load_env.sh``), so exporting a default would silently
# override the endpoint and the run root that a researcher put in ``.env``.
export $(foreach v,PY3 PY MODEL_BASE_URL MODEL_NAME RUN_ROOT,$(if $(filter command line environment,$(origin $(v))),$(v)))

.PHONY: help setup images toolchain registry registry-clawgym registry-ot \
        annotate dataset verify check clean-outputs

help:
	@sed -n '2,15p' $(MAKEFILE_LIST) | sed 's/^# \?//'

# --------------------------------------------------------------------------- #
# 0. the environment
# --------------------------------------------------------------------------- #
# Two interpreters, for two reasons:
#
#   .venv      the helper scripts: PyYAML, pyarrow (the parquet rollouts) and
#              the HuggingFace hub client.
#   .venv312   the drivers: Harbor 0.24.0 needs Python 3.12, and so does
#              the mini harness that runs inside the trial containers.
#
# The `rsi` package (the two Harbor adapters) is installed into `.venv` as
# well, but the drivers import it through `PYTHONPATH`, so a driver run from
# a checkout that was never `pip install -ed` still works.
setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.lock
	.venv/bin/pip install -e .
	python3.12 -m venv .venv312 || python3 -m venv .venv312
	.venv312/bin/pip install --upgrade pip
	.venv312/bin/pip install -r requirements.harbor.lock
	@echo "ok"

# --------------------------------------------------------------------------- #
# 1. the Docker images
# --------------------------------------------------------------------------- #
# One image serves both harnesses, because the harness is the variable in
# this experiment and the environment is not: opencode is baked into the
# image, and the mini side mounts its toolchain bundle on top of it.
#
#   rsi/oc-base-clawgym:1      ubuntu 24.04 + python 3.12 + PyYAML + opencode
#   rsi/oc-base-issue-py310:1  debian trixie + python 3.10  + opencode
#   rsi/oc-base-plain:1        ubuntu 24.04 + python 3.12 + opencode
#
# The first two are needed for ClawGym and OpenThoughts/issue, the third for
# OpenThoughts/tezos and superuser.
images:
	docker build -f opencode_sft/images/oc-base-clawgym.Dockerfile -t rsi/oc-base-clawgym:1 .
	docker build -f opencode_sft/images/oc-base-issue-py310.Dockerfile -t rsi/oc-base-issue-py310:1 .
	docker build -f opencode_sft/images/oc-base-plain.Dockerfile -t rsi/oc-base-plain:1 .

# --------------------------------------------------------------------------- #
# 2. the mini-swe-agent toolchain bundle
# --------------------------------------------------------------------------- #
# mini-swe-agent is *not* in the image: it is a bundle that is bind-mounted
# at /root/.local inside every trial container.  The build exports it from a
# throwaway image and verifies it before it replaces the live one.
toolchain:
	bash scripts/rebuild_toolchain_with_mcp.sh

# --------------------------------------------------------------------------- #
# 3. the task registries
# --------------------------------------------------------------------------- #
# 1.3 GB of task directories, rebuilt from two public HuggingFace datasets.
registry: registry-clawgym registry-ot

registry-clawgym:
	bash scripts/build_registries.sh clawgym

registry-ot:
	bash scripts/build_registries.sh openthoughts

# --------------------------------------------------------------------------- #
# 4. the annotation
# --------------------------------------------------------------------------- #
# The long part: every task, on both harnesses, in batches, with a ledger
# so that an interrupted run resumes.  The script supervises the streams:
# a driver that dies is started again.
#
# The stages run in order -- ClawGym first, then OpenThoughts -- and each
# one checks its own output before the next one starts.
annotate:
	bash opencode_sft/run_all_stages.sh

# Only the ClawGym half, for a quick check.
annotate-clawgym:
	STAGES="clawgym-split clawgym-run" bash opencode_sft/run_all_stages.sh

# --------------------------------------------------------------------------- #
# 5. the dataset
# --------------------------------------------------------------------------- #
# The rollouts become an OpenAI-compatible SFT dataset: one JSON object per
# line, with `messages` (the model-side conversation) and `conversations`
# (the strict user/assistant view), and the three provenance keys
# `dataset` / `model` / `harness`.
#
# The rollouts that trip a discard rule are archived under `invalid/`, not
# deleted.
dataset:
	STAGES="clawgym-convert clawgym-merge openthoughts-convert" \
		bash opencode_sft/run_all_stages.sh

# --------------------------------------------------------------------------- #
# 6. the checks
# --------------------------------------------------------------------------- #
# The validator is an independent program: it re-reads the dataset from
# disk and checks it against the registry, not against the generator's own
# in-memory objects.
verify:
	$(PY3) opencode_sft/clawgym/validate_sft.py \
		--dataset opencode_sft/clawgym/dataset/clawgym-sft.jsonl \
		--registry opencode_sft/registry_clawgym

# A cheap end-to-end smoke test: one task per harness, converted and
# validated.  It is what to run first when something looks wrong.
check:
	bash scripts/smoke_reproduce.sh

# --------------------------------------------------------------------------- #
# housekeeping
# --------------------------------------------------------------------------- #
clean-outputs:
	rm -rf opencode_sft/clawgym/dataset opencode_sft/mini_sft/dataset \
		opencode_sft/dataset
