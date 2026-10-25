# Reproducing the harness-agent SFT annotation

Two agent harnesses -- **mini-swe-agent** and **opencode** -- are run over
the same tasks, in the same environment, against the same model, and their
trajectories are exported as an **OpenAI-API-compatible** SFT dataset.  One
harness is the variable in this experiment; the environment and the model
are held fixed, so the two halves of the dataset are directly comparable.

This document is the path from a fresh clone to the finished dataset.  Every
command is meant to be run from the repository root.

```
git clone https://github.com/Xigmoid/harness-rsi.git
cd harness-rsi
```

---

## 1. What the output is

One JSON object per line, OpenAI Chat Completions shaped:

```json
{
  "dataset": "ClawGym",
  "model": "GLM-5.3-Flash",
  "harness": "Mini",
  "model_id": "glm5.3-flash",
  "model_provider": "rsi_glm",
  "messages":     [ ... the conversation the model actually received ... ],
  "conversations": [ ... the same conversation, strict user/assistant ... ],
  "instruction": "...",
  "reward": 0.94,
  "task": "clawgym-000217-backtest-expert-0-1-0-7d5d53a10803"
}
```

* `messages` is the **model-side** view: the exact message list the agent
  sent to the endpoint, plus the model's final answer (which was never sent
  back).  It is verified byte for byte against a full packet capture --
  see `common/EXECUTION_PLAN.md`.
* `conversations` is the **training** view: the same content with the tool
  calls inlined and the reasoning folded into the content, so that it
  alternates user/assistant strictly.
* The three keys `dataset` / `model` / `harness` are what a trainer filters
  on; `model_id` and `model_provider` are what an auditor follows.

The rollouts that trip a discard rule are **archived, not deleted**: they go
to `invalid/` next to the dataset, with the reason, together with a
`.skipped.jsonl` listing every skipped rollout and why.

---

## 2. Prerequisites

| | |
|---|---|
| Docker | with 10 GB of memory for the VM (`docker info` shows the limit) |
| Python | 3.12 for the drivers, 3.10+ for the helper scripts |
| Disk | ~6 GB for the images, the toolchain bundle, the registries and the run output |
| Network | the model endpoint, and HuggingFace (the task data) |
| Time | the annotation itself is **days**: 15,402 ClawGym tasks + 23,225 OpenThoughts tasks, on two harnesses |

The endpoint has to be reachable **from inside the trial containers**.  On
macOS and Windows that is `host.docker.internal`, not `localhost`.

---

## 3. The steps

### Step 0 -- the environment

```bash
cp .env.example .env      # then edit it: the endpoint, the model, the run root
make setup
```

Two virtual environments, for two reasons: `.venv` carries the helper
scripts' dependencies (PyYAML, pyarrow, the HuggingFace hub client), and
`.venv312` carries Harbor 0.24.0, which needs Python 3.12.  The `rsi`
package -- the two Harbor adapters -- is imported by the drivers through
`PYTHONPATH`, so it does not have to be installed at all.

### Step 1 -- the Docker images

```bash
make images
```

Three images, ~830 MB each.  **One image serves both harnesses**: opencode
is baked into the image, and the mini side mounts its toolchain bundle on
top of it, so the two halves of the split really do run in the same
environment.

| image | for |
|---|---|
| `rsi/oc-base-clawgym:1` | every ClawGym task |
| `rsi/oc-base-issue-py310:1` | OpenThoughts `issue` (Debian trixie + Python 3.10) |
| `rsi/oc-base-plain:1` | OpenThoughts `tezos` and `superuser` |

### Step 2 -- the mini-swe-agent bundle

```bash
make toolchain
```

mini-swe-agent is not in the image: it is a bundle that is bind-mounted at
`/root/.local` inside every trial container.  The build exports it from a
throwaway image, verifies it, and only then replaces the live one.

### Step 3 -- the task registries

```bash
make registry
```

1.3 GB of task directories, rebuilt from two **public** HuggingFace
datasets:

| registry | source | size |
|---|---|---|
| `work/registry_clawgym` | `RUC-AIBOX/ClawGym-Task` (`syn_task.jsonl`) | 15,402 tasks |
| `work/registry` | `open-thoughts/OpenThoughts-Agent-SFT-31.6K` (`train.parquet`) | 23,225 tasks (`issue`, `tezos`, `superuser`) |

The OpenThoughts dataset ships *rollouts*, not tasks, so the task
instructions have to be cut out of the terminus-2 transcripts byte-identically
before the environments can be built; `common/registry/extract_tasks.py` does
that.

The registries are not in git because they are derived data -- this step
rebuilds them in a few minutes.

### Step 4 -- the annotation

```bash
make annotate
```

This is the long part.  `common/run_all_stages.sh` runs the stages in
order -- ClawGym first, then OpenThoughts -- and each one checks its own
output before the next one starts.

* Each harness gets **half** of every dataset, by a hash of the task name:
  the split is deterministic and independent of the row order, so it is the
  same on every machine and in every re-run.
* The work is done in batches, and every finished trial goes into a
  **ledger**.  An interrupted run resumes from the ledger; the drivers
  re-derive it from the trial directories on disk, so even a batch that was
  cut in half is not re-run.
* A **supervisor** polls the streams every minute and starts a driver that
  dies again.  A stream that is finished leaves a `COMPLETE` marker, and
  the supervisor stops only when every stream has one.

To check the whole path in about ten minutes before committing to the days:
`make check` (one task per harness, converted and validated).

### Step 5 -- the dataset

```bash
make dataset
make verify
```

The validator is an **independent** program: it re-reads the dataset from
disk and checks it against the registry, not against the generator's
in-memory objects.  It also has a fault-injection mode -- it is fed
deliberately broken datasets and has to catch every one of them
(`common/validate_sft.py`).

The finished dataset is published as a **GitHub Release** rather than in
git, because it is gigabytes:

```bash
gh release download clawgym-sft --repo Xigmoid/harness-rsi
```

---

## 4. Configuration

Everything machine-specific is in `.env` (see `.env.example`).  The
variables that matter:

| variable | meaning |
|---|---|
| `MODEL_BASE_URL` | the endpoint, **as seen from inside the container** |
| `PROXY_UPSTREAM` | the same endpoint as seen from the host, where the recording proxy runs.  It differs only when the endpoint is on the machine that also runs Docker: the container reaches it at `host.docker.internal`, the host at `127.0.0.1` |
| `MODEL_NAME` | the model id, as the endpoint knows it |
| `API_KEY` | the key, or `EMPTY` for an endpoint that does not check one |
| `RUN_ROOT` | where the trials run.  It has to **survive a reboot**: macOS wipes `/tmp` on every boot, and a multi-day run does not survive losing its ledger.  The default is `~/.rsi-harbor` |
| `MINI_CONCURRENCY` / `OPENCODE_CONCURRENCY` | how many trials run at once, per harness |
| `MINIMAX_SHARE` | the share of each half annotated by the second provider (MiniMax).  `0` = every task goes to the main endpoint |

**A key never reaches a job file.**  The job file goes into the container
and the trial log, so the real key is injected by the recording proxy on
the way out, and the job file only ever holds the placeholder `EMPTY`.

**The environment wins over `.env`.**  `.env` is a default, not an override:
a value that was already in the environment when the script started beats
the file, so

```bash
RUN_ROOT=/tmp/scratch make annotate      # runs in /tmp/scratch
make annotate                             # runs in the run root from .env
```

do what they say.  The rule (and the list of variables it covers) lives in
`common/load_env.sh`, and every script that reads `.env` goes through
it.

---

## 5. The traps this implementation already handles

These are the ones that cost the most time to find.  They are written up
in full, with the evidence, in `common/EXECUTION_PLAN.md`.

| trap | what it looks like | what it does here |
|---|---|---|
| Docker's default address pool is sixteen networks wide | every new trial dies with `all predefined address pools have been fully subnetted`, which looks like an agent failure and is not one | every trial joins one shared network, created once and never taken down |
| the `harbor job` is a child of the driver and does not die with it | the driver is killed, the job keeps starting trials nobody harvests, and the containers pile up until the memory ceiling | the driver kills the stale job and removes the stale containers before each batch |
| `$!` for a backgrounded shell *function* is the wrapper's pid | the "kill the driver" kills a wrapper, the real driver survives, and the same stream runs three times over | the stream functions end in `exec`, so the pid file holds the driver itself |
| a killed trial leaves a *running* container | `docker container prune` only removes stopped ones, and forty orphans cost 4 GB | the containers of the earlier batches are removed explicitly, scoped to the run root |
| the verifier's directory is visible to the agent | the agent reads `reward/test.py` and the trajectory teaches it to peek at the scorer | the rollouts that touch `reward/` are filtered at conversion time and archived under `invalid/` |
| opencode's `tool_calls.arguments` is JavaScript's `JSON.stringify` | Python's `json.dumps` puts a space after every `:` and `,` -- the same object, a different token sequence, so the data is not the bytes the model saw | the converter serialises it the way the wire does |
| opencode's reasoning is its own field | folding it into the content makes the sample read the same while the tokens differ | `messages` keeps `reasoning_content` separate; only `conversations` folds it in |
| opencode's `bash` tool is not a persistent shell | `cd` and `export` do not survive a call | the prompt says so, and the agent is given the `workdir` parameter instead |
| mini's endpoint lives in the container's config, not in the host environment | pointing the run at a second endpoint silently keeps using the first, and the labels say otherwise | the endpoint is written into the config that goes into the container, and there is a hard check that refuses a key file with a remote endpoint |

---

## 6. What this repository contains

Only the annotation pipeline.  Four pieces:

| | |
|---|---|
| `opencode_sft/` | the opencode half: its driver, its converters, its wire check, the launchd agents, and a README that documents what opencode does |
| `mini_sft/` | the mini-swe-agent half: the same file names on the same layout, plus the config the container actually reads |
| `common/` | what the two halves share: the orchestrator, the recording proxy, the schema, the split, the validator, the merger, the registry builders, the base images, the one-off tools, and the notes |
| `rsi/` | the two Harbor adapters (`mini-swe-agent` and `opencode`) and the paths they share with the container side |
| `scripts/`, `datasets/base/toolchain/`, `audit/` | the registry builders, the mini-swe-agent bundle's Dockerfile, and the audit of what Harbor and mini-swe-agent actually do |

The two harness directories are laid out the same way on purpose: `run_rollouts.py`,
`build_sft_dataset.py`, `clawgym/build_sft_dataset.py` and `verify_wire.py` exist on
both sides, so a change to one half can be read against the other without hunting
for the corresponding file.

Everything generated lives under `work/` and is **not** in git: the task
registries, the split manifests, the trials, the converted halves, and the merged
dataset.  `make registry` rebuilds the first two from public sources,
`make annotate` produces the third, and `make dataset` the last two.

The RSI framework the annotation grew out of (a typed-configuration search over
agent harnesses, with its own optimizer, ledger and benchmark suites) is **not**
part of this repository: it is a separate research line, and it shares nothing
with the annotation except the two Harbor adapters above.
