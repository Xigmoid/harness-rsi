# opencode_sft — re-annotating with opencode

This directory is the **opencode half** of the re-annotation.  The
mini-swe-agent half is [`../mini_sft/`](../mini_sft/); what the two share is in
[`../common/`](../common/).

Two public task sets are re-annotated, each split in half by a hash of the
task name so that the two harnesses get comparable tasks:

| dataset | what it ships | registry |
| --- | --- | --- |
| [`RUC-AIBOX/ClawGym-Task`](https://huggingface.co/datasets/RUC-AIBOX/ClawGym-Task) | **tasks** — the rollouts are produced here | `work/registry_clawgym/` |
| [`open-thoughts/OpenThoughts-Agent-SFT-31.6K`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-SFT-31.6K) | **published rollouts** (31,600), annotated with terminus-2 inside Harbor | `work/registry/` |

For the OpenThoughts half the job is to re-run the *same tasks* with
**opencode** as the rollout harness, keeping the task instructions
byte-identical and keeping the full message list of every rollout.

The ClawGym half is described in [`../common/CLAWGYM.md`](../common/CLAWGYM.md):
there the tasks ship without trajectories, so the rollouts are new, and the
task's own verifier is what says whether a rollout is worth training on.

## Layout

This directory is the **opencode half** of the re-annotation.  The
mini-swe-agent half is [`../mini_sft/`](../mini_sft/) and has the same file
names on the same layout, so the two can be read side by side.  What the two
halves share lives in [`../common/`](../common/).

```
opencode_sft/
├── README.md
├── run_rollouts.py             # the driver: batches, a ledger, resume
├── build_sft_dataset.py        # the rollouts → the SFT dataset (two views)
├── verify_wire.py              # the dataset vs. the recorded wire, key by key
├── regression_converter.py     # the converter for the regression set
├── clawgym/
│   ├── build_sft_dataset.py    # the ClawGym trials → the SFT dataset, by reward
│   └── ...                     # (the rest of the ClawGym half is in ../common/)
├── launchd/                    # the proxy and the run, as launchd agents
│   ├── com.rsi.opencode.proxy.plist
│   └── com.rsi.opencode.rollouts.plist
└── dataset/                    # what this half converts to
    ├── pilot/                  # the converted pilot dataset
    │   ├── dataset.jsonl       # one row per rollout
    │   ├── summary.json        # counts and the validation report
    │   └── verification.json   # the independent check
    └── clawgym-opencode-glm-sft.jsonl   # the ClawGym half of the dataset
```

The mini-vs-opencode comparison — the methodology, the reference converter and
the wire-parity check — is in [`../common/pilot/`](../common/pilot/); its
captured artefacts are in `work/pilot/terminal-5task/`.

The shared pieces, and what they are for:

| file | role |
| --- | --- |
| `rsi/opencode_agent.py` | Harbor agent adapter: pins the opencode build, skips the per-trial npm install when the image already has it, runs the task's own setup script, keeps opencode's session store out of the mounted log directory, exports the raw session next to the ATIF trajectory, and records the effective command line |
| `common/run_all_stages.sh` | the orchestrator: it supervises both harnesses, so the two halves never run the same task twice |
| `common/record_proxy.py` | the recording proxy: it writes the wire traffic to disk and injects the API key on the way out |
| `common/split_tasks.py` | the hash split: half of every dataset goes to each harness |
| `common/validate_sft.py` | the dataset validator, with its own fault injection |
| `common/merge_sft.py` | merges the two halves into one dataset |
| `common/images/` | the base images (opencode is baked into them) |
| `common/registry/` | the registry builders |
| `common/tools/` | the one-off tools that were written along the way |

The output of a run lives **outside** the repository, under the run root
(`~/.rsi-harbor` by default), because Harbor has to bind-mount the trial
directories into the containers and `/private/tmp` does not survive a reboot:

```
~/.rsi-harbor/
├── clawgym-opencode/            # the ClawGym half of this harness
│   ├── batch-0001.yaml          # the job the driver wrote for this batch
│   ├── batch-0001/<timestamp>/<trial>/{agent/,artifacts/,result.json,...}
│   ├── progress.jsonl           # the ledger: one line per finished trial
│   └── COMPLETE                 # the marker that says the stream is finished
├── openthoughts-opencode/       # the OpenThoughts half of this harness
├── logs/                        # the recording proxy's log and its blobs
└── pids/                        # the pid files the supervisor reads
```

## Scaling to 31,600 tasks: the family-shared base image

The obvious way to give each of the 29,765 tasks its own environment is one
Docker image per task. That is 29,765 image builds before the first agent
starts, and it is what this design exists to avoid.

The environment of a task is instead split in two:

* the part that is **the same for a whole family** goes into the family's
  base image, and
* the part that **differs per task** is applied at run time, by a script the
  task mounts into its own container.

Harbor is what makes the first half possible: `EnvironmentConfig.docker_image`
names a pre-built image, and when it is set the task needs no
`environment/Dockerfile` at all — Harbor runs the *prebuilt* compose template
(`image: ${PREBUILT_IMAGE_NAME}`) and never builds anything.

```toml
# every task's task.toml
[environment]
docker_image = "rsi/oc-base-swesmith-py:1"
```

### The five images

| image | base | who runs on it |
| --- | --- | --- |
| `rsi/oc-base-issue:1` | ubuntu 24.04 + python3.12 + opencode | the IssueTasks family (8,374 rows) |
| `rsi/oc-base-tezos:1` | *the same image, second tag* | the Tezos family (7,197 rows) |
| `rsi/oc-base-superuser:1` | *the same image, third tag* | the SuperUser family (7,654 rows) |
| `rsi/oc-base-swesmith-py:1` | ubuntu 22.04 + a python 3.10 environment named `testbed` + opencode | the SWE-Smith family, Python repositories (7,394 tasks) |
| `rsi/oc-base-swesmith-go:1` | debian 12 + Go at `/usr/local/go` + opencode | the SWE-Smith family, Go repositories (298 tasks) |

Five images for 31,600 rollouts.

### Three of the four families need no per-task setup at all

The published IssueTasks / Tezos / SuperUser rollouts started in an **empty**
`/app` on ubuntu 24.04. That is not an inference from the upstream Dockerfile
alone — the published rollout of `superuser-005067_copy0007` prints it:

```
root@e97c6cff-...:/app# ls -la /app/
total 0
drwxr-xr-x 2 root root  10 Apr 10 17:47 .
drwxr-xr-x 1 root root 102 Apr 16 19:07 ..
```

before it installs anything. The upstream generator agrees
(`data/swesmith/task_templates.py` builds these tasks as `FROM ubuntu:24.04`
+ `WORKDIR /app` + `python3 python3-pip` and nothing else).

About 5 % of the published rollouts show files in `/app` — `bench.sh`,
`__pycache__`, a downloaded `mediawiki-1.7.1.tar.gz`. Those are files **the
agent created** before the listing, not files of the container;
`common/tools/analyze_app_seed.py` is the script that measures this.

So for these three families the shared image *is* the environment, and the
task directory holds nothing but the instruction.

### SWE-Smith is the one family that needs an overlay

The published SWE-Smith rollouts ran in the official SWE-Smith images with
the repository at `/testbed`, checked out at the instance's bug-injected
commit. The upstream task template renders exactly that:

```
FROM {official image}
RUN git fetch && git checkout {instance_id}
```

and the SWE-Smith **mirror repository has one branch per instance**, named
exactly like the instance id. So the per-task part is a checkout, not an
image:

```bash
# the task's own environment/setup.sh
git clone --shared /rsi/mirrors/<slug>.git /testbed   # 1 second
git checkout -f <instance_id>
uv pip install -e /testbed
```

The script is mounted into the container by the task's own
`environment/docker-compose.yaml`:

```yaml
services:
  main:
    volumes:
      - ./setup.sh:/rsi/setup.sh:ro
      - /private/tmp/rsi-swe-mirrors:/rsi/mirrors:ro
```

and `RsiOpenCode.install()` runs it inside the container, before opencode
starts. A setup that fails fails the trial — a half-prepared environment is
not a rollout.

### Finding the instances was exact, not approximate

The dataset's SWE-Smith task ids are renumbered indices that say nothing
about the environment, so the mapping had to be recovered from the
instruction. The published instruction is the SWE-Smith `problem_statement`
with the markdown fences normalised, and **all 7,693 unique SWE-Smith tasks
match exactly** (`common/tools/map_swesmith.py`). One task in 7,693 has three
instances with the same problem statement; the tie was broken with the
published rollout itself — the instance whose patch touches a file the
original agent actually opened.

### The mirror cache is what makes the overlay cheap

Cloning the pandas mirror from GitHub takes **53 seconds**. The 150 mirrors
behind the 7,693 SWE-Smith tasks are cloned **once** on the host (648 MB in
total) and bind-mounted read-only into the containers, where
`git clone --shared` borrows the object store instead of copying it: **one
second**, for pandas included.

### The SWE-Smith family turned out to be two families

The published rollouts do not agree on the base image, and the disagreement
follows the language of the repository:

| | Python repositories | Go repositories |
| --- | --- | --- |
| base | ubuntu 22.04 (jammy) | debian 12 (bookworm) |
| evidence | `Hit:1 http://archive.ubuntu.com/ubuntu jammy InRelease` | `Hit:1 http://deb.debian.org/debian bookworm InRelease` |
| prompt | `(testbed) root@...:/testbed#` | `root@...:/testbed#` — no environment |
| toolchain | a conda environment named `testbed`, python 3.10.16 / 3.10.18 / 3.11.11 / 3.11.13 | Go at `/usr/local/go`, **not on `PATH`** |

The Go detail is the sharp one: the published rollout answers
`bash: go: command not found` and then finds `/usr/local/go/bin/go` with a
`find`, so the agent has to add it to `PATH` itself. The Go image reproduces
that exactly — Go is installed, and it is deliberately *not* on `PATH`.

`common/tools/derive_swe_env.py` reads both facts out of the published
rollouts (the language from the repository's tree, the python version from
the first `Python x.y.z` the rollout printed) and the registry builder picks
the image and the setup script per task.

## Running it at scale

### The driver: batches, a ledger, and a resume

`opencode_sft/run_rollouts.py` runs the Harbor job in **batches** rather than
as one eight-day process, and writes every finished trial to a ledger
(`progress.jsonl`) as it lands. A laptop sleep, a Docker restart or a
cancelled shell then costs one batch, not the run: the driver reads the
ledger before it starts and skips what is already done.

```bash
python opencode_sft/run_rollouts.py \
    --registry work/registry \
    --manifest work/registry/manifest-opencode.jsonl \
    --run-root ~/.rsi-harbor/openthoughts-opencode \
    --families issue,tezos,superuser \
    --batch-size 50 --concurrency 10 \
    --proxy-check-url http://127.0.0.1:8910/v1/models \
    --api-base http://host.docker.internal:8910/v1
```

`--shard i/n` hands out the tasks by a stable hash of the task name, so two
shards never pick the same task — that is the way to spread the run over
several machines.

### The proxy and the run are launchd agents

A process started from a tool call dies with it, and this run lasts days.
Both the recording proxy and the driver are therefore
[launchd agents](launchd/):

```bash
cp opencode_sft/launchd/com.rsi.opencode.proxy.plist ~/Library/LaunchAgents/
cp opencode_sft/launchd/com.rsi.opencode.rollouts.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.rsi.opencode.proxy.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.rsi.opencode.rollouts.plist
```

`KeepAlive` restarts them if they die, and the driver is restartable anyway
because of the ledger.

### Everything that has to survive a reboot lives under the run root

`/private/tmp` is **not** a safe place for the output of a run that lasts
days. The host rebooted once during the pilot and took the whole tree with
it, including the recording proxy's request log — that log holds the system
prompt and the tool schemas of every rollout, and they exist nowhere else.

So the run root is `~/.rsi-harbor` (`$RUN_ROOT`), outside the repository and
outside `/private/tmp`: the proxy writes to `~/.rsi-harbor/logs/`, the trials
go to `~/.rsi-harbor/clawgym-opencode/` and
`~/.rsi-harbor/openthoughts-opencode/`, and only genuinely disposable scratch
(the mirror cache, the uv cache) stays in `/private/tmp`.

The **ledger** is the other thing that must survive: `progress.jsonl`, one
line per finished trial, next to the batches. A driver that dies restarts and
resumes from it, so a reboot costs the trials that were in flight and nothing
else.

### One Docker network for all 31,600 trials

Harbor gives every trial its own compose project, and every compose project
its own bridge network. Those networks come out of Docker's **default
address pool**, which is `172.17.0.0/12` carved into /24s — sixteen
networks. A trial that ends cleanly takes its network with it, but a trial
that fails, or a job that is cancelled, leaks it, and a leaked network is
never reused.

Once the pool is empty **every new trial dies**, and it dies in a way that
looks like an agent failure:

```
failed to create network random-07-03__6yo9gj7__env_default: all predefined
address pools have been fully subnetted
```

which surfaces as a `RuntimeError` from the *environment*, not from the
model — and it cost the first full-batch test four trials before the real
cause was found.

The obvious fix — widening the pool with `default-address-pools` in
`~/.docker/daemon.json` — **does not work on this host**: Docker Desktop
4.38.0 then starts its VM but dockerd never answers the API, and the engine
never comes up at all. So the pool is left alone and the leak is removed at
the other end: **every trial joins one shared network**, created once and
never taken down.

```yaml
# every task's environment/docker-compose.yaml
networks:
  default:
    name: rsi-shared
    external: true
```

`external: true` means `docker compose down` — which is what Harbor runs
when it cleans a trial up — leaves the network alone. Sharing it is safe:
Harbor reaches the container with `docker compose exec`, which finds it by
its compose labels, not by DNS, and the agent talks to the model through
`host.docker.internal`, which does not go through the network at all.

`docker network prune` cannot be used to clean up the leaks: it would take
the shared network with it whenever no container happens to be attached,
which is most of the time between batches. The driver therefore removes the
leaked networks by name
([`shared_network.py`](shared_network.py)).

### The unit of work is the dataset *row*, not the task

The published dataset has 31,600 rows over 29,765 tasks. Those are **31,600
independent rollouts**: 1,432 tasks were rolled out more than once, and 766
of those repeats used a *different* instruction (a rewritten problem
statement, or one with a `CONSTRAINT:` footer added or removed). Two rows of
the same task with the same instruction still carry two different
conversations — they are two rollouts, not a duplicated row.

So the registry emits **one Harbor task per dataset row**: the first row of a
task keeps the task's own name, the further rows carry the row index
(`<task>__row00042`). That keeps the output joined onto the published dataset
one row to one row.

## The three steps

### 1. Extract the tasks and their instructions

```bash
.venv312/bin/python common/registry/extract_tasks.py            # full run
.venv312/bin/python common/registry/extract_tasks.py --limit 200   # quick check
```

The dataset stores every rollout as a chat transcript whose **first message**
is the terminus-2 system prompt with the task description glued to the end of
it, between two template markers:

```
<terminus-2 prompt>

Task Description:
<the task instruction>          <-- what has to stay identical

Current terminal state:
Current Terminal Screen:
...
```

`extract_tasks.py` cuts the instruction out of that first message and writes
one file per task containing **only the original text** — nothing added,
nothing reformatted. The sha256 of each instruction is recorded in
`index.jsonl` and `instructions.jsonl`, so a later run can prove that the
instructions were not touched.

### 2. Rebuild the task environments

The dataset ships **no environment**: only task ids and instructions. The four
task families — SWE-Smith, StackExchange-SuperUser, StackExchange-Tezos,
IssueTasks — have to be turned back into Harbor tasks
(`task.toml` + `instruction.md` + `environment/` + `tests/`), which is what
`common/registry/build_registry.py` writes and what the driver points at.

The OpenThoughts registry is rebuilt from the public parquet by
`make registry`; the ClawGym one, from the public `syn_task.jsonl`, by
`common/registry/build_clawgym_registry.py`.  Both end in `work/registry/`
and `work/registry_clawgym/`.

The upstream generators are the reference for this:

| family | upstream generator | image |
| --- | --- | --- |
| SWE-Smith | `data/pure_scale/swesmith_100k.py` → `data/swesmith/generate_with_plain_docker.py` | one shared `ubuntu:24.04` + python3; the agent clones the repo and checks out the instance commit at run time |
| StackExchange SuperUser / Tezos | `data/stackexchange/generate_superuser.py`, `data/synthetic_harden/tezos/askllm_hardening.py` | one standard Dockerfile per family |
| IssueTasks | `data/patchers/patch_exp_rle_github_issue_tasks.py` | a fixed `python3 + pytest` image |

### 3. Run the rollouts

```bash
# one family, one harness -- this is what the driver runs
.venv312/bin/python opencode_sft/run_rollouts.py \
    --registry work/registry \
    --manifest work/registry/manifest-opencode.jsonl \
    --run-root "$HOME/.rsi-harbor/openthoughts" \
    --families issue,tezos,superuser \
    --batch-size 50 --concurrency 10
```

The driver writes the Harbor job for the batch next to the trials
(`<run-root>/batch-NNNN.yaml`) and runs it with `PYTHONPATH=.` set, because the
job names the agent class by import path.  The model endpoint is pinned inside
that job file, so no `OPENAI_*` variable has to be exported.

## The pilot: one task per family

`build_pilot_tasks.py` generates the task directories from the extracted
instructions, so the instruction the agent sees is byte-identical to the
published one (the sha256 of each is in `work/pilot/manifest.json`).

| family | task | environment |
| --- | --- | --- |
| SWE-Smith | `swesmith-00000_copy0001` | the repository at the instance commit, cloned from `swesmith/kayak__pypika.1c9646f0` on the branch `kayak__pypika.1c9646f0.lm_rewrite__mgynlcgd` |
| IssueTasks | `issue-000000_copy0001` | astropy under `/app` (the instruction names paths under `/app/`) |
| Tezos | `tezos-0000` | a plain shell |
| StackExchange-SuperUser | `superuser-000000_copy0000` | a plain shell with GNU parallel |

The SWE-Smith instance was found by **matching the instruction text** against
`SWE-bench/SWE-smith` (the task id is a renumbered index, not the dataset's
own row id): `kayak__pypika.1c9646f0.lm_rewrite__mgynlcgd` is the only row
whose `problem_statement` equals the extracted instruction exactly.

The published SWE-Smith images (`jyangballin/swesmith.x86_64.*`) are **amd64
only** and 1.37 GB; emulating them on this arm64 host would make every agent
command slow, so the pilot clones the branch instead. The clone URL and the
resolved commit are recorded in `work/pilot/swesmith_instance.json`.

### Pilot result: 4 / 4 trials, 0 errors

With the faithful environments (ubuntu 24.04, `/testbed` for SWE-Smith, an
empty `/app` for the other three):

| task | trajectory steps | tool calls | what the agent did |
| --- | --- | --- | --- |
| `swesmith-00000_copy0001` | 20 | 18 | explored `/testbed`, found `left_needs_parens` in `pypika/terms.py`, ran `git show` on the bug commit `b8cab54`, and wrote its own `test_issue.py` |
| `issue-000000_copy0001` | 36 | 28 | found `/app` empty, installed astropy, read its `fitswcs.py`, and wrote the module into `/app/astropy/wcs/wcsapi/` |
| `superuser-000000_copy0000` | 6 | 5 | installed GNU parallel itself with `apt-get`, wrote `test_parallel`, and worked out the escaping |
| `tezos-0000` | 2 | 0 | answered the question directly (it is a Q&A task) |

Two of the four behave exactly like the published rollouts: the SWE-Smith
agent writes its own test script instead of using pytest (the published one
does the same), and the SuperUser agent installs GNU parallel itself (the
published one does the same).

The IssueTasks environment is the one that changed the most, and it is the
one the timeout had to follow: with `/app` empty the agent has to build the
module from scratch, and 900 seconds were not enough -- 73 steps and 3.7M
input tokens, still making progress (it had installed astropy, read its
source and was writing the fix).  The agent timeout is now 1800 seconds,
which is what the upstream datagen job uses.

Every trial directory contains:

```
agent/trajectory.json          ATIF v1.7: steps, tool_calls, observations, metrics
agent/opencode.txt             the raw opencode event stream (JSON lines)
agent/opencode/session.json    the raw session: every message and every part
agent/rsi-opencode-audit.json  the effective command line and both versions
```

The copies under `work/pilot/run/` are the small artifacts; the trial
directories themselves stay in the run root (`~/.rsi-harbor/`) because Harbor
has to bind-mount them into the container (see `rsi/paths.py`).  The archived
`session.json` files are gzipped -- the IssueTasks one alone is 21 MB
uncompressed, because the agent read astropy's source into the session.

Two things in `work/pilot/` are worth knowing about:

* `requests.jsonl.gz` is the **recording proxy's** log: the exact request
  bodies the model was sent, which is where the system prompt and the tool
  schemas come from.  It is the only place those bytes exist.
* `prompts.json` is the same thing distilled -- the system prompt, the tool
  schemas and the request count of each of the four tasks.

## The random probe: ten generated instructions, end to end

The published rows can only be checked against the published rows. The
**random probe** closes the other gap: ten instructions the pipeline has
never seen, generated rather than copied, run through the same machinery and
checked end to end.

[`gen_random_probe.py`](gen_random_probe.py) generates them. A seeded RNG
picks one of ten archetypes and fills its slots, so the same seed always
produces the same ten instructions (a failure can be replayed) while another
seed produces a different set. The archetypes are deliberately different from
each other: fix an off-by-one, add validation to a config loader, write a
pytest suite, remove duplicated logic, build a CLI, diagnose a crash,
implement a log parser, implement an HTTP service, speed up a hot loop, port
a module to Go.

Two properties are asserted before anything is written, because both would
silently corrupt a trajectory:

* the instruction contains **no backslash** — opencode escapes the inner
  double quotes of a user message on the wire but *not* the backslashes, so
  an instruction with a backslash does not survive the round trip;
* the instruction has **no trailing newline**, so the sha256 of the file is
  the sha256 of the text the converter reads back out of the trial.

### Result: 10 / 10 trials, 0 problems, 0 rows rejected

```
10/10 Mean: 0.000   0 exceptions   total runtime 28m41s (concurrency 5)
```

| | |
| --- | --- |
| rows | 10, none skipped, none with problems |
| assistant turns | 3, 3, 3, 4, 6, 6, 7, 7, 7, 14 |
| tool calls | 76 |
| rows with the system prompt and the tool schemas | 10 / 10 |
| distinct system prompts / tool sets | 1 / 1 (opencode's own) |
| on disk | 4.2 MB in total, ~170 KB per rollout |

The independent verifier
([`verify_dataset.py`](verify_dataset.py), a separate program that reads the
written `dataset.jsonl` back and checks it against the ground truth) reports
**10 rows, 10 rows ok, 0 problems**: the instruction of every row still
matches the generated instruction byte for byte, `conversations` keeps the
published schema, every tool call in `messages` has its matching tool result,
the two views agree, and the system prompt and the tool schemas are the ones
the model was actually offered.

### The export truncation is gone

In the slice, 8 of 15 session exports were **gzip-valid but JSON-truncated**:
`opencode export` was killed part way through writing. The export now writes
a plain file first, records its exit code and byte count next to it, and
gzips that file — so a partial export is visible instead of silent:

```
Exporting session: ses_e99556d2affeJ9WQ36CyTJZbr3 export rc=0 size=45482
```

All ten exports came back with `rc=0` and parse as valid JSON.

### The two conversion paths agree, down to the tool outputs

The converter can rebuild a rollout from the session export *or* from the
event stream, and it falls back to the latter when the former is unusable.
Both paths were run over all ten rollouts and compared:

* the **set** of tool calls per turn is identical — same tool, same call id,
  same arguments, same output;
* the reasoning and the visible text are identical;
* the only difference is the **order of parallel tool calls inside one
  assistant turn**, because the event stream records them in the order the
  events arrived.

So no data is lost on either path, and the two can be used interchangeably.

### What the probe cost

28m41s for ten rollouts at concurrency 5, i.e. **~7 minutes per rollout**
with five in flight. The bottleneck is the model endpoint, not the host: the
containers idle at 90–460 MB and the same request that took 38 s alone took
much longer with five rollouts sharing one endpoint. That is the number to
watch when the concurrency of the full run is tuned.

## 4. Convert the rollouts into an SFT dataset

```bash
.venv312/bin/python opencode_sft/build_sft_dataset.py \
    --run ~/.rsi-harbor/openthoughts-opencode \
    --requests ~/.rsi-harbor/logs/oc-requests.jsonl \
    --manifest work/registry/manifest.jsonl \
    --out opencode_sft/dataset/pilot

.venv312/bin/python common/tools/verify_dataset.py \
    --dataset opencode_sft/dataset/pilot/dataset.jsonl
```

The pilot produced **4 rows, 73 tool calls, 0 validation problems**.

### The format contract

The published dataset is stricter than it looks. Scanning all 31,600 rows gives
three invariants that the converted rows have to keep, otherwise the same
training code cannot load them:

| invariant | value |
| --- | --- |
| role vocabulary | `user` and `assistant` **only** — no `system`, no `tool` (401,602 + 398,653 messages, 0 exceptions) |
| message keys | exactly `content` and `role` (800,255 messages, 0 exceptions) |
| shape | `user` (prompt) → `assistant` → `user` (observation) → …; 773 rows have a second `user` message at index 1 (the resume episodes) |

So the converted row carries **two views of the same rollout**:

`conversations` — the published schema.
The system prompt and the task instruction become the two leading `user`
messages; each assistant turn is one `assistant` message whose tool calls are
flattened into inline `<tool_call>{…}</tool_call>` text; the tool results of one
turn become **one** `user` observation message holding one `<tool_result>`
block per tool. Folding them into one message is what keeps the strict
alternation: an opencode turn may fire three tools in parallel, and three
consecutive `user` messages would break the invariant.

`messages` — the serve-parity view.
`role: "system"` with the real prompt, `role: "user"` with the task, assistant
turns with **structured** `tool_calls: [{type, function: {name, arguments}}]`
(`arguments` is a JSON *string*, so the row stays Arrow-uniform), and one
`role: "tool"` message per result with its `tool_call_id`. This is the view a
tool-aware trainer consumes; the reasoning stays verbatim in `content` inside a
`<think>` block, as in the published dataset.

The two protocols are deliberately **not** mixed in one SFT set.

### Where the bytes come from

| what | source |
| --- | --- |
| system prompt + tool schemas | the request the recording proxy captured — the only place the exact bytes the model saw exist (`opencode debug agent` does not expose the prompt) |
| reasoning, text, tool calls, tool outputs | opencode's own `session.json` export |
| task instruction | `work/tasks/instructions/<task>.md`, copied byte-for-byte out of the published row |

### Two opencode wire details worth knowing

**The user message is wrapped in double quotes.** opencode sends
`"<instruction>"`, escaping the inner double quotes but **not** the
backslashes — it is not a JSON round trip. A task with three literal
backslashes arrives with three literal backslashes, while `json.dumps` would
have produced six escaped characters. The rule was verified against the
recorded requests of all four pilot tasks. The dataset therefore stores the
clean instruction in `instruction` and the wire form in `instruction_wire`,
with a sha256 of each, so the "the instruction did not change" claim can be
checked without guessing which one the model saw.

**opencode runs a connectivity probe before the real task.** It asks the model
to say "OK", with the *full* system prompt and the *full* tool list, so it is
indistinguishable from a real request by shape alone. It is excluded because
its user message never matches a trial, but a filter that only looks at "has
tools" would let it through.

### Two things the pilot exposed, and what they cost

**A transient provider error killed a trial.**  Harbor's opencode adapter
turns *every* opencode error event into `NonZeroAgentExitCodeError`, so a
dropped connection on the model endpoint and a real agent crash look the same
to the job's retry policy.  The pilot hit
`InvalidHTTPResponse fetching "http://host.docker.internal:8899/v1/chat/completions"`
on its twelfth step and the trial was simply lost.  At 31,600 rollouts against
one endpoint that is not a rare event.

`RsiOpenCode` now recognises the provider-shaped error events and re-raises
them as the matching `Api*Error` -- `ApiRateLimitError`, `ApiInternalServerError`,
`ApiOverloadedError`, `ApiConnectionClosedError` -- which the job's own
`retry.include_exceptions` already lists.  A configuration error
(`ProviderModelNotFoundError`) is deliberately **not** mapped: retrying it would
retry the same misconfiguration.

**A timed-out rollout was being thrown away.**  The published dataset keeps
its timed-out rollouts -- 10,981 of its 31,600 rows carry
`result: AgentTimeoutError` -- so dropping them would have shrunk the
re-annotated dataset by a third.  Two changes fix that:

* the session export moved into a `finally` block, so a failed trial still
  writes its `session.json`;
* when even that is missing (the container was torn down), the converter
  falls back to opencode's **event stream**, grouping the parts by their
  `messageID` and rebuilding the turns from it.  The instruction comes from
  the trial's own task directory, and the wire form is rebuilt from it, so the
  sha256 of the instruction still matches the published dataset.

Both paths go through the same validation, and the independent verifier
checks the result against the published parquet.

### The verification is a separate program

`verify_dataset.py` reads the written `dataset.jsonl` back and checks it
against the **published** parquet — the generator checking its own output
would prove nothing. Per row it verifies that the instruction still matches a
published task description byte for byte (sha256), that the schema invariants
hold, that every tool call has a matching result and that the two views agree,
that the system prompt and the tool schemas equal the recorded ones, and that
no invisible character slipped in.

It was fault-injection tested: six deliberate corruptions (instruction off by
one character, broken `tool_call_id`, tampered system prompt, a `tool` role in
the `conversations` view, desynchronised tool arguments, an injected
zero-width space) are all caught, and the unmutated file passes.

## Environment fidelity: the dataset is the ground truth

The task ids are renumbered indices (`swesmith-00000_copy0001`), so they say
nothing about the environment.  The **published rollouts** do, and the pilot
environments are reconstructed from them rather than from the task names.

### What the published rollouts prove

| fact | evidence | rows |
| --- | --- | --- |
| the base is **ubuntu:24.04** | `cat /etc/os-release` → `PRETTY_NAME="Ubuntu 24.04.3 LTS"` | 1,474 rollouts print it; 1,298 answer 24.04.4, 151 answer 24.04.3 |
| the apt sources are ubuntu's own | `apt-get update` → `Hit:1 http://archive.ubuntu.com/ubuntu noble InRelease` | the SuperUser rollout |
| the system python is **3.12** | `python3 --version` → `Python 3.12.3` | 1,720 rollouts |
| the SWE-Smith python is **3.10.16** | `python --version` → `Python 3.10.16` | 6,913 rollouts (2,749 answer 3.10.18) |
| the working directory is `/app` | the initial `Current Terminal Screen:` | 23,225 rows |
| the working directory is `/testbed` | the same, for SWE-Smith | 8,375 rows |
| `/testbed` **is** the repository root | the first `ls -la` shows `pypika/`, `setup.py` and `PyPika.egg-info` side by side | the SWE-Smith rollouts |
| `/app` starts **empty** for the other three families | the IssueTasks rollout found `/app` empty and wrote `astropy/wcs/wcsapi/fitswcs.py` from scratch | 13,383 of 23,225 rows; the rest are task-seeded |
| the container has **no systemd** | `timedatectl: command not found`, `systemctl: command not found` | the Tezos rollouts |
| GNU parallel is **not** pre-installed | the SuperUser rollout ran `apt-get install -y parallel` itself | the SuperUser rollouts |

The upstream generator agrees: `data/swesmith/task_templates.py` in
`open-thoughts/OpenThoughts-Agent` builds the non-repository tasks as

```dockerfile
FROM ubuntu:24.04
WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip && ...
```

and the repository tasks on top of the official SWE-smith image, which is the
same ubuntu plus a conda environment named `testbed` with python 3.10 and the
package installed editable.

### What the pilot got wrong before

| | before | now |
| --- | --- | --- |
| base | `rsi/tc-python-3-13-slim-bookworm` — Debian bookworm, python 3.13 | `rsi/oc-pilot:2` — ubuntu 24.04, python 3.12 |
| SWE-Smith repository | `/testbed/repo` | `/testbed` (the repository root) |
| SWE-Smith python | 3.13 (system) | 3.10.16 in a `testbed` environment on PATH |
| SWE-Smith install | none | `pip install -e .` → `PyPika.egg-info`, `import pypika` works |
| IssueTasks | a full astropy checkout at `/app/app` | **empty** `/app`, as in the published rollout |
| SuperUser | GNU parallel pre-installed | not installed — the agent installs it, as in the published rollout |

### Checking it instead of asserting it

`check_env_fidelity.py` reads the environment facts out of the published
parquet, builds each pilot task image, runs the same observations inside the
container, and reports the differences:

```
$ python common/tools/check_env_fidelity.py
3/4 environments match the published rollout
```

SWE-Smith matches exactly: the working directory, all 18 files of `/testbed`
(including `PyPika.egg-info`), `git log -1` → `d7b141d Remove F2P Tests`
(the mutation commit), `python --version` → `Python 3.10.16`, and
`import pypika` → `/testbed/pypika/__init__.py`.

The one difference that is left is the **Ubuntu point release**: the published
dataset spans 24.04.3 and 24.04.4, and the floating `ubuntu:24.04` tag
resolves to 24.04.5.  Pinning a point release is not an option either --
Docker Hub retires those tags, and the pull of `ubuntu:24.04.4` fails from
this network with a TLS handshake timeout.

### The SWE-Smith image itself is not reproducible here

The upstream image `jyangballin/swesmith.x86_64.*` is **amd64 only** and
1.37 GB; emulating it on this arm64 host makes every agent command slow.  The
pilot therefore reconstructs its *contents* -- the same repository at the same
commit, the same python 3.10.16, the same editable install, the same
`PyPika.egg-info` -- on the arm64-native base.  The reconstruction is checked
against the published rollout, but it is a reconstruction.

Two details that only show up when the agent actually uses the environment:

* the venv needs `--seed`, or `pip` resolves to the **system** pip and
  refuses to install anything (PEP 668, "externally managed environment").
  The upstream conda environment has no such rule, so an agent that runs
  `pip install pytest` there succeeds;
* the repository's `requirements-dev.txt` pins Sphinx 1.6.5 and tox 3.14.3,
  which do not install on python 3.10.  It is deliberately not installed: the
  published rollout never used any of those tools -- it wrote its own
  `test_issue.py` and ran it with `python`.

## Traps that cost time

**1. `name: opencode` silently discards your `import_path`.** Harbor's factory
creates its *built-in* OpenCode agent when the agent name is one of the
built-in names, so the job fails before a single trial starts:

```
Invalid kwargs for agent 'opencode':
  Unknown option 'export_session'.
```

The fix is a custom agent name — the same trick the mini-swe-agent job uses
with `rsi-mini-swe-agent`. The job file uses `name: rsi-opencode`.

**2. The built-in `openai` provider cannot take a custom model id.** Covered
above: `ProviderModelNotFoundError`, surfacing as a generic server error.

**3. `Cli`-annotated options become command-line flags.** Harbor renders every
`Cli` field of the agent's options model into the agent's command line. An
option like `export_session` must therefore *not* be annotated, or opencode
receives `--export-session` and rejects the whole run.


**5. A docker build context is not a build context.**  `docker build -f
common/images/oc-pilot-ubuntu24.Dockerfile -t rsi/oc-pilot:2 .` uploads
the **whole repository** as the build context -- 3.4 GB, including the
virtualenv and the 30,737 extracted instruction files.  The build sat there
for fifteen minutes and produced nothing.  The Dockerfile copies nothing, so
the context only has to exist: building with `common/images` as the
context (and a `.dockerignore` with `*` in it) took **70 seconds**.  The root
`.dockerignore` now excludes `work/tasks`, `work/pilot` and the two
`dataset/` directories as well.

**6. A background process started with `( ... & )` inside a tool call dies
with that call.**  The recording proxy was started that way, survived one
pilot, and was gone by the next: the container then failed with
`Cannot connect to API: Unable to connect`, and all four trials died in
under two minutes with an empty `opencode.txt`.  A process that has to outlive
the call that starts it belongs in the runtime's own background task, not in a
detached subshell.

## Two things worth knowing about the environment

**Image and sandbox are different granularities.** The image is a read-only
template; the sandbox is the running container and it *is* mutated by the
agent (files edited, packages installed, tests run). So:

- one sandbox per trial — never share one between tasks;
- one image per *family*, not per task — the 31,600 tasks need 1–3 images;
- 4 concurrent trials = 4 containers sharing 1 image, not 4 images.

Harbor tags images by content (`hb__<environment_id>`), so an unchanged task
reuses the cached image instead of rebuilding it.

**The dataset's `trace_source` is not the task family.** It holds
`main` / `summarization-1-answers` / `null` — which sub-agent produced the
trace. The family is recovered from the task id prefix
(`swesmith-` / `superuser-` / `tezos-` / `issue-`), and `extract_tasks.py`
writes it as `family`.

## Findings from the extraction

**Extraction result: 31,600 / 31,600 rows, 0 failures.**

| | |
| --- | --- |
| rows | 31,600 |
| unique task ids | 29,765 |
| distinct instructions | 30,737 (972 tasks carry a second variant) |
| mean messages per rollout | 25.3 |
| rows by family | swesmith 8,375 · issue 8,374 · superuser 7,654 · tezos 7,197 |

Three things the data forced:

1. **A task id is not a unique key for an instruction.** 766 tasks carry two
   variants of the same task. Comparing the two texts of `tezos-04221`:

   ```
   7748 chars  vs  7596 chars,  similarity 0.9901
   - variant A ends with an extra footer:
       ---
       CONSTRAINT: Do not install any new packages. Use only what is already available.
       CONSTRAINT: Write at least 3 tests to verify your solution works.
   ```

   Across all variant pairs the relation is `rewritten` 783, `suffix_removed`
   96, `suffix_added` 93 — so most are genuine rewrites of the same task, not
   a footer. This is the upstream "synthetically augmented" step. **The
   instructions are therefore keyed by `(task, sha256)`**, and every row of
   `index.jsonl` carries the path of the exact file its instruction was
   written to, so a re-annotation can reproduce the instruction of any
   individual rollout.

2. **The terminus-2 prompt exists in two versions** in the published data.
   They differ in a single line of the harness prompt:

   ```
   - You must end every command with a newline (\n) or it will not execute.
   + Most bash commands should end with a newline (\n) to cause them to execute
   ```

   1,806 rows (5.7%) use the older wording. The **task instructions are
   unaffected** — the drift is in the harness prompt only — but the prompt
   template hash is not constant across the dataset.

3. **A third of the rollouts are timeouts.** `result` is `null` for a
   successful rollout and carries the exception name otherwise:

   | result | rows |
   | --- | --- |
   | `null` (success) | 20,205 |
   | `AgentTimeoutError` | 10,981 |
   | `ContextLengthExceededError` | 286 |
   | `DaytonaNotFoundError` / `DaytonaError` | 102 |
   | `CancelledError` / `RuntimeError` / other | 26 |

   The upstream filter (at least 5 model turns) has to be applied before the
   data is used for training.

## One trap in the opencode wiring

opencode's **built-in** `openai` provider ships its own model registry and
rejects a custom model id, even when the model is declared under
`provider.openai.models`:

```
ProviderModelNotFoundError: Model not found: openai/glm5.3-flash.
```

The error surfaces as a bare `UnknownError: Unexpected server error. Check
server logs for details.` — it does *not* say "model not found", so it reads
like a server problem. The fix is a **custom provider** built on
`@ai-sdk/openai-compatible`, which is what the driver's job template does:

```yaml
model_name: rsi_glm/glm5.3-flash
opencode_config:
  provider:
    rsi_glm:
      npm: "@ai-sdk/openai-compatible"
      options:
        baseURL: "http://YOUR-ENDPOINT:PORT/v1"
        apiKey: "EMPTY"
```

The endpoint itself is fine: it answers `/v1/models`, `/v1/chat/completions`
and `/v1/responses`, and the model produces native tool calls (opencode's
`write` / `read` tools fire correctly against it).

## What was verified before anything was run at scale

| check | result |
| --- | --- |
| the job file the driver writes (`<run-root>/batch-NNNN.yaml`) against Harbor's `JobConfig` | validates (4 concurrent trials, docker backend, retry policy) |
| the agent's `kwargs` against `RsiOpenCodeOptions` | validates; unknown keys are rejected |
| the opencode command line that `RsiOpenCode` renders | only `--agent` is ever appended — the boolean knobs are **not** `Cli`-annotated, because Harbor renders every `Cli` field into the command line and opencode would reject `--export-session` as an unknown flag |
| the merged `opencode.json` that Harbor writes into the container | probed end-to-end: `reasoning` → `tool_use(write)` → `step_finish` → `tool_use(read)`, file written |
| `opencode export <session-id>` with Harbor's `XDG_DATA_HOME` | produces `{"info", "messages"}` with every part (text, reasoning, tool inputs/outputs, step boundaries) — the role lives at `message.info.role` |

The probe also showed that opencode has to be started with `</dev/null`, which
Harbor's command already does — without it, a run that inherits an open stdin
hangs instead of failing.


