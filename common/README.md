# common — what the two harness halves share

The re-annotation runs two harnesses over the same tasks, in the same
environment, against the same model:

| half | directory | harness |
|---|---|---|
| Mini | [`../mini_sft/`](../mini_sft/) | mini-swe-agent 2.4.6 |
| Opencode | [`../opencode_sft/`](../opencode_sft/) | opencode |

Everything that is not specific to one of them is here.

---

## The orchestration

### `run_all_stages.sh`

The orchestrator.  It runs the stages in order and checks each one's output
before the next one starts:

```
clawgym-split → clawgym-run → clawgym-convert → clawgym-merge
              → openthoughts-split → openthoughts-run → openthoughts-convert
```

Each stage is a shell function, and `STAGES` selects which ones to run:

```bash
STAGES="clawgym-split clawgym-run" bash common/run_all_stages.sh
```

Three things in it are worth knowing, because each of them was a bug first:

* **the streams are functions whose last command is `exec`.**  A backgrounded
  shell *function* forks a subshell, and `$!` is that subshell's pid, not the
  driver's.  Without `exec` the pid file points at a wrapper, "kill the
  driver" kills the wrapper, and the same stream runs three times over.
* **the supervisor polls, it does not wait in turn.**  A stream that dies is
  started again *while the others keep running*.  Waiting in turn would leave
  a crashed stream idle for as long as the slowest one needs, and the whole
  run is days long.
* **the liveness check reads the process state, not just `kill -0`.**  An
  exited-but-unreaped process is a *zombie*, and `kill -0` succeeds on a
  zombie.  Without the state check the supervisor would believe a crashed
  driver is still running, and the run would stop moving with nothing in the
  log to say why.

### `load_env.sh`

`.env` loading, with the right precedence: **the environment wins over the
file**.  `.env` is a default, not an override.

```bash
RUN_ROOT=/tmp/scratch make annotate      # runs in /tmp/scratch
make annotate                             # runs in the run root from .env
```

The rule (and the list of variables it covers) lives here, and every script
that reads `.env` goes through it.  `source .env` on its own is an
*assignment*: it overwrites the caller's variables with no error at all.

---

## The wire

### `record_proxy.py`

A recording proxy in front of the model endpoint.  It writes every request
body to disk and injects the API key on the way out.

Two things it has to get right:

* **it must *override* the `Authorization` header, not only fill in a
  missing one.**  opencode sends its own placeholder, and the endpoint
  rejects it.
* **a key never reaches a job file.**  The job file goes into the container
  and into the trial log, so the real key is injected by the proxy on the way
  out and the job file only ever holds the placeholder `EMPTY`.

In *blob* mode it stores the system prompt and the tool schemas once, by
content digest, instead of once per rollout -- 80 KB and 10 KB against a
~120 byte reference.  opencode's session export does not contain either of
them, so the request is the only ground truth for them.

### `schema.py`

The shared schema.  Every row carries the three keys a trainer filters on:

| key | value |
|---|---|
| `dataset` | `ClawGym` / `OpenThoughts` |
| `model` | `GLM-5.3-Flash` |
| `harness` | `Mini` / `Opencode` |

and the two an auditor follows, `model_id` and `model_provider`.

---

## The data

### `split_tasks.py`

The split.  Each dataset is cut in half by the sha256 of the task name, and
the two manifests go to the two harnesses.  It is deterministic and
independent of the row order, so it is the same on every machine and in every
re-run.

```bash
.venv/bin/python common/split_tasks.py \
    --registry work/registry_clawgym \
    --report common/registry/split_report.json
```

`--minimax-share` cuts a share out for a second endpoint; it is `0` now (the
second model was withdrawn), but the mechanism is still there.

### `registry/`

The registry builders, and the reports they write:

| file | what it builds |
|---|---|
| `build_clawgym_registry.py` | `work/registry_clawgym/` -- 15,402 ClawGym tasks |
| `build_registry.py` | `work/registry/` -- the OpenThoughts families |
| `extract_tasks.py` | cuts the task instructions out of the published rollouts |
| `extract_user_instructions.py` | the same, over all 31,600 dataset rows |
| `probe_verifiers.py` | runs every ClawGym verifier against an empty workspace |

The OpenThoughts dataset ships *rollouts*, not tasks, so the task
instructions have to be cut out of the terminus-2 transcripts byte-identically
before the environments can be built.

### `validate_sft.py`

The dataset validator.  It is an **independent** program: it re-reads the
dataset from disk and checks it against the registry, not against the
generator's in-memory objects.  It also has a fault-injection mode -- it is
fed deliberately broken datasets and has to catch every one of them
(`--inject`).

### `merge_sft.py`

Merges the two halves into one dataset.  It checks that the two sides do not
overlap and do not leave a gap, that every row has a `harness`, and that the
fields are complete -- and it writes atomically.

---

## The infrastructure

| file | role |
|---|---|
| `shared_network.py` | creates the one Docker network every trial joins |
| `add_shared_network_compose.py` | adds it to a task's compose file |
| `docker_cleanup.py` | reaps the containers and the harbor jobs a dead driver left |
| `minimax.py` | the second endpoint; **off** by default |
| `images/` | the base images (opencode is baked into them) |
| `tools/` | the one-off tools that were written along the way |

`minimax.py` is kept because the mechanism is sound and switching the second
model back on is a matter of command-line flags -- but the current
orchestrator runs **only** the two GLM streams.

---

## The notes

* [`CLAWGYM.md`](./CLAWGYM.md) -- what the ClawGym task family is: the
  environment contract, the verifier, the reward, and the filters.
* [`EXECUTION_PLAN.md`](./EXECUTION_PLAN.md) -- the traps, with the evidence.

The two READMEs of the harness halves are the best introduction to what each
harness actually does:

* [`../mini_sft/README.md`](../mini_sft/README.md) (Chinese)
* [`../opencode_sft/README.md`](../opencode_sft/README.md)
