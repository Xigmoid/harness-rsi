# ClawGym re-annotation: tasks → harness-agent trajectories → SFT

This is the ClawGym half of the re-annotation pipeline.  The OpenThoughts
half (`opencode_sft/mini_sft/`) re-runs *published rollouts* through
mini-swe-agent.  This half starts from the other kind of dataset:
`RUC-AIBOX/ClawGym-Task` ships **tasks**, not trajectories, so the rollouts
are produced here, and the task's own verifier is what says whether a
rollout is worth training on.

Everything runs against **one** Docker image.  The user set that constraint
explicitly, and the dataset supports it: the per-task part of a ClawGym task
is its files, not its environment.

---

## 1. What the dataset actually is

`RUC-AIBOX/ClawGym-Task` → `syn_task.jsonl`, 405 MB, **15,402 tasks**, no
duplicate prompts.  Each line:

| field | meaning |
|---|---|
| `prompt` | the task instruction, given to the agent verbatim |
| `hook_code` | the verifier command -- **the same line for all 15,402 tasks**: `python3 reward/test.py /root/.openclaw/workspace` |
| `hook_lang` | `bash` for all of them |
| `input_files` | `{file_path, file_format, content}` -- the task's own files |
| `rules` | extra rubric rules (4,303 tasks) |
| `meta_data` | two different schemas, see below |

Two families hide in one file:

* **rows 0–6,093 -- the "arc" tasks.**  `meta_data` carries `task_id`
  (`arc_<skill>_<hash>`), `validation_mode` (`code` / `code_and_rubric`),
  `reward_aggregation`, `bundle_name`, `input_mount_dir`, and a nested
  `metadata` with `primary_skill_id`, `supporting_skill_ids`, `task_source`,
  `difficulty` (4, or 5 for exactly one task) and `difficulty_reason`.
* **rows 6,094–15,401 -- the "persona" tasks.**  `meta_data` is
  `{category, macro_category, action}` -- a taxonomy, not an identifier.

Both families use the same file layout and the same hook, so one registry
builder and one image serve both.

## 2. The environment contract

The single `hook_code` pins the layout:

```
python3 reward/test.py /root/.openclaw/workspace
```

* the verifier lives at `reward/test.py` **relative to the working
  directory** → the working directory is `/root/.openclaw/workspace`;
* the verifier's argument is the workspace root, and it reads the inputs
  from `<workspace>/input/` and checks the deliverables under
  `<workspace>/output/`;
* every path in `input_files` is relative to the workspace root -- not just
  the ones under `input/`: 454 tasks ship `config/`, 196 ship `scripts/`,
  171 ship `data/`, and so on.

**The verifier is visible to the agent.**  That is not a choice: 15 prompts
say "Do not write anything under reward/" and 12 say "Use only relative
paths under input/, output/, and reward/".  Those instructions only make
sense if `reward/` is there.  Hiding it would change the task.

The cost of that is reward hacking, and it is handled in the *filter*, not
in the environment (§6).

## 3. One image for 15,402 tasks

`opencode_sft/images/oc-base-clawgym.Dockerfile` → `rsi/oc-base-clawgym:1`
(649 MB).

* **ubuntu 24.04, Python 3.12.3.**  Not a preference: six verifiers contain
  an f-string with a backslash inside the expression
  (`f"{p.replace('\\', '/')}"`), which is a syntax error before 3.12 and
  legal from 3.12 on (PEP 701).  Those six are among the 15,402.
* **PyYAML.**  101 verifiers `import yaml`; without it they cannot be scored.
* **build-essential.**  Harbor's mini adapter asks for
  `("curl", "bash", "build_tools", "git")`, and `build_tools` means `gcc` and
  `make`.  When they are missing, Harbor runs `apt-get install -y
  build-essential` **inside every trial** -- and `deb.debian.org` answers at
  0.18 MB/s from this host, so the trials never get going.  Baking it in
  costs one download instead of one per trial.
* **The workspace skeleton**: `/root/.openclaw/workspace/{input,output,reward}`,
  empty.  `WORKDIR` is the workspace.

The apt sources are restored to the way ubuntu ships them after the build
downloads from a faster mirror -- the agent's own `apt-get update` has to
reach the same archive the original container did.

### The injection mechanism

Harbor uploads a task's `environment/` directory into `[environment].workdir`
when the task names a pre-built `docker_image` and has no
`environment/Dockerfile` or `environment/docker-compose.yaml`
(`harbor/environments/base.py:_upload_environment_dir_after_start`,
`harbor/environments/definition.py:should_upload_environment_dir`).

The ClawGym tasks need that file **and** a `docker-compose.yaml` (for the
shared network, §8), and the presence of the compose file switches the
upload off.  The two mechanisms cannot both be used, so the payload is
mounted instead:

* the task's files live in `environment/payload/`, which is mounted
  read-only at `/rsi/payload`;
* the container copies them into the workspace when it starts:
  `cp -a /rsi/payload/. /root/.openclaw/workspace/ ; exec sleep infinity`.

One mount covers any number of top-level entries, and it has to be one: the
ClawGym payloads have up to a couple of hundred distinct top-level
directories (`input/`, `config/`, `scripts/`, `data/`, and dozens of
one-off names), plus root-level files, and 244 tasks ship **only** the
verifier.  Mounting them one by one would mean a compose file per task with
a mount per entry.

The copy also keeps the workspace ordinary -- the agent gets writable
directories, exactly as it would from an upload -- and the image's own
`output/` stays writable in the container's layer.

## 4. The registry

`opencode_sft/clawgym/build_registry.py` → `opencode_sft/registry_clawgym/`
(15,402 task directories, 61,111 files, 649 MB).

```
clawgym/clawgym-000000-arc-security-audit-358537a6060c/
    task.toml            # docker_image = rsi/oc-base-clawgym:1
                         # workdir     = /root/.openclaw/workspace
    instruction.md       # the prompt, verbatim
    environment/
        docker-compose.yaml   # the shared network + the payload mount
        payload/
            input/skills_audit_input.json
            reward/test.py
    tests/test.sh        # runs the verifier, writes the reward
```

`tests/test.sh` runs the task's own hook from the workspace -- the same
program, the same argument, the same relative path the hook was written for
-- and turns its JSON into the single float Harbor reads from
`/logs/verifier/reward.txt`.

## 5. The reward: two shapes, measured not guessed

Harbor's verifier reads one float, but the ClawGym verifiers do not agree on
a shape.  `opencode_sft/clawgym/probe_verifiers.py` **runs** all 15,402
verifiers against an empty workspace and records what they printed:

| shape | count | example |
|---|---|---|
| `reward` -- an explicit aggregate | 6,091 | `{"reward": 0.0, "has_report_file": false, ...}` |
| `flat` -- a mapping of check → number | 9,302 | `{"synopsis_word_count_200_250": 0.0, ...}` |

A static scan gets this wrong: the flat verifiers never contain the literal
string `"scores"`, so a `grep` classifies 9,284 of them as "no recognised
shape".  Running them is the only way to know.

The probe also found **nine apparent failures that are not failures**:

* six are the Python-3.12 f-strings above (the probe ran on 3.11);
* three read `input/` without tolerating its absence (the probe's workspace
  had no `input/`).

With Python 3.12 and `input/` present, **all 15,402 run clean**.

`tests/test.sh` prefers the explicit `reward` and falls back to the mean of
the numeric checks, and records which rule it used in
`verifier/reward_rule.txt`.

## 6. The prompt, the config and the filters

### The instance template

mini's packaged instance template opens with *"Please solve this issue:
{{task}}"* and a workflow about finding a codebase, reproducing an issue and
editing source code.  The ClawGym tasks are not issues, there is no
codebase, and the deliverables are files the task names.  A template that
contradicts the task teaches the model to work from a framing the task never
used.

`opencode_sft/build_mini_config.py --profile clawgym` replaces exactly that
head.  **Everything from "## Command Execution Rules" onwards is byte
identical to mini's own template** -- that section *is* the protocol (the
bash tool, the subshell rule, the submission command), and the protocol is
the harness.  `model` and `environment` are identical between the two
profiles; only `agent.instance_template` differs.

### The filters

`opencode_sft/clawgym/build_sft_dataset.py`:

* `--min-reward` (default 1.0) -- most verifiers report the *fraction* of
  checks passed, so this is a knob, not a boolean;
* `--drop-reward-touched` (default on) -- drops the rollouts whose
  assistant turns mention `reward/`.  The verifier is visible (§2), so an
  agent that reads `reward/test.py` learns the expected values and can
  satisfy the checks without doing the work; training on that teaches the
  model to look for the grader.  Only the *assistant* turns are scanned --
  the prompt itself mentions `reward/` in a few hundred tasks.

## 7. Running it

The 15,402 tasks are split in two, one half per harness, by the sha256 of the
task name (`opencode_sft/split_tasks.py`): **Mini 7,749 / Opencode 7,653**.
The split is deterministic and independent of the row order, so the two
halves hold the same kinds of task and the comparison between them means
something.

```bash
# the registry (15,402 tasks)
.venv/bin/python opencode_sft/clawgym/build_registry.py \
    --jsonl .cache/clawgym/syn_task.jsonl \
    --out opencode_sft/registry_clawgym --clean

# the split
.venv/bin/python opencode_sft/split_tasks.py \
    --registry opencode_sft/registry_clawgym \
    --report opencode_sft/clawgym/split_report.json

# the config
.venv/bin/python opencode_sft/build_mini_config.py --protocol toolcall \
    --profile clawgym --out opencode_sft/clawgym/mini-config-clawgym.yaml

# ---- the Mini half --------------------------------------------------------
.venv312/bin/python opencode_sft/run_mini_rollouts.py \
    --registry opencode_sft/registry_clawgym \
    --manifest opencode_sft/registry_clawgym/manifest-mini.jsonl \
    --run-root /private/tmp/rsi-harbor/clawgym \
    --families clawgym \
    --min-config opencode_sft/clawgym/mini-config-clawgym.yaml \
    --batch-size 100 --concurrency 24 \
    --enable-verifier --shuffle-seed 20261024

# the SFT dataset
.venv/bin/python opencode_sft/clawgym/build_sft_dataset.py \
    --run-root /private/tmp/rsi-harbor/clawgym \
    --registry opencode_sft/registry_clawgym \
    --out opencode_sft/clawgym/dataset/clawgym-sft.jsonl

# ---- the Opencode half ----------------------------------------------------
# The recording proxy is what gives the converter the system prompt and the
# tool schemas: neither is in opencode's session export.  It runs in *blob*
# mode, which stores those two once by content digest instead of once per
# rollout -- 80 KB and 10 KB against a ~120 byte reference.
.venv/bin/python opencode_sft/record_proxy.py \
    --port 8910 --host 0.0.0.0 \
    --upstream http://YOUR-ENDPOINT:PORT/v1 \
    --out /private/tmp/rsi-harbor/logs/oc-requests.jsonl \
    --compact --dedupe --blob-dir /private/tmp/rsi-harbor/logs/oc-blobs

.venv312/bin/python opencode_sft/run_rollouts.py \
    --registry opencode_sft/registry_clawgym \
    --manifest opencode_sft/registry_clawgym/manifest-opencode.jsonl \
    --run-root /private/tmp/rsi-harbor/clawgym-opencode \
    --families clawgym \
    --batch-size 50 --concurrency 6 --enable-verifier \
    --proxy-check-url http://127.0.0.1:8910/v1/models \
    --api-base http://host.docker.internal:8910/v1

.venv/bin/python opencode_sft/clawgym/build_opencode_sft_dataset.py \
    --run-root /private/tmp/rsi-harbor/clawgym-opencode \
    --registry opencode_sft/registry_clawgym \
    --requests /private/tmp/rsi-harbor/logs/oc-requests.jsonl \
    --blob-dir /private/tmp/rsi-harbor/logs/oc-blobs \
    --out opencode_sft/clawgym/dataset/clawgym-opencode-sft.jsonl

# ---- the independent check, and its own fault injection --------------------
.venv/bin/python opencode_sft/clawgym/validate_sft.py \
    --dataset opencode_sft/clawgym/dataset/clawgym-sft.jsonl \
    --registry opencode_sft/registry_clawgym \
    --run-root /private/tmp/rsi-harbor/clawgym
.venv/bin/python opencode_sft/clawgym/validate_sft.py \
    --dataset opencode_sft/clawgym/dataset/clawgym-sft.jsonl \
    --registry opencode_sft/registry_clawgym --inject

# ---- merge the two halves -------------------------------------------------
.venv/bin/python opencode_sft/clawgym/merge_sft.py \
    --dataset opencode_sft/clawgym/dataset/clawgym-sft.jsonl \
    --dataset opencode_sft/clawgym/dataset/clawgym-opencode-sft.jsonl \
    --out opencode_sft/clawgym/dataset/clawgym-merged-sft.jsonl
```

`--shuffle-seed` matters: the manifest is ordered by dataset row, and the
first 6,094 rows are one task family, so a run in manifest order spends its
first batches on one kind of task.

## 7a. The checker, and its fault injection

`opencode_sft/clawgym/validate_sft.py` re-opens the written dataset and
checks it against the **source of truth** -- the registry's
`instruction.md`, the ledger's reward, and the OpenAI message schema -- not
against the converter's own objects.

It is checked against itself too.  `--inject` runs it over deliberately
broken copies and requires every mutation to be caught:

| mutation | caught as |
|---|---|
| swap two messages | `first_turn_not_user`, `assistant_after_system`, `tool_count_mismatch` |
| drop a tool message | `dangling_tool_call_id`, `tool_count_mismatch` |
| duplicate a tool message | `tool_call_answered_twice`, `tool_count_mismatch` |
| rename a role | `no_system_first` |
| add a stray key | `stray_key` |
| break the tool_call json | `arguments_not_json` |
| touch the instruction | `instruction_mismatch` |
| change the reward | `bad_reward`, `reward_mismatch` |
| tamper the instruction in the prompt | `instruction_not_in_prompt` |
| tamper the wire instruction | `instruction_wire_mismatch` |
| rename the harness | `wire_form_on_a_mini_row`, `instruction_not_in_prompt` |

The last two only make sense for one of the harnesses, so a mutation that
does not apply raises `NotApplicable` and is reported as **n/a** rather than
as missed -- a mini row has no wire instruction, and pretending otherwise
would make the injection report meaningless.

Two findings from writing it are worth keeping, because both look like
defects and are not:

* **one dangling `tool_call_id` is expected.**  The final assistant message
  is the model's last output; it was never sent back to the model, and mini
  never ran its command (``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` is
  intercepted), so its observation is not in the sample.  Any *other*
  dangling id is a real defect.
* **several tool messages in a row are normal.**  One assistant message may
  carry several tool calls, and each is answered by its own tool message.

## 7b. The two harnesses are different shapes, and the checker knows it

| | Mini | Opencode |
|---|---|---|
| where the messages come from | the harness's own trajectory | rebuilt from the session's part tree |
| the system prompt | 62 characters, in the trajectory | 80 KB, **only in the captured request** |
| the instruction on the wire | verbatim | wrapped in double quotes, inner quotes escaped |
| the reasoning | `reasoning_content` | `reasoning_content` |
| the tool arguments | a JSON string | a JSON string, serialised **without spaces** (`JSON.stringify`) |
| the last message | always assistant | assistant, or a tool message if the rollout was cut |

The last row is the one that catches people: an opencode rollout that is cut
off right after a tool result ends with a `tool` message, and that
observation *was* sent to the model, so it belongs in the sample.

The escaped instruction is the other one.  opencode puts the task on the wire
as `"the instruction"` with the inner quotes escaped and the backslashes
left alone -- it is not a JSON round trip.  A checker that looks for the bare
instruction in the prompt flags every task whose prompt contains a quote.

## 7c. The wire check: is the sample the model's own conversation?

A converter that produces a plausible message list proves nothing.  The
question is whether the sample **is** the conversation the model received.

`opencode_sft/verify_opencode_wire.py` answers it by putting the recording
proxy in front of the endpoint for a few trials -- in full mode, every
request body -- and comparing the converted rows with the requests that were
really sent.  The rule is the one the pilot established:

> the converted `messages` must equal the **last** recorded request's
> messages, byte for byte, and must be exactly one message longer -- the
> extra one being the model's final output, which was never sent back.

Three things had to be got right for that to hold, and **all three were
found by running the comparison**, not by reading the code:

* `tool_calls[].function.arguments` is serialised the way JavaScript's
  `JSON.stringify` does it -- no space after `:` or `,`.  Python's
  `json.dumps` default puts one there.  The two parse to the same object and
  are a different token sequence.
* the reasoning is its own field (`reasoning_content`), not folded into the
  content.  Folding makes the sample *read* the same and *tokenise*
  differently.
* the tool result messages carry `role`, `content` and `tool_call_id` -- and
  nothing else.

The result on the pilot: **4 of 4 rollouts byte-identical**, zero
differences.  One of them was a rollout that read `reward/test.py`; it is
byte-identical too, which is the point -- the filter drops it because it is
harmful training data, not because it is broken.

And because two implementations of the same conversion is one too many, the
shared converter is pinned to the audited pilot implementation by
`opencode_sft/regression_opencode_converter.py`: 5 of 5 rollouts
byte-identical.

## 8. Pitfalls

### The toolchain bundle is bind-mounted read-write

`.cache/rsi-local` is mounted at `/root/.local` in every trial, and uv's
default tool and python directories live under `/root/.local/share/uv`.
**A fallback install therefore writes into the shared mount**: it removes the
tool environment and reinstalls it, which breaks the bundle for every other
trial using the same mount -- and the fast path then fails for the *next*
batch too, so the damage compounds.

This happened: a batch whose fast path did not trigger ran four fallbacks
concurrently, `share/uv/tools/mini-swe-agent/bin` was removed, and the
bundle had to be re-exported from the toolchain image.

Two fixes, both in place:

* `rsi/harbor_agent.py` redirects uv's directories (`UV_TOOL_DIR`,
  `UV_TOOL_BIN_DIR`, `UV_PYTHON_INSTALL_DIR`, `UV_CACHE_DIR`) into
  `/tmp/rsi-uv`, so a fallback install cannot reach the mount;
* `run_mini_rollouts.py` preflights the bundle before a batch and refuses to
  start when the shim or the tool environment is missing.

The bundle's shims are symlinks to absolute paths **inside** the container,
so on the host they are dangling by design -- the check is "does the link
exist" (`os.path.lexists`), not "does it resolve".

### Every trial needs the shared Docker network, or the address pool runs out

Harbor gives every trial its own compose project and therefore its own
bridge network, and those come out of Docker's default address pool --
**sixteen** of them.  A trial that fails leaks its network, and once the pool
is empty every later trial dies with::

    failed to create network ...: all predefined address pools have been
    fully subnetted

which looks exactly like an agent failure and is not one.

The ClawGym registry hit this for real: its `environment/` directory held
the task's files and nothing else, so there was no compose file, so no
trial joined the shared network, and the run wedged with 14 leaked
networks and every new trial failing to start.  The fix is the compose
overlay in §3: every task declares the shared network, which is
``external``, so ``docker compose down`` (what Harbor runs when it cleans a
trial up) leaves it alone.

**The two mechanisms are exclusive.**  A task that has a
``environment/docker-compose.yaml`` does not get its ``environment/``
uploaded -- that is Harbor's rule, not a choice -- so the payload has to
arrive as a mount (§3).  A task without a compose file gets the upload but
also gets its own network, and there are 15,402 of them.

### `reward/` is visible, so the filter has to be sharp

See §2 and §6.  A rollout that reads the verifier is dropped.

### The verifier needs `input/`

Three verifiers read `input/` without tolerating its absence, so the probe
had to create it -- and so does the environment, which is why the image
ships the directory and the registry fills it.

### Python 3.12, not 3.11

Six verifiers are syntax errors on 3.11.  The image is ubuntu 24.04
(3.12.3); the local `.venv` is 3.11 and the probe has to run under
`.venv312` for those six.
