# RSI — Recursive Harness Self-Improvement as a typed-configuration search

> **Two things live in this repository.**
>
> 1. **The RSI framework** — the research plan in
>    [`01-research-report.md`](./01-research-report.md) and
>    [`02-coding-plan.md`](./02-coding-plan.md): a typed-configuration search
>    over agent harnesses.  Start below.
> 2. **The harness-agent SFT annotation** (`opencode_sft/`) — the same two
>    harnesses (**mini-swe-agent** and **opencode**) re-run over 38,627 tasks
>    in the *same* environment against the *same* model, and their
>    trajectories exported as an OpenAI-API-compatible SFT dataset.
>    **To reproduce that: [`REPRODUCE.md`](./REPRODUCE.md).**

An implementation of the research plan in [`01-research-report.md`](./01-research-report.md)
and [`02-coding-plan.md`](./02-coding-plan.md).

RHI ([arXiv:2607.15524](https://arxiv.org/abs/2607.15524)) improves an agent
harness by letting an LLM rewrite the harness prompt, and keeps the generations
that win a pairwise comparison.  This framework keeps the *loop* and replaces
the *search space*:

| | RHI | this framework |
|---|---|---|
| candidate | a natural-language harness prompt | a **typed configuration** (`HarnessSpec`) |
| optimiser | an LLM rewriting the prompt | a typed search: random search → per-field bandit → CEM |
| signal | LLM-as-judge pairwise win rate | an **automatic benchmark score** |
| attribution | post-hoc embedding clustering | **per-field effect sizes** |
| benchmark | 30 synthetic ML-research tasks | 8 benchmarks over 4 capabilities |

The point of the typed search is not convenience: it makes the *edit* the unit
of credit assignment.  Every generation changes 1–2 named fields, so "this
generation is better" becomes "changing `agents[1].contract.out_fields` to
`(findings, evidence)` is better", which is a claim that survives being
checked by someone else.

---

## 1. The research question this framework is built for

> **Is the optimal harness contract task-specific?**

RHI's component-level analysis found that *contracts* change the most across
tasks and cluster most clearly by task, while *roles* are almost
indistinguishable.  The information-theoretic reading is that a contract
should be a **sparse, task-dependent** interface between an orchestrator and
its subagents: the subagent should see only what the downstream decision
needs.

That hypothesis is only testable if the same harness can be optimised against
different task families and the results compared.  So the framework supports
two optimisation modes:

### Specialised (default) — one harness per capability

The optimiser's objective is the composite score of **one** capability family
(`math`, `reasoning`, `instruction_following`, `code`).  Running the loop once
per family produces one specialised harness per family, and the comparison
across families is the experiment.

### Joint — one harness for everything (the control)

The objective is the composite over the whole suite.  This is the control
experiment that answers *"how much can a single generic harness do?"*, the
counterpart of the random-search baseline.  It is **not** the default, for
three reasons:

1. **The noise floor.**  A composite's resolution is the sum of the members'
   variances, so `δ_min` grows while the per-benchmark effect does not.  With
   an edit budget of 1–2 fields per generation, generations are the scarcest
   resource in the study; spending them on a noisier objective is expensive.
2. **The cost asymmetry.**  A SWE-bench task costs minutes in a container; an
   AIME task costs seconds.  A composite over benchmarks with a 100× spread
   in cost *and* noise is dominated by the expensive ones.
3. **Homogenisation is the known failure mode.**  Athanor documented four
   agents sharing one harness converging onto the same recipe within an hour.
   Optimising one harness across heterogeneous tasks *guarantees* that
   failure at the harness level — and it would make the task-specificity
   hypothesis untestable, because a generic compromise harness is the expected
   outcome whether or not the hypothesis is true.

The per-benchmark scores are recorded either way, so a joint run can be broken
down per capability afterwards, and the **transfer matrix** (rows: harness,
columns: benchmark) can be built from the same data.  A diagonal-dominant
matrix means the specialisation is real; a flat matrix means the contract
dimensions are task-independent — which is a publishable negative result, not a
failure.

### The protocol that follows from the choice

One evolution chain per capability family, then a cross-evaluation of every
specialised harness on every suite — the transfer matrix is the experiment's
primary result:

```bash
# one chain per capability, all branching from the same baseline (gen 0)
for suite in math reasoning instruction_following code; do
  rsi gen propose --objective $suite --parent 0 --optimizer bandit_ucb --seed 1
done
# -> gen01 (math), gen02 (reasoning), gen03 (instruction_following), gen04 (code)

# cross-evaluate: the specialised harness on a foreign suite
rsi gen evaluate --gen 1 --suite reasoning      # the math harness on reasoning
rsi analyze transfer                             # rows = harness, columns = benchmark
```

The generation *number* is not the chain: `--parent 0` is what makes a chain.
Several proposals may share a parent, and each gets its own number — a
proposal never overwrites an existing spec (there is an `--overwrite` flag
for the rare case where that is what you want).

The seed is **per chain**: the framework mixes the objective into the seed
(crc32, not Python's salted `hash`), so the same `--seed` on two chains gives
two *different* proposals instead of the same edit twice — and the same
`--seed` on the same chain is still exactly reproducible.

The joint mode is run **once**, as the control, with the same edit budget and
the same number of generations:

```bash
rsi gen propose --objective all --optimizer bandit_ucb --seed 1
```

Two things are kept identical across the chains, because they are the
confounders that would otherwise decide the outcome:

* the **contract vocabulary** is the same for every capability (§4.2) — a
  capability-specific vocabulary would rig the comparison before it starts;
* the **dev/held-out split** of every benchmark is seeded, hashed and reused
  across generations, so a capability chain never sees a different task set
  than the joint chain.

The outcome is decided by two numbers, not by taste:

* **diagonal dominance** of the transfer matrix → the specialisation is real
  and the RHI hypothesis survives;
* a **flat** matrix → the contract dimensions are task-independent, which is a
  negative result worth publishing — and it is only reachable because the
  specialised chains were run and cross-evaluated.

---

## 2. Installation

```bash
python3.12 -m venv .venv && . .venv/bin/activate     # Harbor needs >= 3.12
pip install -e ".[harbor,dev]"                        # mini-swe-agent 2.4.6, harbor 0.24.0
```

Two environments are used in practice:

* the **host** runs the framework, the optimiser, the statistics and the
  ledger (Python >= 3.11 is fine);
* the **container** runs the agent.  Harbor installs mini-swe-agent *inside*
  the task container with `uv tool install mini-swe-agent==2.4.6` — the
  `rsi` package therefore has to be importable **inside the task image**
  (publish it, or mount the source and add it to `PYTHONPATH`).

## 3. Quick start

```bash
# scaffold: gen-00 spec, frozen base config, Harbor job templates, dev/held-out splits
rsi init --suite math --model openai/glm5.3-flash \
         --api-base http://<host>:8000/v1 --api-key EMPTY

# measure the baseline and the resolution of the pipeline
rsi gen evaluate --gen 0 --suite math
rsi noise-floor --gen 0 --subset aime_dev --seeds 1234 1235 1236 1237 1238

# let the optimiser propose the next generation for that capability
rsi gen propose --objective math --optimizer random_search
rsi gen evaluate --gen 1 --suite math
rsi gen compare 1 0

# analyse
rsi analyze fields            # per-field effect sizes
rsi analyze transfer          # cross-capability transfer matrix
rsi analyze homogenization    # per-field agreement between consecutive generations
rsi report                    # ledger/report.md
```

`rsi init` writes **placeholder** task ids when no task list is supplied.
Replace them before reporting any number:

```bash
rsi init --suite math --tasks aime=/path/to/aime_task_ids.txt
```

## 4. The pieces

```
rsi/
├── schema.py        typed HarnessSpec: validation, canonical SHA256, (de)serialization
├── fields.py        the registry of optimizable fields (the attribution skeleton)
├── benchmarks.py    the benchmark registry and the capability suites
├── rhi_agent.py     RhiAgent: the multi-hop loop on mini-swe-agent
├── hop_runner.py    hop scheduling, contract rendering, payload extraction
├── adapter.py       spec -> mini-swe-agent config, with the C-4 expansion check
├── harbor_agent.py  a Harbor adapter that pins the generation's config
├── ifeval_checkers.py  the 25 IFEval instruction checkers (port of Google's reference)
├── optimizer/       random_search (L0) | bandit_ucb, bandit_thompson (L1) | cem (L2)
├── evaluation/      subsets | runner (harbor/mini/synthetic) | stats | noise_floor | suite
├── ledger/          the generation ledger, the cross-generation diff, the report
└── cli.py           the command line interface
```

Two more top-level directories sit beside `rsi/`:

```
tools/mini_swe_agent/     the local mini-swe-agent CLI setup (glm5.3-flash @ vLLM)
                          overrides.yaml -> regen.py -> local-glm.yaml (self-contained
                          config; the built-in mini.yaml is a *replacement*, not a merge base)
opencode_sft/             SFT data production from real agent rollouts
                          dataset/terminal-5task/ holds the 5-task capture -> OpenAI
                          conversion (10/10 verified byte-for-byte against the wire)
```

### 4.1 The spec

```yaml
version: 0
agents:
  - id: orchestrator
    role: coordinator
    instruction_template: standard
    instruction_slots: {goal: primary, required_outputs: [summary], ...}
    contract:
      out_fields: [plan, summary]
      in_fields: [task, findings, evidence, patch]
      format: json
      max_tokens: 800
      free_text: true
    model: openai/glm5.3-flash
    tools: [read_file, grep_file, run_bash]
workflow:
  - {id: h1, source: orchestrator, target: investigator, pass_fields: [task, plan]}
  - {id: h3, source: investigator, target: orchestrator, pass_fields: [findings, evidence]}
auxiliary_rules:
  acceptance_gates: [...]
  failure_fallback: [retry_once, degrade_to_free_text]
  recall_triggers: [...]
runtime: {max_iterations: 40, observation_max_chars: 10000, context_keep_recent: 20,
          seed: 1234, temperature: 0.0, max_hops: 6, hop_context_max_chars: 6000,
          min_delegations: 1}
```

Validation is fail-fast and covers: unique ids, a well-formed workflow (one
orchestrator that every agent reports back to), hop fields that the source can
produce *and* the target declared it accepts, a task that enters through the
orchestrator, and bounded runtime knobs.

**Instructions are slots, never prose.**  A free-text instruction cannot be
attributed and lets the optimiser drift into noise, so the instruction is a
fixed template with typed slots (`goal`, `required_outputs`,
`forbidden_outputs`, `verification`).

### 4.2 The contract vocabulary is shared across capabilities

The vocabulary in `fields.py` spans code (`patch`, `diff`, `test_results`),
math (`answer`, `derivation`, `constraint`) and reasoning (`check`,
`candidates`, `assumption`).  It is deliberately **the same vocabulary for
every benchmark**: a capability-specific vocabulary would rig the
cross-capability comparison before it started.  What differs between
capabilities is *which* subset of the vocabulary the optimiser selects — and
that is the measurement.

### 4.3 The agent

`RhiAgent` is a subclass of mini-swe-agent's `DefaultAgent`:

* every agent has its **own message list** — a subagent never sees the
  orchestrator's or another subagent's transcript;
* the context of a subagent is **filtered through its contract**, so only the
  fields it declared in `in_fields` reach it;
* the orchestrator receives **structured payloads**, not raw trajectories;
* a subagent may not submit the final answer, and the orchestrator may not
  submit before it has delegated at least `runtime.min_delegations` times —
  without that rule the model happily does the whole task itself and the
  contract/hop dimensions are never exercised (set `min_delegations: 0` for
  the single-agent ablation);
* a payload that cannot be parsed is handled by `auxiliary_rules.failure_fallback`
  (`retry_once` / `degrade_to_free_text` / `abort`) and **counted**, never
  allowed to crash the run.

Both action protocols are supported and the rendered prompts follow the
protocol the model actually speaks:

* `text` (**default**) — the model emits a code block, parsed with the
  tolerant action regex.  This is the only protocol that works inside a
  Harbor container: litellm 1.106 imports its MCP handler as soon as a
  request carries `tools`, and Harbor installs mini-swe-agent **without**
  `mcp` (`audit/harbor_mini_swe_agent.md` §5.1), so every tool-call request
  dies with `ModuleNotFoundError: No module named 'mcp'`.  The Harbor job
  must use `rsi.harbor_agent.RsiMiniSweAgent` so that Harbor cannot switch
  the model class back (see the audit);
* `toolcall` — the model emits an OpenAI tool call; this is mini-swe-agent
  v2's recommended path and works locally (and in a container that has
  `mcp` installed).

### 4.4 The measurement

* **dev / held-out** splits, seeded and archived with a SHA256 that is
  re-checked on every run — a changed split invalidates the whole study;
* **multi-seed continuisation**: a task's score is the fraction of seeds it
  passed, so the per-task difference is continuous instead of binary;
* **paired bootstrap** (10k resamples) + **exact McNemar**; a generation is
  accepted only when the CI lower bound is above zero *and* McNemar rejects;
* **the noise floor first**: `rsi noise-floor` measures what a *zero* edit
  looks like, and reports the smallest detectable improvement `δ_min`.  If
  `δ_min` exceeds the improvement one generation is expected to produce, the
  protocol needs more seeds or more tasks — not a better optimiser;
* **failed runs count as failures**; they are never dropped.

### 4.5 The ledger

One immutable YAML record per generation: the spec, its SHA256, the diff
against the parent, the per-task scores, the statistics against the parent,
the cost and the runtime metadata.  The report (`ledger/report.md`) contains
the score table, the per-generation diffs, the per-field effect sizes, the
transfer matrix and the homogenization curve.

The homogenization metric is the **mean per-field agreement** between a
generation and **its own parent** (1.0 = identical configurations).  Every
field of the shared field space contributes one pair of unit vectors built
from a *shared* vocabulary, so the measure is sensitive to list fields
(`contract.in_fields`, `contract.out_fields`, `workflow[].pass_fields`) and
to numeric fields — the encoding it replaces reported 1.0 for a generation
that had actually changed a contract field.  The curve follows the
**lineage**, not the generation numbers, because the specialised mode runs
several chains in parallel and two consecutive numbers are usually two
*siblings*.

A report built from the synthetic test double carries a
**"SYNTHETIC DATA — NOT A MEASUREMENT"** banner; the synthetic backend is for
plumbing tests only.

---

## 5. The benchmarks

| key | capability | Harbor dataset | tasks | cost |
|---|---|---|---|---|
| `sbv` | code | `swebench-verified` 1.0 | 500 | heavy |
| `tb21` | code | `terminal-bench` 2.0 | 89 | heavy |
| `livecode` | code | `livecodebench` 6.0 | 100 | light |
| `aime` | math | `aime` 1.0 | 60 | light |
| `ineqmath` | math | `ineqmath` 1.0 | 100 | light |
| `gpqa` | reasoning | `gpqa-diamond` 1.0 | 198 | light |
| `arcagi` | reasoning | `arc_agi_2` 1.0 | 167 | light |
| `reagym` | reasoning | `reasoning-gym-hard` | 288 | light |
| `ifeval` | instruction following | local task set (not in the registry) | 541 | light |

Suites: `code`, `math`, `reasoning`, `instruction_following`, `all`.

**Two deviations from the plan, both deliberate:**

* the plan assumes Terminal-Bench **2.1 with 445 tasks**; Harbor 0.24.0's
  registry carries **2.0 with 89 tasks**.  The dev/held-out *ratio* of the
  plan (300/445) is kept and applied to what is available (60/29).
* SWE-bench Verified is 500 tasks in the registry, so the plan's 300/200
  split is reproduced exactly.

The dev/held-out ratio is the invariant, not the absolute number: the absolute
numbers follow from the benchmark that is actually installed.

### 5.1 IFEval — the local task set

IFEval is **not** in Harbor's registry, so `scripts/build_ifeval_tasks.py`
generates it: one Harbor task per prompt (541 tasks), each with a verifier
that applies that prompt's verifiable instructions.

```bash
# 1. write the shared base image and build it
python scripts/build_ifeval_tasks.py --write-base --build-base

# 2. generate the 541 task directories
python scripts/build_ifeval_tasks.py --out datasets/ifeval
```

The **base image** (`rsi/ifeval-base:1`) carries the verifier dependencies
(pytest + the CTRF reporter, langdetect, nltk with the punkt models) so the
verifier is deterministic and offline, and the `rsi` package itself in
`/opt/rsi`, exposed through `PYTHONPATH` — because the `uv tool` venv that
Harbor creates for mini-swe-agent is isolated, so a `pip install` into the
image's system Python would *not* be visible to the agent.  Each task only
pins `environment.docker_image`; no per-task Dockerfile is built.

The reward is the IFEval **strict accuracy**: a prompt counts as followed only
when *all* of its instructions are followed.  `rsi/ifeval_checkers.py` is a
faithful port of Google's reference implementation, and
`tests/test_ifeval_differential.py` runs it against a verbatim mirror of that
reference (`tests/vendor/ifeval_official/`) over all 541 prompts × a response
corpus — 40k+ comparisons that must agree.  Two deliberate differences, both
required for a reproducible benchmark: a missing parameter is an error instead
of a random value, and a repeated instruction id is combined with a logical
AND.

---

## 6. The traps this implementation already handles

| trap | where it is handled |
|---|---|
| the model emits ` ```bash `, mini expects ` ```mswea_bash_command ` | tolerant `action_regex` in `adapter.py` (text protocol only) |
| an unregistered model makes litellm raise | `cost_tracking: ignore_errors` + token counts as the cost proxy |
| a self-made end-of-task marker is ignored | the native `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` is used, and only the orchestrator may issue it |
| a tool that is not on `PATH` | the tools are part of the spec and rendered into the prompt |
| Harbor's `mini` config is layered under the generation | `assert_full_expansion` + the pinned Harbor adapter |
| a broken contract payload crashes the run | `failure_fallback` + a parse-failure counter |
| the retry policy differs between generations | it is written into the frozen job file, and the retry count is recorded |
| a failed run is dropped and the score inflated | failures are counted as failures |
| Harbor installs mini-swe-agent without `mcp` → every tool call dies | the default protocol is `text`; the finding is documented in the audit §5.1 |
| the `rsi` package is invisible to Harbor's isolated `uv tool` venv | the package is copied to `/opt/rsi` and exposed through `PYTHONPATH` |
| `:latest` images | pin the digest in the job file |
| a split file is overwritten | the split carries a SHA256 that is re-checked |
| `to_dict` drops fields | the spec has its own hand-written serialization |

---

## 7. Reproducing a run

Every generation's mini config is written next to its run artefacts
(`runs/genNNN/<subset>/`), the job file that was used is written verbatim, and
the trial directory contains the exact command line Harbor assembled
(`/logs/agent/rsi-injected-config.txt`).  A generation can therefore be
re-run from the ledger alone.

```bash
rsi spec show 3          # the spec of a generation
rsi spec diff 2 3        # what changed
rsi gen compare 3 2      # was it accepted
```

## 8. Tests

```bash
pytest -q          # no network, no Docker
```

The suite covers the schema's validation rules, the field registry, the
contract rendering and the payload extraction, the statistics (bootstrap,
McNemar, the acceptance rule, Holm-Bonferroni), the subsets' determinism and
tamper detection, the optimizers' edit budget, the ledger's round trip, the
Harbor job invariance, and the whole CLI loop in a temporary project.

Two test groups deserve their own note:

* `tests/test_ifeval_checkers.py` — a hand-written positive/negative case
  for every one of the 25 instruction types, cross-checked against Google's
  reference implementation;
* `tests/test_ifeval_differential.py` — the same comparison over **all 541
  IFEval prompts** × a response corpus (40k+ checks) and a deterministic fuzz
  corpus.  Any semantic drift in `rsi/ifeval_checkers.py` fails the build.
