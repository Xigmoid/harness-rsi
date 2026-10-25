# harness-rsi — re-annotating agent trajectories with mini-swe-agent and opencode

Two agent harnesses — **mini-swe-agent** and **opencode** — are run over the
same tasks, in the same environment, against the same model, and their
trajectories are exported as an **OpenAI-API-compatible SFT dataset**.

The harness is the variable in this experiment.  The environment and the model
are held fixed, so the two halves of the dataset are directly comparable: the
same task, the same verifier, the same reward, two different harnesses.

```
ClawGym            15,402 tasks   RUC-AIBOX/ClawGym-Task
OpenThoughts       23,225 tasks   open-thoughts/OpenThoughts-Agent-SFT-31.6K
                                     (issue / tezos / superuser)
                                   -------------------------------
                                   38,627 tasks, split by hash of the
                                   task name, half to each harness
```

**To reproduce it: [`REPRODUCE.md`](./REPRODUCE.md).**

---

## The output

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

* `messages` is the **model-side** view: the exact message list the agent sent
  to the endpoint, plus the model's final answer (which was never sent back).
  It is verified byte for byte against a full packet capture.
* `conversations` is the **training** view: the same content with the tool
  calls inlined and the reasoning folded into the content, so that it
  alternates user/assistant strictly.
* `dataset` / `model` / `harness` are the three keys a trainer filters on;
  `model_id` and `model_provider` are what an auditor follows.

The rollouts that trip a discard rule are **archived, not deleted**: they go to
`invalid/` next to the dataset, with the reason, together with a
`.skipped.jsonl` listing every skipped rollout and why.

The finished dataset is published as a **GitHub Release**, not in git.

---

## The layout

```
opencode_sft/            the pipeline
├── run_rollouts.py          the opencode driver: batches, ledger, harvest
├── run_mini_rollouts.py     the same for mini-swe-agent
├── run_all_stages.sh        the orchestrator: split → run → convert → merge,
│                            with a supervisor that restarts a dead driver
├── record_proxy.py          the recording proxy: it writes the wire traffic to
│                            disk and injects the API key on the way out
├── schema.py                the shared schema: `dataset` / `model` / `harness`
├── clawgym/                 the ClawGym half: registry builder, converters,
│                            validator, merger
├── images/                  the three base images (opencode is baked in)
├── mini_sft/                the mini config for the OpenThoughts families
└── EXECUTION_PLAN.md        the traps, with the evidence

rsi/                       the two Harbor adapters
├── harbor_agent.py         mini-swe-agent: pins the config, mounts the bundle
├── opencode_agent.py       opencode: pins the build, exports the raw session
└── paths.py                the paths the host and the container agree on

scripts/                   registry builders, the toolchain build, the release
datasets/base/toolchain/   the mini-swe-agent bundle, as a Dockerfile
audit/                     what Harbor and mini-swe-agent actually do
```

---

## Quick start

```bash
cp .env.example .env      # then edit it: the endpoint, the model, the run root
make setup                 # two virtual environments
make images                # the three base images
make toolchain             # the mini-swe-agent bundle that is bind-mounted
make registry              # both task registries, rebuilt from HuggingFace
make annotate              # the rollouts -- this is the part that takes days
make dataset && make verify # convert, validate, merge
```

Before committing to the days, `make check` runs the whole path end to end in
about ten minutes: one task per harness, converted and validated.

---

## What is not in this repository

| | |
|---|---|
| the two task registries (1.3 GB) | rebuilt from two public HuggingFace datasets by `make registry` |
| the toolchain bundle (668 MB) | rebuilt by `make toolchain` |
| the Docker images (~2.5 GB) | rebuilt by `make images` |
| the rollouts and the dataset | produced by `make annotate`; the dataset is published as a GitHub Release |

Everything that is a *derived* artefact is derived data.  What is in here is
the code that produces it, and the notes on what it cost to get right.

---

## The traps this implementation already handles

The ones that cost the most time to find are written up in full, with the
evidence, in [`opencode_sft/EXECUTION_PLAN.md`](./opencode_sft/EXECUTION_PLAN.md).
The short list:

* Docker's default address pool is sixteen networks wide — every new trial dies
  with `all predefined address pools have been fully subnetted`, which looks
  like an agent failure and is not one.  Every trial joins one shared network.
* the `harbor job` is a child of the driver and does not die with it, so a
  killed driver leaves trials running that nobody harvests.
* `$!` for a backgrounded shell *function* is the wrapper's pid, so "kill the
  driver" kills a wrapper and the same stream runs three times over.
* the recording proxy must **override** the `Authorization` header, not only
  fill in a missing one: opencode sends its own placeholder, and the endpoint
  rejects it.
* opencode's `tool_calls.arguments` is JavaScript's `JSON.stringify` — Python's
  `json.dumps` produces the same object with a different token sequence, so
  the data would not be the bytes the model saw.
* mini's endpoint lives in the *container's* config, not in the host
  environment: pointing a run at a second endpoint silently keeps using the
  first, while the labels say otherwise.

---

## License

MIT.
