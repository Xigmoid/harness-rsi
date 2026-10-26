"""Generate a random set of instructions, and the Harbor registry to run them.

This is a **control probe** for the re-annotation pipeline.  The instructions
here are *not* from the published dataset, so nothing about them can be
compared with the published rollouts.  What they do test is the part the
published rows cannot: that an instruction the pipeline has never seen goes
through opencode end to end and comes back as a complete, checkable
trajectory -- the instruction reaches the model unchanged, the tool calls and
their results are all there, and the whole thing can be replayed.

The instructions are **generated, not hand-written**.  A seeded RNG picks the
archetype and fills its slots, so the same seed always produces the same ten
instructions (so a run is reproducible and a failure can be replayed), while a
different seed produces a different set of ten.  Nothing here is a copy of a
published task: the slots -- the module names, the data, the bug, the domain
-- are drawn from pools that do not exist upstream.

Every generated instruction is **self-contained**: the working directory is
empty, so the instruction has to carry the seed code and the acceptance
criteria itself.

Two properties are asserted before anything is written, because both would
silently corrupt the trajectory:

* the instruction contains **no backslash**.  opencode puts the user message
  on the wire wrapped in double quotes, escaping the inner double quotes but
  *not* the backslashes, so an instruction with a backslash does not survive
  the round trip and the converter could not prove the model saw the same
  bytes the file holds.
* the instruction has **no trailing newline**, so that the sha256 of the
  file is the sha256 of the text (the converter strips the instruction it
  reads back out of the trial directory).

Usage::

    python common/tools/gen_random_probe.py --seed 20261023 --n 10
    python opencode_sft/run_rollouts.py \
        --registry work/random_probe/registry \
        --run-root ~/.rsi-harbor/random-probe \
        --batch-size 10 --concurrency 5 --families random
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import shutil
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from common.shared_network import compose_snippet  # noqa: E402

#: The plain family image: an empty ``/app`` on ubuntu 24.04, with python3.12,
#: node and opencode.  The generated tasks are self-contained, so they need
#: no per-task setup at all -- the image *is* the environment, exactly like
#: the issue / tezos / superuser families of the real re-annotation.
IMAGE = "rsi/oc-base-plain:1"

FAMILY = "random"

TASK_TOML = """version = "1.0"

[metadata]
author_name = "OpenThoughts-Agent re-annotation (random probe)"
author_email = "noreply@localhost"
difficulty = "hard"
category = "software-engineering"
tags = ["agent", "terminal", "random-probe"]

[verifier]
timeout_sec = 600.0

[agent]
timeout_sec = 1800.0

# The shared base image.  Setting ``docker_image`` makes Harbor use it as
# is, so the probe needs no image build at all.
[environment]
docker_image = "{image}"
"""

TEST_SH = """#!/bin/bash
# The probe does not score anything: the verifier is disabled, exactly like
# the upstream datagen job.  This file exists so the task directory is a
# complete Harbor task.
mkdir -p /logs/verifier
echo 0 > /logs/verifier/reward.txt
"""

#: The image of a task that needs a toolchain the plain image does not have.
#:
#: The plain image is ubuntu 24.04 with python3.12 and node -- no Go.  The
#: Go port task needs a Go toolchain, and the SWE-Smith **Go** image is
#: exactly that: debian 12 with Go 1.24.5 at ``/usr/local/go``, deliberately
#: not on ``PATH`` (which is what the published Go rollouts saw too).  That
#: image also carries python3.11, so the python original of the port has
#: something to run under.
GO_IMAGE = "rsi/oc-base-swesmith-go:1"

#: The overlay that moves the workdir of that image to ``/app``.
#:
#: Every probe task works in ``/app``, and the plain image's workdir is
#: already ``/app``; the Go image's workdir is ``/testbed``, so the task
#: carries a one-line compose overlay.  This is the same mechanism the
#: SWE-Smith family of the real re-annotation uses -- an overlay on top of
#: the shared image, never a per-task build.
GO_COMPOSE_YAML = """# The Go toolchain is only in the SWE-Smith Go image, whose workdir is
# /testbed.  The probe's tasks all work in /app.
services:
  main:
    working_dir: /app
"""

# --------------------------------------------------------------------------- #
# the slots the generator fills
# --------------------------------------------------------------------------- #

DOMAINS = [
    "order reconciliation",
    "sensor readings",
    "access tokens",
    "build artifacts",
    "shipping manifests",
    "payment retries",
    "index shards",
    "audit trails",
    "quote snapshots",
    "feature flags",
]

MODULE_NAMES = [
    "ledger",
    "scheduler",
    "dispatcher",
    "collector",
    "resolver",
    "compactor",
    "rotator",
    "normalizer",
    "allocator",
    "reconciler",
]

COLLECTIONS = [
    "events",
    "records",
    "entries",
    "samples",
    "fragments",
    "batches",
    "documents",
    "measurements",
]

#: The two formats used by the parser tasks.  Neither contains a backslash.
LINE_FORMATS = [
    "2026-03-11 08:14:22 | INFO  | worker-3 | batch complete (128 items)",
    "2026-03-11 08:14:25 | WARN  | worker-1 | retry 2 of 5 for shard 07",
    "2026-03-11 08:15:01 | ERROR | worker-0 | connection reset by peer",
]


def _slot(rng: random.Random) -> dict[str, str]:
    """One draw of the slots."""
    module = rng.choice(MODULE_NAMES)
    other = rng.choice([m for m in MODULE_NAMES if m != module])
    return {
        "domain": rng.choice(DOMAINS),
        "module": module,
        "other_module": other,
        "collection": rng.choice(COLLECTIONS),
        "line": rng.choice(LINE_FORMATS),
        "n": str(rng.randint(3, 97)),
        "limit": str(rng.choice([10, 25, 50, 100, 250, 500])),
        "threshold": f"{rng.randint(2, 40) / 10:.1f}",
    }


# --------------------------------------------------------------------------- #
# the ten archetypes
# --------------------------------------------------------------------------- #

Archetype = Callable[[random.Random, dict[str, str]], tuple[str, str]]


def arch_fix_range(rng: random.Random, s: dict[str, str]) -> tuple[str, str]:
    module = s["module"]
    title = f"Fix the off-by-one in {module}.py"
    body = f'''The working directory /app is empty. Create the following file and fix the bug described below.

Create `/app/{module}.py`:

```python
def page(items, size, page_number):
    """Return one page of `items`, 1-indexed."""
    start = page_number * size
    end = start + size
    return items[start:end]
```

Write a script `/app/{module}_check.py` that builds a list of {s["n"]} distinct
strings and asserts that paging over it with page size {s["limit"]} returns
exactly the same items, in the same order, as slicing the list directly.

What is wrong
-------------
`page` is documented as 1-indexed, but the body treats `page_number` as
0-indexed: asking for page 1 returns the *second* page, and asking for the
page after the last one returns an empty list instead of an error.

How to decide when it is right
------------------------------
- `page(items, size, 1)` must return `items[0:size]`.
- `page(items, size, 2)` must return `items[size:2 * size]`.
- The last page may be shorter than `size`; that is expected.
- A page number beyond the last one must raise `IndexError` with a message
  that names the page number.

Deliverables
------------
- `/app/{module}.py` with the corrected `page` function.
- `/app/{module}_check.py`, a script that exits 0 and prints a short summary
  when every rule above holds.
'''
    return title, body


def arch_add_validation(rng: random.Random, s: dict[str, str]) -> tuple[str, str]:
    module = s["module"]
    collection = s["collection"]
    title = f"Add validation to the {module} config loader"
    body = f'''The working directory /app is empty. Create the files described below.

Create `/app/{module}_config.py`:

```python
DEFAULTS = {{
    "retries": 3,
    "timeout": 30.0,
    "batch_size": {s["limit"]},
}}


def load_config(overrides=None):
    """Merge `overrides` on top of DEFAULTS and return the result."""
    config = dict(DEFAULTS)
    config.update(overrides or {{}})
    return config
```

The loader currently accepts anything. Add validation.

Rules
-----
- `retries` must be an `int` (not a `bool`) between 0 and 10 inclusive.
  A `bool` is rejected, because `True` is an `int` in python but is never a
  sensible retry count.
- `timeout` must be a number strictly greater than 0 and at most 600.
- `batch_size` must be a positive `int`.
- A key that is not in DEFAULTS is rejected.
- Every rejection raises `ValueError` whose message names the offending key.

How to decide when it is right
------------------------------
Write `/app/{module}_config_check.py` that exercises one accepted value and
one rejected value per rule, prints the message of each `ValueError`, and
exits 0 only if every rejection raised and every accepted value was returned
unchanged.

Deliverables
------------
- `/app/{module}_config.py` with the validation in place.
- `/app/{module}_config_check.py`.
'''
    return title, body


def arch_write_tests(rng: random.Random, s: dict[str, str]) -> tuple[str, str]:
    module = s["module"]
    collection = s["collection"]
    title = f"Write the test suite for {module}.py"
    body = f'''The working directory /app is empty. Create the files described below.

Create `/app/{module}.py`:

```python
import re

WORD = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Split `text` into lowercase word tokens."""
    return WORD.findall(text.lower())


def counts(text):
    """Return a dict of token -> number of occurrences."""
    out = {{}}
    for token in tokenize(text):
        out[token] = out.get(token, 0) + 1
    return out


def top(counts_dict, n):
    """Return the n most frequent tokens, most frequent first.

    Ties are broken by the token's own alphabetical order.
    """
    return sorted(counts_dict.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
```

Write the test suite for it.

Requirements
------------
- Use `pytest`. Put the tests in `/app/test_{module}.py`.
- Cover `tokenize`: lowercase folding, splitting on everything that is not
  a letter or a digit, and the empty string.
- Cover `counts`: an empty string gives an empty dict, repeated tokens are
  counted, and punctuation does not create tokens.
- Cover `top`: fewer tokens than `n`, exactly `n` tokens, more than `n`, and
  the tie-breaking rule.
- At least one test must fail against a deliberately broken implementation
  of `top` that breaks ties by insertion order instead of alphabetically.

How to decide when it is right
------------------------------
`cd /app && python -m pytest test_{module}.py -q` passes, and
`python -m pytest /app/test_{module}.py -q` reports at least 8 tests.

Deliverables
------------
- `/app/test_{module}.py`.
'''
    return title, body


def arch_refactor(rng: random.Random, s: dict[str, str]) -> tuple[str, str]:
    module = s["module"]
    other = s["other_module"]
    title = f"Remove the duplicated {module} logic"
    body = f'''The working directory /app is empty. Create the files described below.

Create `/app/{module}.py`:

```python
def normalize_name(value):
    return value.strip().lower().replace(" ", "_")


def normalize_email(value):
    return value.strip().lower()


def normalize_tag(value):
    return value.strip().lower().replace(" ", "_")
```

Create `/app/{other}.py`:

```python
def normalize_name(value):
    return value.strip().lower().replace(" ", "_")


def normalize_email(value):
    return value.strip().lower()


def normalize_tag(value):
    return value.strip().lower().replace(" ", "_")
```

The two modules duplicate each other. The refactoring has to keep every
behaviour identical.

Rules
-----
- The normalisation logic lives in exactly one place.
- Both modules keep their existing public names and their existing
  behaviour -- `normalize_email` must still *not* replace spaces, and
  `normalize_tag` must still replace them. That difference is intentional.
- Neither module imports the other.
- A third module `/app/{module}_common.py` holds the shared implementation
  and is imported by both.

How to decide when it is right
------------------------------
Write `/app/{other}_check.py` that imports both modules, runs the three
functions over a list of inputs that includes values with leading and
trailing spaces, uppercase letters and inner spaces, and prints the results.
It exits 0 only if the two modules agree on every input and the behaviour
matches the original rules.

Deliverables
------------
- `/app/{module}_common.py`, `/app/{module}.py`, `/app/{other}.py`.
- `/app/{other}_check.py`.
'''
    return title, body


def arch_build_cli(rng: random.Random, s: dict[str, str]) -> tuple[str, str]:
    module = s["module"]
    collection = s["collection"]
    title = f"Build the {module} command line tool"
    body = f'''The working directory /app is empty. Build a command line tool from scratch.

The tool is called `{module}`. It manages a JSON Lines collection of
{collection} stored in a file whose path is given by `--file`.

Subcommands
-----------
- `{module} add --file PATH --name NAME --value VALUE` appends one record.
- `{module} list --file PATH` prints one record per line, in the order the
  records were added.
- `{module} sum --file PATH --field FIELD` prints the sum of the numeric
  `field` of every record, as a plain number with no units.
- `{module} count --file PATH` prints how many records the file holds.

Rules
-----
- A record is a JSON object with at least the keys `name` and `value`.
- `--file` is required for every subcommand, and the tool exits with a
  non-zero status and a message on stderr when it is missing, when the file
  does not exist, or when the file is not valid JSON Lines.
- `sum` on a field that is missing from a record, or whose value is not a
  number, is an error: the tool exits non-zero and says which record is
  at fault.
- `sum` over an empty file prints `0`.
- The tool is a single executable file `/app/{module}` with a shebang, and
  it runs under `python3` only -- no third party packages, no pip install.

How to decide when it is right
------------------------------
Write `/app/{module}_check.sh` that runs the tool through one add, a list,
a sum and a count, plus the two error cases, and exits 0 only if every
expected output matches exactly.

Deliverables
------------
- `/app/{module}`, the executable.
- `/app/{module}_check.sh`, the check script.
'''
    return title, body


def arch_diagnose(rng: random.Random, s: dict[str, str]) -> tuple[str, str]:
    module = s["module"]
    collection = s["collection"]
    title = f"Diagnose and fix the crash in {module}.py"
    body = f'''The working directory /app is empty. Create the files described below.

Create `/app/{module}.py`:

```python
import json


def load_{collection}(path):
    """Return the {collection} of a JSON Lines file as a list of dicts."""
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize({collection}):
    """Return the number of {collection} and the number of distinct names."""
    names = {{item["name"] for item in {collection}}}
    return {{"count": len({collection}), "names": len(names)}}
```

Create `/app/{module}_data.jsonl` with {s["n"]} lines, where every line is a
JSON object with the keys `name` and `value`, and the name of the {s["n"]}th
record is the same as the name of the {int(s["n"]) // 2}th one. The value of
each record is a distinct integer.

The script below crashes:

```python
from {module} import load_{collection}, summarize

{collection} = load_{collection}("/app/{module}_data.jsonl")
print(summarize({collection}))
```

What to deliver
---------------
- The reason the crash happens, written out in prose in
  `/app/{module}_diagnosis.md`. Do not only patch the symptom: the diagnosis
  has to name the exact expression that fails and the value it fails on.
- A fixed `/app/{module}.py` that handles the file you were given, and
  handles a file where one line is not valid JSON by skipping that line and
  reporting it on stderr.
- `/app/{module}_run.py`, which loads the data and prints the summary.

How to decide when it is right
------------------------------
`python3 /app/{module}_run.py` prints a summary whose `count` is {s["n"]} and
whose `names` is fewer than {s["n"]}, and exits 0.
'''
    return title, body


def arch_parser(rng: random.Random, s: dict[str, str]) -> tuple[str, str]:
    module = s["module"]
    title = f"Implement the {module} log parser"
    body = f'''The working directory /app is empty. Create the files described below.

The file `/app/{module}.log` has lines of this shape:

```
{s["line"]}
```

The four fields are separated by ` | `: a timestamp, a level, a worker name
and a message. The message part itself may contain ` | `.

Create `/app/{module}_parse.py` with:

```python
def parse_line(line):
    """Return (timestamp, level, worker, message) for one log line.

    Returns None for a line that does not have all four fields.
    """
```

Rules
-----
- The split is on the **first three** separators only, so the message keeps
  any ` | ` it contains.
- A line that does not have four fields returns `None`.
- A line that is empty or only whitespace returns `None`.
- The timestamp is returned unchanged, as a string.
- The level is returned upper-cased.

Create `/app/{module}.log` with {s["n"]} lines, at least one of which is
malformed and at least one of which has a ` | ` inside the message.

How to decide when it is right
------------------------------
Write `/app/{module}_check.py` that parses every line of the log, counts the
levels, prints the count per level and the number of malformed lines, and
exits 0 only if the counts add up to the number of lines in the file.

Deliverables
------------
- `/app/{module}_parse.py`.
- `/app/{module}.log`.
- `/app/{module}_check.py`.
'''
    return title, body


def arch_http(rng: random.Random, s: dict[str, str]) -> tuple[str, str]:
    module = s["module"]
    collection = s["collection"]
    title = f"Implement the {module} HTTP service"
    body = f'''The working directory /app is empty. Build a small HTTP service from
scratch, using only the python standard library.

The service keeps a collection of {collection} in memory.

Endpoints
---------
- `GET /{collection}` returns a JSON array of every record.
- `POST /{collection}` accepts a JSON object body and stores it, answering
  `201` with the stored object.
- `GET /{collection}/<id>` returns one record, or `404` if there is no such
  id.
- `DELETE /{collection}/<id>` removes a record and answers `204`, or `404`
  if there is no such id.
- Any other path answers `404`.

Rules
-----
- The server listens on port 8080, bound to 0.0.0.0.
- The ids are assigned by the server and are increasing integers, starting
  at 1.
- A `POST` whose body is not a JSON object answers `400` and does not store
  anything.
- The server handles each request in its own thread, so two concurrent
  requests are both answered.

Create `/app/{module}_server.py` with the server, and
`/app/{module}_client.py` with a script that starts the server, exercises
all five cases with `urllib`, prints every status code, and shuts the server
down.

How to decide when it is right
------------------------------
`python3 /app/{module}_client.py` exits 0 and prints `201` for the create,
`200` for the list and the fetch, `404` for the two missing ids, `204` for
the delete, and `400` for the bad body.

Deliverables
------------
- `/app/{module}_server.py`.
- `/app/{module}_client.py`.
'''
    return title, body


def arch_optimize(rng: random.Random, s: dict[str, str]) -> tuple[str, str]:
    module = s["module"]
    collection = s["collection"]
    title = f"Speed up the hot loop in {module}.py"
    body = f'''The working directory /app is empty. Create the files described below.

Create `/app/{module}.py`:

```python
def dedupe({collection}):
    """Return the {collection} without duplicates, keeping the first
    occurrence of each name."""
    seen = []
    out = []
    for item in {collection}:
        if item["name"] not in seen:
            seen.append(item["name"])
            out.append(item)
    return out
```

With {s["n"]} records this is slow, because `seen` is a list and the
`in` check is a linear scan.

Rules
-----
- The behaviour stays exactly the same: the first occurrence of each name
  is kept, and the order of the result is the order of the input.
- The membership test must not be a linear scan.
- The function still accepts any iterable of dicts with a `name` key.

How to decide when it is right
------------------------------
Create `/app/{module}_bench.py` that builds {s["n"]} records with
{s["limit"]} distinct names, times `dedupe` before and after the change,
prints both timings and the length of the result, and exits 0 only if the
result length is {s["limit"]} and the result is identical to the result of
the original implementation (keep a copy of the original as
`dedupe_original` in the same file, so the comparison is inside the
benchmark).

Deliverables
------------
- `/app/{module}.py`, with both the original and the optimised version.
- `/app/{module}_bench.py`.
'''
    return title, body


def arch_port_go(rng: random.Random, s: dict[str, str]) -> tuple[str, str]:
    module = s["module"]
    title = f"Port {module}.py to Go"
    body = f'''The working directory /app is empty. Create the files described below.

Create `/app/{module}.py`:

```python
import re

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Split `text` into lowercase word tokens."""
    return TOKEN.findall(text.lower())


def counts(text):
    """Return a dict of token -> number of occurrences."""
    out = {{}}
    for token in tokenize(text):
        out[token] = out.get(token, 0) + 1
    return out
```

Port it to Go.

Rules
-----
- `/app/{module}.go` has `package main` and a function
  `func Tokenize(text string) []string` and `func Counts(text string) map[string]int`.
- The port behaves identically to the python original on the same inputs,
  including on text that contains digits, punctuation, uppercase letters and
  text that is empty.
- The Go program reads a string from standard input and prints the counts,
  one `token count` pair per line, sorted by token.

How to decide when it is right
------------------------------
- `go version` works. If `go` is not on `PATH`, find it -- on this machine
  the toolchain lives in `/usr/local/go`, so `/usr/local/go/bin/go` works.
- `/app/{module}_check.sh` runs the python original and the Go program on
  the same {s["n"]} inputs, compares the outputs, and exits 0 only if they
  are identical for every input.

Deliverables
------------
- `/app/{module}.go`.
- `/app/{module}_check.sh`.
'''
    return title, body


ARCHETYPES: list[Archetype] = [
    arch_fix_range,
    arch_add_validation,
    arch_write_tests,
    arch_refactor,
    arch_build_cli,
    arch_diagnose,
    arch_parser,
    arch_http,
    arch_optimize,
    arch_port_go,
]

#: The archetype that needs the Go toolchain, and therefore the Go image.
GO_ARCHETYPE_INDEX = ARCHETYPES.index(arch_port_go)


# --------------------------------------------------------------------------- #
# the generator
# --------------------------------------------------------------------------- #


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_instruction(text: str) -> None:
    """Refuse an instruction that would not survive the wire round trip."""
    if "\\" in text:
        raise ValueError(
            "the instruction contains a backslash; opencode escapes quotes on "
            "the wire but not backslashes, so the instruction would not "
            "round-trip"
        )
    if text != text.strip():
        raise ValueError("the instruction has leading or trailing whitespace")
    if text.endswith("\n"):
        # The file on disk holds the instruction and *nothing else*, so that
        # the sha256 of the file is the sha256 of the text.  The converter
        # strips what it reads back out of the trial directory, so a trailing
        # newline would make the two hashes differ by one byte.
        raise ValueError("the instruction must not end with a newline")
    if not text:
        raise ValueError("the instruction is empty")


def generate(n: int, seed: int) -> list[dict[str, Any]]:
    """The n generated instructions."""
    rng = random.Random(seed)
    picks = list(range(len(ARCHETYPES)))
    rng.shuffle(picks)
    while len(picks) < n:
        picks += picks
    picks = picks[:n]

    out: list[dict[str, Any]] = []
    for position, archetype_index in enumerate(picks):
        slots = _slot(rng)
        title, body = ARCHETYPES[archetype_index](rng, slots)
        text = f"# {title}\n\n{body.rstrip()}"
        check_instruction(text)
        name = f"random-{position:02d}-{archetype_index:02d}"
        image = GO_IMAGE if archetype_index == GO_ARCHETYPE_INDEX else IMAGE
        out.append(
            {
                "task": name,
                "family": FAMILY,
                "title": title,
                "archetype": ARCHETYPES[archetype_index].__name__,
                "image": image,
                "instruction": text,
                "instruction_sha256": sha256(text),
                "slots": slots,
            }
        )
    return out


def build_registry(
    entries: list[dict[str, Any]], root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Write the Harbor task directories, the manifest and the task index."""
    if root.exists():
        shutil.rmtree(root)
    family_dir = root / FAMILY
    family_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    index: list[dict[str, Any]] = []
    for position, entry in enumerate(entries):
        target = family_dir / entry["task"]
        (target / "tests").mkdir(parents=True, exist_ok=True)
        (target / "environment").mkdir(parents=True, exist_ok=True)

        # The instruction is written byte for byte, with **no** trailing
        # newline, so that the sha256 of the file is the sha256 of the text
        # the converter reads back out of the trial.
        (target / "instruction.md").write_text(entry["instruction"], encoding="utf-8")
        image = entry.get("image") or IMAGE
        (target / "task.toml").write_text(TASK_TOML.format(image=image), encoding="utf-8")
        test_sh = target / "tests" / "test.sh"
        test_sh.write_text(TEST_SH, encoding="utf-8")
        test_sh.chmod(0o755)

        # Every task joins the one shared Docker network; see
        # ``common/shared_network.py`` for why.
        overlay = compose_snippet()
        if image != IMAGE:
            # The task needs a toolchain the plain image does not have, so
            # it runs on the Go image and carries the overlay that moves the
            # workdir to /app as well.
            overlay += GO_COMPOSE_YAML
        (target / "environment" / "docker-compose.yaml").write_text(
            overlay, encoding="utf-8"
        )

        manifest.append(
            {
                "task": entry["task"],
                "family": FAMILY,
                "path": str(target.relative_to(root)),
                "image": image,
                "cwd": "/app",
                "instruction_sha256": entry["instruction_sha256"],
                "instruction_chars": len(entry["instruction"]),
                "has_setup": image != IMAGE,
                "dataset_rows": [position],
                "source_rows": [position],
                "canonical_row": position,
                "archetype": entry["archetype"],
                "title": entry["title"],
            }
        )
        index.append(
            {
                "row": position,
                "task": entry["task"],
                "family": FAMILY,
                "trace_source": "main",
                "model": "hosted_vllm/glm",
                "model_provider": "hosted_vllm",
                "result": None,
                "run_id": None,
                "trial_name": None,
                "date": None,
                "episode": None,
                "n_messages": None,
                "instruction_sha256": entry["instruction_sha256"],
                "instruction_path": f"instructions/{entry['task']}.md",
                "extracted": True,
            }
        )
    return manifest, index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10, help="how many instructions")
    parser.add_argument("--seed", type=int, default=20261023, help="the RNG seed")
    parser.add_argument(
        "--out", default=str(HERE / "random_probe"), help="the output directory"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = generate(args.n, args.seed)

    registry = out_dir / "registry"
    manifest, index = build_registry(entries, registry)

    instructions_dir = out_dir / "instructions"
    instructions_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        (instructions_dir / f"{entry['task']}.md").write_text(
            entry["instruction"], encoding="utf-8"
        )

    with (registry / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in manifest:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with (out_dir / "index.jsonl").open("w", encoding="utf-8") as handle:
        for record in index:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with (out_dir / "instructions.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(
                json.dumps(
                    {
                        "task": entry["task"],
                        "title": entry["title"],
                        "archetype": entry["archetype"],
                        "instruction_sha256": entry["instruction_sha256"],
                        "instruction_chars": len(entry["instruction"]),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    # The ground truth for the verifier: the task -> instruction hash table,
    # in the same shape the published dataset gives.
    ground_truth = out_dir / "ground_truth.jsonl"
    with ground_truth.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(
                json.dumps(
                    {"task": entry["task"], "instruction_sha256": entry["instruction_sha256"]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    summary = {
        "n": len(entries),
        "seed": args.seed,
        "image": IMAGE,
        "registry": str(registry),
        "instructions": [
            {
                "task": e["task"],
                "title": e["title"],
                "archetype": e["archetype"],
                "chars": len(e["instruction"]),
            }
            for e in entries
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
