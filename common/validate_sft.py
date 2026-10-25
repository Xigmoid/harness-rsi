"""An independent checker for the ClawGym SFT dataset.

Why an independent checker
--------------------------

The converter (``build_sft_dataset.py``) builds the samples; this program
re-opens the file that was written and checks it against the **source of
truth** -- the registry's ``instruction.md``, the ledger's reward, and the
OpenAI message schema -- not against the converter's own objects.  A
converter that checks its own output checks nothing.

What is checked
---------------

schema
    ``role`` is one of system/user/assistant/tool; a message carries only the
    keys its role allows; ``tool_calls`` entries have ``id``/``type``/
    ``function{name, arguments}``; the arguments are a JSON *string*.

structure
    The conversation starts with the system message, the first user message
    is the instruction, tool results follow the assistant turn that asked
    for them, every ``tool_call_id`` is answered exactly once, and the last
    message is an assistant one.

fidelity
    The instruction in the sample is byte-identical to the registry's
    ``instruction.md``; the task id, the reward and the trial name in the
    sample are the ones the ledger recorded; the reward is in [0, 1].

one record, one line
    The raw text of the record holds none of the characters that break a
    line-oriented reader.  The *content* may hold them -- the converter
    escapes them, because the content has to be the bytes the model saw --
    but a raw one inside the record would split it into two half-records for
    a reader that uses ``str.splitlines()``.

The fault injection
-------------------

A checker that has only ever seen good data proves nothing.  ``--inject``
runs the checker against deliberately broken copies and asserts that every
mutation is caught:

===========================  ==========================================
mutation                     what must fail
===========================  ==========================================
swap two messages            alternation
drop a tool message          dangling tool_call_id
duplicate a tool message     tool_call_id answered twice
rename a role                role vocabulary
add a stray key              key set
break the tool_call json     arguments are not JSON
touch the instruction        instruction fidelity
change the reward            reward fidelity
===========================  ==========================================

Usage::

    python common/validate_sft.py \
        --dataset work/dataset/clawgym-sft.jsonl \
        --registry work/registry_clawgym \
        --run-root /private/tmp/rsi-harbor/clawgym
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

#: The keys each role may carry, beyond ``role`` and ``content``.
ROLE_KEYS: dict[str, set[str]] = {
    "system": {"content"},
    "user": {"content"},
    "assistant": {"content", "tool_calls", "reasoning_content", "provider_specific_fields"},
    "tool": {"content", "tool_call_id"},
}

#: The keys a tool call may carry.
TOOL_CALL_KEYS = {"id", "type", "function"}
FUNCTION_KEYS = {"name", "arguments"}

#: The characters that break a line-oriented reader.  ``str.splitlines()``
#: breaks on all of them, so a record that contains one is read as two
#: half-records.  They are legitimate inside a JSON string -- and they do
#: occur in the model's output.
LINE_BREAK_CHARS = frozenset(
    "\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"
)


class Report:
    """The checker's findings, counted by kind."""

    def __init__(self) -> None:
        self.problems: Counter[str] = Counter()
        self.examples: dict[str, str] = {}

    def fail(self, kind: str, detail: str) -> None:
        self.problems[kind] += 1
        self.examples.setdefault(kind, detail)

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        if self.ok:
            return "OK"
        lines = [f"{kind}: {count}" for kind, count in self.problems.most_common()]
        for kind in self.problems:
            lines.append(f"  e.g. {kind}: {self.examples[kind]}")
        return "\n".join(lines)


def check_sample(
    row: dict[str, Any],
    report: Report,
    registry: Path | None,
    ledger: dict[str, dict[str, Any]],
    raw: str = "",
) -> None:
    """Check one sample in place.

    ``raw`` is the **text of the record as it is stored in the file**.  It is
    not optional in spirit: one of the checks below can only be made on it.
    """
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        report.fail("no_messages", str(row.get("task")))
        return

    # The two harnesses have different shapes, and the checker has to know
    # which one it is looking at:
    #
    #   * mini ends with the model's final answer, so the last message is an
    #     assistant one.  opencode can be cut off right after a tool result,
    #     in which case the last message is a tool one -- and that tool result
    #     *was* sent to the model, so it belongs in the sample.
    #   * mini's task instruction is the user message verbatim; opencode
    #     wraps it in double quotes on the wire, escaping the inner quotes.
    harness = str(row.get("harness") or row.get("source") or "Mini")
    is_opencode = "opencode" in harness.lower()

    # -- schema ---------------------------------------------------------- #
    roles: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            report.fail("message_not_a_dict", f"index {index}")
            return
        role = message.get("role")
        if role not in ROLE_KEYS:
            report.fail("bad_role", f"index {index}: {role!r}")
            return
        roles.append(role)
        extra = set(message) - ROLE_KEYS[role] - {"role"}
        if extra:
            report.fail("stray_key", f"index {index}: {sorted(extra)}")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            report.fail("content_not_a_string", f"index {index}")
        if role in ("system", "user", "tool") and content is None:
            report.fail("missing_content", f"index {index}: {role}")
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict) or set(call) != TOOL_CALL_KEYS:
                report.fail("bad_tool_call_keys", f"index {index}: {sorted(call) if isinstance(call, dict) else call}")
                continue
            if call.get("type") != "function":
                report.fail("bad_tool_call_type", f"index {index}")
            function = call.get("function") or {}
            if set(function) != FUNCTION_KEYS:
                report.fail("bad_function_keys", f"index {index}: {sorted(function)}")
                continue
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                report.fail("arguments_not_a_string", f"index {index}")
                continue
            try:
                json.loads(arguments)
            except json.JSONDecodeError:
                report.fail("arguments_not_json", f"index {index}: {arguments[:60]!r}")

    # -- the record is one line -------------------------------------------- #
    # A record is one line, and the line ends at "\n".  The characters below
    # break a line-oriented reader (``str.splitlines()`` breaks on all of
    # them), and the model's output contains them -- a Hungarian name in a
    # CSV, a line separator in a report.
    #
    # The converter **escapes** them, so the decoded content still holds the
    # character -- and it must: the content has to be the bytes the model
    # saw, and dropping a character is a bigger lie than keeping one.  What
    # must not happen is a *raw* one inside the record, because then a
    # ``splitlines()`` reader sees two half-records.
    #
    # So the check is on the raw text of the record, not on the content.
    for char in raw:
        if char in LINE_BREAK_CHARS:
            report.fail(
                "line_break_in_record",
                f"U+{ord(char):04X} raw inside the record",
            )
            break

    # -- structure -------------------------------------------------------- #
    if roles[0] != "system":
        report.fail("no_system_first", f"first role {roles[0]!r}")
    if is_opencode:
        # A rollout cut off right after a tool result ends with a tool
        # message, and that observation *was* sent to the model.
        if roles[-1] not in ("assistant", "tool"):
            report.fail("last_not_assistant", f"last role {roles[-1]!r}")
    elif roles[-1] != "assistant":
        report.fail("last_not_assistant", f"last role {roles[-1]!r}")
    if "user" not in roles:
        report.fail("no_user_message", str(row.get("task")))

    # The shape of the conversation.  mini's loop is "model answers -> the
    # tool runs -> the observation -> the model answers again", so the roles
    # run
    #
    #     system, user, (assistant, tool*), assistant
    #
    # with two wrinkles:
    #
    #   * one assistant message may carry several tool calls, and each of
    #     them is answered by its own tool message -- so several tool
    #     messages in a row are normal;
    #   * mini's format-error recovery appends a *user* message carrying the
    #     error text, so a user message may follow an assistant message
    #     directly.
    #
    # The checks that follow all read the raw sequence -- stripping the tool
    # messages would put the assistant turns next to each other and hide
    # exactly the defects they are meant to find.
    if len(roles) >= 2 and roles[1] != "user":
        report.fail("first_turn_not_user", f"role 1 is {roles[1]!r}")
    for index in range(1, len(roles)):
        previous = roles[index - 1]
        current = roles[index]
        if current == "tool" and previous not in ("assistant", "tool"):
            report.fail("tool_not_after_assistant", f"index {index}: {previous} -> tool")
        if current == "assistant" and previous not in ("user", "tool"):
            report.fail("assistant_after_" + str(previous), f"index {index}")
        if current == "user" and previous not in ("system", "user", "assistant", "tool"):
            report.fail("user_after_" + str(previous), f"index {index}")

    # Every assistant turn's tool calls are answered by exactly that many
    # tool messages, immediately after it.
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") != "assistant":
            index += 1
            continue
        wanted = len(message.get("tool_calls") or [])
        seen = 0
        cursor = index + 1
        while cursor < len(messages) and messages[cursor].get("role") == "tool":
            seen += 1
            cursor += 1
        if wanted and seen != wanted and index != len(messages) - 1:
            # The *last* assistant message is exempt: it is the model's final
            # output, it was never sent back to the model, and mini never ran
            # its command -- so its observation is not in the sample at all.
            report.fail(
                "tool_count_mismatch",
                f"index {index}: {wanted} tool calls, {seen} tool messages",
            )
        index = cursor if wanted else index + 1

    asked: Counter[str] = Counter()
    answered: Counter[str] = Counter()
    last_assistant_ids: set[str] = set()
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict) and isinstance(call.get("id"), str):
                    asked[call["id"]] += 1
                    if index == len(messages) - 1:
                        last_assistant_ids.add(call["id"])
        elif message.get("role") == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str):
                report.fail("tool_without_id", f"index {index}")
            else:
                answered[tool_call_id] += 1
    for tool_call_id, count in asked.items():
        if count > 1:
            report.fail("duplicate_tool_call_id", tool_call_id)
        if answered[tool_call_id] == 0:
            # One dangling id is *expected*: the final assistant message is the
            # model's last output, it was never sent back to the model, and
            # mini never ran its command (``COMPLETE_TASK_AND_SUBMIT_FINAL_
            # OUTPUT`` is intercepted).  Its observation is therefore not in
            # the sample.  Any other dangling id is a real defect.
            if tool_call_id not in last_assistant_ids:
                report.fail("dangling_tool_call_id", tool_call_id)
    for tool_call_id, count in answered.items():
        if count > 1:
            report.fail("tool_call_answered_twice", tool_call_id)
        if tool_call_id not in asked:
            report.fail("orphan_tool_message", tool_call_id)

    # -- fidelity --------------------------------------------------------- #
    reward = row.get("reward")
    if not isinstance(reward, (int, float)) or not (0.0 <= float(reward) <= 1.0):
        report.fail("bad_reward", f"{row.get('task')}: {reward!r}")

    if not is_opencode and row.get("instruction_wire") is not None:
        # A mini row carries the instruction verbatim; a row that also
        # carries a *wire* form belongs to opencode, so the harness label
        # and the row disagree.
        report.fail("wire_form_on_a_mini_row", str(row.get("task")))

    if registry is not None:
        instruction_path = registry / "clawgym" / str(row.get("task")) / "instruction.md"
        if instruction_path.exists():
            expected = instruction_path.read_text(encoding="utf-8")
            actual = row.get("instruction")
            if actual != expected:
                report.fail("instruction_mismatch", str(row.get("task")))
            else:
                user_texts = [
                    message.get("content") or ""
                    for message in messages
                    if message.get("role") == "user"
                ]
                if is_opencode:
                    # opencode puts the instruction on the wire wrapped in
                    # double quotes, with the inner quotes escaped and the
                    # backslashes left alone -- so the text the model saw is
                    # not the instruction, it is the instruction in opencode's
                    # encoding.  Looking for the bare instruction here would
                    # flag every task whose prompt contains a quote.
                    wire = '"' + expected.replace('"', '\\"') + '"'
                    if not any(wire in text for text in user_texts):
                        report.fail("instruction_not_in_prompt", str(row.get("task")))
                    recorded_wire = row.get("instruction_wire")
                    if recorded_wire is not None and recorded_wire != wire:
                        report.fail("instruction_wire_mismatch", str(row.get("task")))
                elif not any(expected in text for text in user_texts):
                    report.fail("instruction_not_in_prompt", str(row.get("task")))
        else:
            report.fail("instruction_missing_on_disk", str(row.get("task")))

    if ledger:
        record = ledger.get(str(row.get("task")))
        if record is None:
            report.fail("not_in_ledger", str(row.get("task")))
        else:
            if record.get("reward") is not None and abs(float(record["reward"]) - float(reward)) > 1e-9:
                report.fail("reward_mismatch", str(row.get("task")))
            if record.get("trial_name") != row.get("trial_name"):
                report.fail("trial_mismatch", str(row.get("task")))


# -- the fault injection --------------------------------------------------- #

MUTATIONS: dict[str, Callable[[dict[str, Any]], None]] = {}


def mutation(name: str) -> Callable[[Callable[[dict[str, Any]], None]], Callable[[dict[str, Any]], None]]:
    def register(function: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], None]:
        MUTATIONS[name] = function
        return function

    return register


@mutation("swap two messages")
def _swap(row: dict[str, Any]) -> None:
    messages = row["messages"]
    if len(messages) >= 3:
        messages[1], messages[2] = messages[2], messages[1]


@mutation("drop a tool message")
def _drop_tool(row: dict[str, Any]) -> None:
    messages = row["messages"]
    for index, message in enumerate(messages):
        if message.get("role") == "tool":
            del messages[index]
            return


@mutation("duplicate a tool message")
def _duplicate_tool(row: dict[str, Any]) -> None:
    messages = row["messages"]
    for index, message in enumerate(messages):
        if message.get("role") == "tool":
            messages.insert(index, copy.deepcopy(message))
            return


@mutation("rename a role")
def _rename_role(row: dict[str, Any]) -> None:
    row["messages"][0]["role"] = "assistant"


@mutation("add a stray key")
def _stray_key(row: dict[str, Any]) -> None:
    row["messages"][0]["unexpected"] = "surprise"


@mutation("break the tool_call json")
def _break_arguments(row: dict[str, Any]) -> None:
    for message in row["messages"]:
        for call in message.get("tool_calls") or []:
            call["function"]["arguments"] = "{not json"
            return


@mutation("touch the instruction")
def _touch_instruction(row: dict[str, Any]) -> None:
    row["instruction"] = (row.get("instruction") or "") + " (tampered)"


@mutation("change the reward")
def _change_reward(row: dict[str, Any]) -> None:
    row["reward"] = 1.75


class NotApplicable(Exception):
    """A mutation that does not apply to the row it was given.

    The two harnesses have different shapes, so some mutations only make
    sense for one of them -- tampering with opencode's wire instruction is
    meaningless for a mini row, which has no wire instruction.  Such a
    mutation is *skipped*, not "missed": it is not evidence that the checker
    is broken.
    """


@mutation("tamper the instruction in the prompt")
def _tamper_prompt(row: dict[str, Any]) -> None:
    """Change the instruction inside the prompt the model was given.

    The check that catches this is a *containment* check, so the
    instruction cannot merely be prefixed or suffixed -- it has to be
    changed in a way that removes it from the text.
    """
    instruction = row.get("instruction") or ""
    if not instruction:
        raise NotApplicable("no instruction")
    harness = str(row.get("harness") or row.get("source") or "Mini")
    needle = (
        '"' + instruction.replace('"', '\\"') + '"'
        if "opencode" in harness.lower()
        else instruction
    )
    for message in row.get("messages") or []:
        if message.get("role") != "user":
            continue
        content = message.get("content") or ""
        if needle in content:
            message["content"] = content.replace(
                needle, needle[0] + "X" + needle[1:], 1
            )
            return
    raise NotApplicable("the instruction is not in the prompt")


@mutation("tamper the wire instruction")
def _tamper_wire(row: dict[str, Any]) -> None:
    """Change the recorded wire form of the instruction."""
    if row.get("instruction_wire") is None:
        raise NotApplicable("this harness has no wire instruction")
    row["instruction_wire"] = row["instruction_wire"] + " (tampered)"


@mutation("rename the harness")
def _rename_harness(row: dict[str, Any]) -> None:
    """Claim the row came from the other harness.

    A row that says ``opencode`` is checked with opencode's rules -- the
    wire instruction, the trailing tool message -- so a mislabelled row
    would slip through the checks that belong to its real shape.
    """
    harness = str(row.get("harness") or row.get("source") or "Mini")
    row["harness"] = "Mini" if "opencode" in harness.lower() else "Opencode"


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Read a JSONL file, tolerating a torn last line.

    The ledger is appended to *while* the rollouts run, so a reader can catch
    it mid-write and see half a record.  A checker that dies on that is a
    checker that cannot be run against a live run, so the torn line is
    counted and skipped instead.

    The **raw** lines are returned alongside the records: one of the checks
    can only be made on the text as it is stored, not on the decoded record.
    """
    records: list[dict[str, Any]] = []
    raws: list[str] = []
    torn = 0
    if not path.exists():
        return records, torn, raws
    # Split on "\n" only.  ``str.splitlines()`` also breaks on U+2028, \x0b
    # and eight other characters, and the model's output contains them -- a
    # record holding one of them would be read as two half-records.
    for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            torn += 1
            continue
        raws.append(line)
    return records, torn, raws


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--registry", default="")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--inject",
        action="store_true",
        help="run the fault injection instead of the normal check",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    rows, dataset_torn, raws = read_jsonl(dataset_path)
    if args.limit:
        rows = rows[: args.limit]
        raws = raws[: args.limit]

    registry = Path(args.registry).resolve() if args.registry else None
    ledger: dict[str, dict[str, Any]] = {}
    ledger_torn = 0
    if args.run_root:
        ledger_path = Path(args.run_root) / "progress.jsonl"
        ledger_records, ledger_torn, _ = read_jsonl(ledger_path)
        for record in ledger_records:
            name = record.get("registry_name")
            if name:
                ledger[name] = record

    if args.inject:
        if not rows:
            print("the dataset is empty; nothing to inject into", file=sys.stderr)
            return 2
        base = rows[0]
        report = Report()
        check_sample(base, report, registry, ledger, raw=raws[0] if raws else "")
        if not report.ok:
            print(
                "FATAL: the first sample does not pass the checker, so the "
                "injection would prove nothing:\n" + report.summary(),
                file=sys.stderr,
            )
            return 2
        print(f"the clean sample passes; injecting {len(MUTATIONS)} mutations")
        missed: list[str] = []
        skipped: list[str] = []
        for name, mutate in MUTATIONS.items():
            broken = copy.deepcopy(base)
            try:
                mutate(broken)
            except NotApplicable as reason:
                skipped.append(name)
                print(f"  n/a     {name:28s} -> {reason}")
                continue
            report = Report()
            check_sample(
                broken, report, registry, ledger, raw=raws[0] if raws else ""
            )
            if report.ok:
                missed.append(name)
                print(f"  MISSED  {name}")
            else:
                print(f"  caught  {name:28s} -> {', '.join(report.problems)}")
        if missed:
            print(f"\n{len(missed)} mutation(s) were not caught: {missed}", file=sys.stderr)
            return 1
        if skipped:
            print(f"\n{len(skipped)} mutation(s) did not apply to this sample: {skipped}")
        print("\nall applicable mutations were caught")
        return 0

    report = Report()
    for row, raw in zip(rows, raws):
        check_sample(row, report, registry, ledger, raw=raw)
    print(f"dataset   : {dataset_path}")
    print(f"samples   : {len(rows)}")
    if dataset_torn:
        print(f"torn lines: {dataset_torn} (a torn line in the dataset is a defect)")
    if ledger_torn:
        print(f"ledger torn lines: {ledger_torn} (the run is still appending)")
    if report.ok:
        print("result    : OK -- every sample passes every check")
        return 0
    print("result    : PROBLEMS")
    print(report.summary())
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
