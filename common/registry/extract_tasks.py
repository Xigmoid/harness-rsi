"""Extract ``(task, instruction)`` pairs from OpenThoughts-Agent-SFT-31.6K.

The published dataset stores every rollout as a chat transcript whose *first*
message is the terminus-2 system prompt with the task description glued to the
end of it:

    <terminus-2 prompt>

    Task Description:
    <the task instruction>

    Current terminal state:
    Current Terminal Screen:
    ...

The instruction is what has to stay byte-identical when the dataset is
re-annotated with another harness, so it is cut out of that first message here
and written to disk on its own -- one file per task, raw text, nothing added.

Outputs (under ``--out``):

==========================  ===================================================
``index.jsonl``             one record per *dataset row* (all 31,600)
``instructions.jsonl``      one record per *(task, instruction variant)*
``instructions/<task>.md``  the raw instruction text of one task
``instructions/variants/``  the additional variants of a task
``failed.jsonl``            rows whose instruction could not be cut out
``summary.json``            counts, per-family breakdown, integrity checks
==========================  ===================================================

A task id is **not** a unique key for an instruction: 766 tasks carry two
variants of the same task, differing by an appended ``CONSTRAINT:`` footer.
The instructions are therefore keyed by ``(task, sha256)``, and every row of
``index.jsonl`` points at the exact file its instruction was written to.

Usage::

    python common/registry/extract_tasks.py
    python common/registry/extract_tasks.py --limit 200      # quick check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO = Path(__file__).resolve().parent.parent.parent

REPO_ID = "open-thoughts/OpenThoughts-Agent-SFT-31.6K"

#: The template marker that separates the harness prompt from the task.
TASK_MARKER = "Task Description:"

#: The template marker that starts the terminal-state section.  Everything
#: between the two markers is the task instruction.
STATE_MARKER = "Current terminal state:"

#: The task families of the dataset, keyed by the id prefix.  The dataset's
#: own ``trace_source`` column is *not* the family -- it holds ``main`` /
#: ``summarization-1-answers`` / ``null``, i.e. which sub-agent produced the
#: trace -- so the family has to be recovered from the task id.
FAMILY_PREFIXES = {
    "swesmith-": "swesmith",
    "superuser-": "superuser",
    "tezos-": "tezos",
    "issue-": "issue",
}

#: Columns worth reading; ``conversations`` is the big one.
COLUMNS = [
    "task",
    "conversations",
    "trace_source",
    "model",
    "model_provider",
    "result",
    "run_id",
    "trial_name",
    "date",
    "episode",
]

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def family_of(task: str) -> str:
    """The task family of a task id, or ``other``."""
    for prefix, family in FAMILY_PREFIXES.items():
        if task.startswith(prefix):
            return family
    return "other"


def safe_name(task: str) -> str:
    """A filesystem-safe name for a task id.

    The ids in this dataset (``tezos-02839``, ``superuser-010082_copy0004``)
    are already safe, but a task id must never be able to escape the output
    directory.
    """
    name = _SAFE_NAME.sub("_", task).strip("._") or "task"
    if name != task:
        # Keep the mapping reversible for the rare id that needs escaping.
        digest = hashlib.sha256(task.encode("utf-8")).hexdigest()[:8]
        name = f"{name}-{digest}"
    return name


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def first_user_message(conversations: Any) -> str | None:
    """The first user/human message of a conversation.

    ``conversations`` is normally a list of ``{"role", "content"}`` dicts, but
    the column is a plain string in some rows of the source parquet (the
    viewer serialises it that way when it truncates a cell), so a JSON string
    is accepted as well.
    """
    if isinstance(conversations, str):
        try:
            conversations = json.loads(conversations)
        except json.JSONDecodeError:
            return None
    if not isinstance(conversations, list):
        return None
    for message in conversations:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or message.get("from") or ""
        if role in ("user", "human"):
            content = message.get("content")
            if content is None:
                content = message.get("value")
            if isinstance(content, str):
                return content
    return None


def extract_instruction(message: str) -> tuple[str | None, str | None]:
    """Cut the task instruction out of the first message.

    Returns ``(instruction, template_sha256)``.  ``instruction`` is ``None``
    when the message does not carry the expected markers; ``template_sha256``
    identifies the harness prompt that preceded the task, so a template change
    in the source data shows up as a new hash instead of silently blending in.
    """
    marker_at = message.find(TASK_MARKER)
    if marker_at < 0:
        return None, None
    template = message[:marker_at]
    body = message[marker_at + len(TASK_MARKER) :]
    state = re.search(r"\n[ \t]*" + re.escape(STATE_MARKER), body)
    if state is None:
        return None, sha256(template)
    instruction = body[: state.start()].strip()
    if not instruction:
        return None, sha256(template)
    return instruction, sha256(template)


def parquet_files(repo_id: str, revision: str | None) -> list[str]:
    """The parquet files of the dataset repository."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo_id, repo_type="dataset", revision=revision)
    return sorted(f for f in files if f.endswith(".parquet"))


def download(repo_id: str, filenames: Iterable[str], revision: str | None) -> list[Path]:
    """Download (or reuse the cache of) the given parquet files."""
    from huggingface_hub import hf_hub_download

    paths = []
    for name in filenames:
        paths.append(
            Path(
                hf_hub_download(
                    repo_id,
                    name,
                    repo_type="dataset",
                    revision=revision,
                )
            )
        )
    return paths


def iter_rows(paths: Iterable[Path], limit: int | None) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(global_row_index, row)`` from the parquet files."""
    import pyarrow.parquet as pq

    row_index = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256, columns=COLUMNS):
            for row in batch.to_pylist():
                yield row_index, row
                row_index += 1
                if limit is not None and row_index >= limit:
                    return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO_ID, help="dataset repo id")
    parser.add_argument("--revision", default=None, help="dataset revision")
    parser.add_argument(
        "--out",
        default=str(REPO / "work" / "tasks"),
        help="output directory",
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N rows")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="use the local HuggingFace cache instead of downloading",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="delete the previous instruction files before writing",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    instructions_dir = out_dir / "instructions"
    if args.clean and instructions_dir.exists():
        # Only the generated instruction tree is removed; the output directory
        # itself and everything else in it are left alone.
        shutil.rmtree(instructions_dir)
    instructions_dir.mkdir(parents=True, exist_ok=True)

    files = parquet_files(args.repo, args.revision)
    print(f"parquet files: {len(files)}")
    paths = files if args.skip_download else download(args.repo, files, args.revision)

    index_path = out_dir / "index.jsonl"
    instructions_path = out_dir / "instructions.jsonl"
    failed_path = out_dir / "failed.jsonl"

    family_counter: Counter[str] = Counter()
    trace_source_counter: Counter[str] = Counter()
    result_counter: Counter[str] = Counter()
    template_counter: Counter[str] = Counter()
    model_counter: Counter[str] = Counter()
    task_rows: Counter[str] = Counter()
    task_family: dict[str, str] = {}
    # (task, sha256) -> record of the instruction variant.
    variants: dict[tuple[str, str], dict[str, Any]] = {}
    task_order: list[str] = []

    n_rows = 0
    n_extracted = 0
    n_failed = 0
    n_unique = 0
    messages_total = 0

    variants_dir = instructions_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    with (
        index_path.open("w", encoding="utf-8") as index_file,
        failed_path.open("w", encoding="utf-8") as failed_file,
    ):
        for row_index, row in iter_rows(paths, args.limit):
            n_rows += 1
            task = row.get("task") or ""
            trace_source = row.get("trace_source")
            model = row.get("model")
            result = row.get("result")

            family = family_of(task)
            family_counter[family] += 1
            trace_source_counter[str(trace_source)] += 1
            result_counter[str(result)] += 1
            model_counter[str(model)] += 1

            conversations = row.get("conversations")
            n_messages = len(conversations) if isinstance(conversations, list) else 0
            messages_total += n_messages

            message = first_user_message(conversations)
            instruction = None
            template = None
            if message is not None:
                instruction, template = extract_instruction(message)
            if template:
                template_counter[template] += 1

            digest = None
            instruction_path = None
            if instruction is not None:
                n_extracted += 1
                digest = sha256(instruction)
                task_rows[task] += 1
                task_family.setdefault(task, family)
                key = (task, digest)
                if key not in variants:
                    # The first variant of a task is the canonical one; the
                    # others are the same task with a different instruction.
                    canonical = not any(t == task for t, _ in variants)
                    if canonical:
                        relative = f"instructions/{safe_name(task)}.md"
                        target = instructions_dir / f"{safe_name(task)}.md"
                    else:
                        relative = f"instructions/variants/{safe_name(task)}__{digest[:8]}.md"
                        target = variants_dir / f"{safe_name(task)}__{digest[:8]}.md"
                    target.write_text(instruction, encoding="utf-8")
                    variants[key] = {
                        "task": task,
                        "family": family,
                        "sha256": digest,
                        "path": relative,
                        "canonical": canonical,
                        "n_rows": 0,
                        "n_messages_min": n_messages,
                        "n_messages_max": n_messages,
                        "n_chars": len(instruction),
                        "_text": instruction,
                    }
                    if canonical:
                        n_unique += 1
                        task_order.append(task)
                record = variants[key]
                record["n_rows"] += 1
                record["n_messages_min"] = min(record["n_messages_min"], n_messages)
                record["n_messages_max"] = max(record["n_messages_max"], n_messages)
                instruction_path = record["path"]
            else:
                n_failed += 1
                failed_file.write(
                    json.dumps(
                        {
                            "row": row_index,
                            "task": task,
                            "trace_source": trace_source,
                            "reason": "no task marker"
                            if message is None
                            else "no terminal-state marker",
                            "first_message_head": (message or "")[:200],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            index_file.write(
                json.dumps(
                    {
                        "row": row_index,
                        "task": task,
                        "family": family,
                        "trace_source": trace_source,
                        "model": model,
                        "model_provider": row.get("model_provider"),
                        "result": result,
                        "run_id": row.get("run_id"),
                        "trial_name": row.get("trial_name"),
                        "date": row.get("date"),
                        "episode": row.get("episode"),
                        "n_messages": n_messages,
                        "instruction_sha256": digest,
                        "instruction_path": instruction_path,
                        "extracted": digest is not None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if n_rows % 2000 == 0:
                print(
                    f"  {n_rows} rows, {n_unique} unique tasks, "
                    f"{n_extracted} extracted, {n_failed} failed",
                    flush=True,
                )

    # How do the variants of a task relate to each other?  The upstream data
    # augments a task by appending a constraint footer, so most pairs differ
    # by a suffix; a pair that differs in the middle is a rewritten task and
    # has to be looked at by hand.
    variant_relations: Counter[str] = Counter()
    tasks_with_variants: list[str] = []
    for task in task_order:
        task_variants = [record for (t, _), record in variants.items() if t == task]
        if len(task_variants) < 2:
            continue
        tasks_with_variants.append(task)
        canonical = next(r for r in task_variants if r["canonical"])
        for other in task_variants:
            if other is canonical:
                continue
            a, b = canonical["_text"], other["_text"]
            if b.startswith(a):
                variant_relations["suffix_added"] += 1
            elif a.startswith(b):
                variant_relations["suffix_removed"] += 1
            else:
                variant_relations["rewritten"] += 1

    with instructions_path.open("w", encoding="utf-8") as instructions_file:
        for (task, digest), record in variants.items():
            payload = {k: v for k, v in record.items() if k != "_text"}
            instructions_file.write(
                json.dumps(payload, ensure_ascii=False) + "\n"
            )

    summary = {
        "repo": args.repo,
        "revision": args.revision,
        "rows": n_rows,
        "unique_tasks": n_unique,
        "instruction_variants": len(variants),
        "tasks_with_multiple_variants": len(tasks_with_variants),
        "variant_relations": dict(variant_relations.most_common()),
        "extracted": n_extracted,
        "failed": n_failed,
        "mean_messages_per_row": (messages_total / n_rows) if n_rows else 0,
        "rows_by_family": dict(family_counter.most_common()),
        "unique_tasks_by_family": dict(
            Counter(task_family.values()).most_common()
        ),
        "rows_by_trace_source": dict(trace_source_counter.most_common()),
        "rows_by_result": dict(result_counter.most_common()),
        "rows_by_model": dict(model_counter.most_common()),
        "distinct_templates": len(template_counter),
        "template_counts": dict(template_counter.most_common()),
        "files": {
            "index": str(index_path),
            "instructions": str(instructions_path),
            "instruction_files": str(instructions_dir),
            "variant_files": str(variants_dir),
            "failed": str(failed_path),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
