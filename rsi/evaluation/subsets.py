"""Fixed dev / held-out task splits.

The split is the anti-overfitting backbone of the whole protocol: the optimizer
only ever sees ``dev``, ``heldout`` is reported.  A split that changes between
generations invalidates every comparison, so the split files carry a SHA256
that is recorded in the ledger and checked on every run (trap T-11).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "Subset",
    "SPLIT_SEED",
    "DEFAULT_SPLITS",
    "make_split",
    "load_subset",
    "save_subset",
    "subset_sha256",
    "verify_subset",
    "list_subsets",
]

#: Fixed across the whole study.  Never change it: changing it invalidates
#: every cross-generation comparison.
SPLIT_SEED = 20260715

#: Default dev/held-out sizes, per the protocol of the plan.
DEFAULT_SPLITS: dict[str, tuple[int, int]] = {
    "tb21": (300, 145),
    "sbv": (300, 200),
}


@dataclass(frozen=True)
class Subset:
    """One named, immutable task subset."""

    name: str
    split: str  # "dev" | "heldout"
    benchmark: str
    task_ids: tuple[str, ...]
    seed: int = SPLIT_SEED
    source: str = ""

    def __post_init__(self) -> None:
        if self.split not in ("dev", "heldout"):
            raise ValueError(f"subset split must be 'dev' or 'heldout', got {self.split!r}")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError(f"subset {self.name!r} contains duplicate task ids")

    @property
    def n(self) -> int:
        return len(self.task_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "split": self.split,
            "benchmark": self.benchmark,
            "seed": self.seed,
            "source": self.source,
            "task_ids": list(self.task_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Subset":
        return cls(
            name=data["name"],
            split=data["split"],
            benchmark=data.get("benchmark", ""),
            task_ids=tuple(data["task_ids"]),
            seed=int(data.get("seed", SPLIT_SEED)),
            source=data.get("source", ""),
        )

    def sha256(self) -> str:
        return subset_sha256(self)


def subset_sha256(subset: Subset) -> str:
    """Hash the *content* of a subset (not its file bytes)."""
    payload = json.dumps(
        {
            "benchmark": subset.benchmark,
            "name": subset.name,
            "seed": subset.seed,
            "split": subset.split,
            "task_ids": list(subset.task_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_split(
    benchmark: str,
    task_ids: Sequence[str],
    *,
    n_dev: int,
    n_heldout: int | None = None,
    seed: int = SPLIT_SEED,
    source: str = "",
) -> tuple[Subset, Subset]:
    """Split ``task_ids`` into a dev and a held-out subset.

    The split is a seeded shuffle, so it is reproducible and independent of
    the order in which the task list was collected.
    """
    ids = list(dict.fromkeys(task_ids))
    if n_dev + (n_heldout or 0) > len(ids):
        raise ValueError(
            f"cannot split {len(ids)} tasks into dev={n_dev} + heldout={n_heldout}"
        )
    rng = random.Random(f"{benchmark}:{seed}")
    shuffled = ids[:]
    rng.shuffle(shuffled)
    dev_ids = tuple(sorted(shuffled[:n_dev]))
    rest = shuffled[n_dev:]
    if n_heldout is not None:
        rest = rest[:n_heldout]
    heldout_ids = tuple(sorted(rest))
    dev = Subset(
        name=f"{benchmark}_dev",
        split="dev",
        benchmark=benchmark,
        task_ids=dev_ids,
        seed=seed,
        source=source,
    )
    heldout = Subset(
        name=f"{benchmark}_heldout",
        split="heldout",
        benchmark=benchmark,
        task_ids=heldout_ids,
        seed=seed,
        source=source,
    )
    return dev, heldout


def make_split_by_ratio(
    benchmark: str,
    task_ids: Sequence[str],
    *,
    dev_ratio: float,
    seed: int = SPLIT_SEED,
    source: str = "",
) -> tuple[Subset, Subset]:
    """Split by ratio instead of by absolute size.

    The protocol of the plan fixes the *ratio* (300/445 for Terminal-Bench,
    300/500 for SWE-bench Verified) and the absolute numbers follow from the
    size of the benchmark that is actually installed.  This keeps the split
    meaningful when the available task set is smaller than the one the plan
    assumed -- Terminal-Bench 2.0 in Harbor's registry has 89 tasks, not the
    445 of 2.1.
    """
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError(f"dev_ratio must be in (0, 1), got {dev_ratio}")
    ids = list(dict.fromkeys(task_ids))
    if len(ids) < 2:
        raise ValueError(f"cannot split {len(ids)} task(s) into dev and heldout")
    n_dev = int(round(len(ids) * dev_ratio))
    n_dev = max(1, min(n_dev, len(ids) - 1))
    return make_split(
        benchmark, ids, n_dev=n_dev, n_heldout=len(ids) - n_dev, seed=seed, source=source
    )


def save_subset(subset: Subset, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = subset.to_dict()
    payload["sha256"] = subset.sha256()
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def load_subset(path: str | Path) -> Subset:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Subset.from_dict(data)


def verify_subset(path: str | Path) -> Subset:
    """Load a subset and check its recorded SHA256 (trap T-11)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    subset = Subset.from_dict(raw)
    recorded = raw.get("sha256")
    actual = subset.sha256()
    if recorded and recorded != actual:
        raise ValueError(
            f"subset {path} was modified: recorded sha256 {recorded} != actual {actual}. "
            "Changing a split invalidates every cross-generation comparison."
        )
    return subset


def list_subsets(directory: str | Path) -> list[Subset]:
    out = []
    for p in sorted(Path(directory).glob("*.json")):
        try:
            out.append(verify_subset(p))
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return out
