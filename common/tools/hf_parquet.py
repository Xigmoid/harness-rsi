"""Where the published OpenThoughts parquet is on this machine.

The HuggingFace cache keeps one snapshot directory per revision, and the
revision hash changes whenever the dataset is re-uploaded.  So the file is
**found** rather than written down: a hard-coded snapshot hash is a path that
works on exactly one machine and breaks silently everywhere else.

Usage::

    from hf_parquet import published_parquet

    table = pq.read_table(published_parquet(), columns=["task", "conversations"])
"""

from __future__ import annotations

from pathlib import Path

#: The dataset repo, in the cache's own naming.
REPO_ID = "open-thoughts--OpenThoughts-Agent-SFT-31.6K"

#: The file inside the snapshot.
FILE = "train.parquet"


def hub_root(hub: Path | str | None = None) -> Path:
    """The HuggingFace cache directory."""
    if hub is not None:
        return Path(hub)
    return Path.home() / ".cache" / "huggingface" / "hub"


def published_parquet(hub: Path | str | None = None) -> Path:
    """The published parquet.

    When the cache holds more than one revision, the **most recently
    written** one wins: that is the one a fresh ``huggingface-cli download``
    just fetched.
    """
    snapshots = hub_root(hub) / f"datasets--{REPO_ID}" / "snapshots"
    candidates = [path for path in snapshots.glob(f"*/{FILE}") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"{FILE} is not in the HuggingFace cache under {snapshots}. "
            f"Download it first, e.g. "
            f"`huggingface-cli download open-thoughts/{REPO_ID.replace('--', '-')}`."
        )
    candidates.sort(key=lambda path: path.stat().st_mtime)
    return candidates[-1]


def published_parquet_default(hub: Path | str | None = None) -> str:
    """The published parquet as a string, or ``""`` when it is not cached.

    For an argparse default: the path must not raise at import time, because
    ``--help`` has to work on a machine that never downloaded the dataset.
    """
    try:
        return str(published_parquet(hub))
    except (FileNotFoundError, OSError):
        return ""
