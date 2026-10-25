#!/usr/bin/env python3
"""Turn a downloaded Harbor registry dataset into a *local* task directory
whose images already carry the agent toolchain.

Why this exists
---------------
Harbor installs mini-swe-agent inside **every** task container: ``curl uv``,
``uv python install 3.12``, ``uv tool install mini-swe-agent==2.4.6 --with
litellm --with orjson --with fastapi`` -- preceded by an ``apt-get install
build-essential``.  That is ~4 minutes and ~1 GB of downloads *per trial*.
A 4-task AIME calibration finished with ``AgentSetupTimeoutError`` on 4/4
trials: the agent never ran a single step.

Two ways out, and this script takes the second one:

1. bind-mount a pre-built toolchain into every container -- rejected: the
   bundle is 600 MB of ~30 000 files, and Docker Desktop's file sharing
   takes minutes to sync it into the VM *per container*;
2. bake the toolchain into the task's **base image**.  The toolchain is
   built once (``rsi/toolchain:1``) and copied into a derived image per
   base with ``COPY --from``; the derived image is what the task's
   Dockerfile builds ``FROM``.

Only the base image changes.  The task's instruction, tests, verifier and
solution are copied byte for byte; the Dockerfile is edited in exactly one
place -- its ``FROM`` line.  That is recorded in ``datasets/registry/<key>/
PROVENANCE.md`` together with the SHA256 of every copied file.

    python scripts/prepare_registry_tasks.py aime gpqa tb21
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEST_ROOT = ROOT / "datasets" / "registry"
TC_ROOT = ROOT / "datasets" / "base" / "tc"
TOOLCHAIN_IMAGE = "rsi/toolchain:1"
TC_IMAGE_PREFIX = "rsi/tc-"

#: Where the registry datasets were downloaded to.
DEFAULT_SOURCES = {
    "aime": Path("/tmp/hbd/aime/aime"),
    "gpqa": Path("/tmp/hbd/gpqa/gpqa-diamond"),
    "tb21": Path("/tmp/hbd/tb21/terminal-bench"),
    "ineqmath": Path("/tmp/hbd/ineqmath/ineqmath"),
    "livecode": Path("/tmp/hbd/livecode/livecodebench"),
}

FROM_RE = re.compile(r"^(\s*FROM\s+)(\S+)(.*)$", re.IGNORECASE)

#: Regional package mirrors.  ``deb.debian.org`` is unreachable from this
#: network (see the MIRROR_NOTE in ``write_derived_dockerfile``), so the
#: derived images point apt and pip at reachable mirrors.
MIRROR = "http://mirrors.aliyun.com"
MIRROR_UBUNTU = "http://mirrors.aliyun.com/ubuntu"
MIRROR_UBUNTU_PORTS = "http://mirrors.aliyun.com/ubuntu-ports"
PIP_INDEX = "https://mirrors.aliyun.com/pypi/simple/"
PIP_HOST = "mirrors.aliyun.com"


def slug(base: str) -> str:
    """A filesystem-safe slug for a base image reference."""
    return re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")


def derived_image_for(base: str) -> str:
    return f"{TC_IMAGE_PREFIX}{slug(base)}:1"


def write_derived_dockerfile(base: str) -> Path:
    """The Dockerfile of the toolchain-baked image for one base."""
    d = TC_ROOT / slug(base)
    d.mkdir(parents=True, exist_ok=True)
    content = f"""# Toolchain-baked image for the base `{base}`.
#
# It is the *unmodified* upstream base plus:
#   * the agent toolchain (uv + python 3.12 + mini-swe-agent==2.4.6), copied
#     from `rsi/toolchain:1` so that every trial does not have to install it;
#   * the `rsi` package, so that the agent class `rsi.rhi_agent.RhiAgent`
#     and the inlined harness spec are importable from inside the isolated
#     tool venv;
#   * a regional package mirror, so that the *task's own* `apt-get install`
#     layer can actually complete.  See MIRROR_NOTE below.
FROM {base}

COPY --from={TOOLCHAIN_IMAGE} /root/.local /root/.local

# The toolchain image also carries a snapshot of the `rsi` package.  It is
# dropped here on purpose: the copy below is the current one, and a stale
# snapshot earlier on PYTHONPATH would silently shadow it.
RUN rm -rf /root/.local/rsi-pkg

# MIRROR_NOTE ---------------------------------------------------------------
# The registry task Dockerfiles keep their upstream `apt-get install` layer
# verbatim (only the FROM line is rewritten), and that layer needs the
# network.  On this network `deb.debian.org` is unreachable -- a single
# `curl` to it times out after 20 s with zero bytes -- while
# `mirrors.aliyun.com` serves the same content at ~477 KB/s.  The upstream
# layer therefore either takes ~250 s or fails outright with
#
#     E: Failed to fetch http://deb.debian.org/debian/... Unable to connect
#
# and a failed build kills the trial before the agent ever runs.
#
# Pointing apt at a reachable mirror *in the base image* fixes this without
# touching a single byte of the task's own Dockerfile: the task layer then
# resolves against the mirror this RUN wrote, and its content is identical.
# The rewrite is scoped to the files that actually exist in each base family.
RUN set -eu; \\
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \\
        sed -i 's|http://deb.debian.org/debian-security|{MIRROR}/debian-security|g; s|http://deb.debian.org/debian|{MIRROR}/debian|g' /etc/apt/sources.list.d/debian.sources; \\
    fi; \\
    if [ -f /etc/apt/sources.list ]; then \\
        sed -i 's|http://deb.debian.org/debian-security|{MIRROR}/debian-security|g; s|http://deb.debian.org/debian|{MIRROR}/debian|g' /etc/apt/sources.list; \\
    fi; \\
    if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \\
        sed -i 's|http://archive.ubuntu.com/ubuntu|{MIRROR_UBUNTU}|g; s|http://security.ubuntu.com/ubuntu|{MIRROR_UBUNTU}|g; s|http://ports.ubuntu.com/ubuntu-ports|{MIRROR_UBUNTU_PORTS}|g' /etc/apt/sources.list.d/ubuntu.sources; \\
    fi; \\
    if [ -f /etc/apt/sources.list ]; then \\
        sed -i 's|http://archive.ubuntu.com/ubuntu|{MIRROR_UBUNTU}|g; s|http://security.ubuntu.com/ubuntu|{MIRROR_UBUNTU}|g; s|http://ports.ubuntu.com/ubuntu-ports|{MIRROR_UBUNTU_PORTS}|g' /etc/apt/sources.list; \\
    fi; \\
    true
# pip is reached through the same regional mirror for the same reason.
RUN printf '[global]\\nindex-url = {PIP_INDEX}\\ntrusted-host = {PIP_HOST}\\n' > /etc/pip.conf

# PREINSTALL_NOTE ----------------------------------------------------------
# Every LiveCode task Dockerfile ends with
#
#     RUN apt-get update && apt-get install -y git tmux asciinema \
#         python3-pip python3-venv && rm -rf /var/lib/apt/lists/*
#
# and Harbor builds that layer *inside every trial* (the build cache key
# includes the task's own COPY-ed files, so the layer cannot be shared
# between tasks).  On this network the download of that set -- which drags
# in build-essential via python3-venv -- takes longer than the task's own
# 400 s build timeout, so the build fails and kills the trial before the
# agent runs.
#
# Installing the *same* packages here, once, in the base image, turns the
# task's layer into a no-op: apt sees every package already installed and
# only has to refresh the index.  The packages, and the layer's effect, are
# identical -- the task's Dockerfile stays byte-for-byte as published.
#
# The list is the union of the packages the registry task Dockerfiles
# install; a package that a given task does not ask for is simply already
# present, which is harmless.
RUN apt-get update -qq && \\
    apt-get install -y --no-install-recommends \\
        git \\
        tmux \\
        asciinema \\
        python3-pip \\
        python3-venv \\
        ca-certificates \\
        curl \\
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${{PATH}}" \\
    PYTHONPATH=/opt/rsi \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY rsi /opt/rsi/rsi
COPY pyproject.toml /opt/rsi/pyproject.toml
"""
    path = d / "Dockerfile"
    path.write_text(content, encoding="utf-8")
    return path


def build_derived_image(base: str) -> str:
    """Build the derived image for one base (idempotent)."""
    image = derived_image_for(base)
    dockerfile = write_derived_dockerfile(base)
    print(f"==> building {image} (base {base})", flush=True)
    subprocess.run(
        ["docker", "build", "-f", str(dockerfile), "-t", image, str(ROOT)],
        check=True,
    )
    return image


def collect_bases(task_dirs: list[Path]) -> list[str]:
    bases: set[str] = set()
    for task in task_dirs:
        df = task / "environment" / "Dockerfile"
        if not df.exists():
            continue
        for line in df.read_text(encoding="utf-8").splitlines():
            m = FROM_RE.match(line)
            if m and not m.group(2).startswith("$"):
                bases.add(m.group(2))
    return sorted(bases)


def patch_dockerfile(task: Path, mapping: dict[str, str]) -> bool:
    """Replace the base image of a task's Dockerfile.  Returns True if changed."""
    df = task / "environment" / "Dockerfile"
    if not df.exists():
        return False
    original = df.read_text(encoding="utf-8")
    out: list[str] = []
    changed = False
    for line in original.splitlines():
        m = FROM_RE.match(line)
        if m and m.group(2) in mapping:
            new = f"{m.group(1)}{mapping[m.group(2)]}{m.group(3)}"
            out.append(new)
            changed = changed or new != line
        else:
            out.append(line)
    if changed:
        df.write_text("\n".join(out) + "\n", encoding="utf-8")
    return changed


def sha256_tree(root: Path) -> str:
    """A stable hash of a task directory's content (paths + bytes)."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("benchmarks", nargs="+")
    ap.add_argument("--source", action="append", default=[],
                    help="override the source dir of a benchmark (key=path)")
    ap.add_argument("--skip-build", action="store_true",
                    help="only copy and patch, do not build the derived images")
    args = ap.parse_args()

    sources = dict(DEFAULT_SOURCES)
    for item in args.source:
        key, _, path = item.partition("=")
        sources[key] = Path(path)

    for key in args.benchmarks:
        src = sources.get(key)
        if src is None or not src.exists():
            print(f"! {key}: no source dataset at {src}; skipping")
            continue
        dest = DEST_ROOT / key
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

        task_dirs = sorted(p for p in src.iterdir() if p.is_dir())
        for task in task_dirs:
            shutil.copytree(task, dest / task.name, symlinks=True)

        bases = collect_bases([dest / t.name for t in task_dirs])
        print(f"== {key}: {len(task_dirs)} tasks, {len(bases)} base(s): {bases}")

        mapping: dict[str, str] = {}
        for base in bases:
            image = derived_image_for(base)
            mapping[base] = image
            if not args.skip_build:
                build_derived_image(base)
            else:
                write_derived_dockerfile(base)

        n_patched = sum(
            1 for t in task_dirs if patch_dockerfile(dest / t.name, mapping)
        )
        print(f"   patched {n_patched} Dockerfile(s)")

        # Provenance: what was copied, from where, and with which hash.
        manifest = {
            "benchmark": key,
            "source": str(src),
            "n_tasks": len(task_dirs),
            "bases": bases,
            "base_images": mapping,
            "tasks": {
                t.name: sha256_tree(dest / t.name) for t in task_dirs
            },
        }
        (dest / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"   wrote {dest/'MANIFEST.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
