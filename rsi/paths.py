"""Well-known paths of the project.

Kept in a module of its own so that the host-side scripts (which build the
job templates) and the container-side agent adapter agree on the toolchain
bundle without importing Harbor.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "TOOLCHAIN_BUNDLE_DIR",
    "TOOLCHAIN_MOUNT_TARGET",
    "toolchain_available",
    "harbor_scratch_dir",
]

#: The project root (the directory that holds ``rsi/``, ``harness/``, ...).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: The host directory that holds the pre-built agent toolchain.  It is
#: bind-mounted into every task container at ``TOOLCHAIN_MOUNT_TARGET``.
#:
#: The bundle is built once from ``datasets/base/toolchain/Dockerfile`` and
#: exported with ``docker cp``; see ``scripts/build_toolchain.sh``.
TOOLCHAIN_BUNDLE_DIR = PROJECT_ROOT / ".cache" / "rsi-local"

#: Where the bundle is mounted inside the task container.  It is uv's default
#: location, so the venv's absolute shebangs keep resolving.
TOOLCHAIN_MOUNT_TARGET = "/root/.local"


def toolchain_available() -> bool:
    """True when the toolchain bundle has been built and exported.

    The check uses ``lexists`` on purpose: ``bin/mini-swe-agent`` is a
    *symlink* to an absolute path inside the container (``/root/.local/...``),
    so following it on the host would always fail.
    """
    return os.path.lexists(TOOLCHAIN_BUNDLE_DIR / "bin" / "mini-swe-agent")


def harbor_scratch_dir() -> Path:
    """The directory Harbor's trial directories have to live in.

    Harbor bind-mounts every trial's log directory into the task container,
    so the job output has to sit on a path that the container runtime can
    mount.  On macOS that rules out two obvious choices:

    * the project directory -- it hangs the mount (Docker Desktop's file
      sharing stalls on it);
    * ``/tmp`` and ``$TMPDIR`` -- both are symlinks into ``/private/...``,
      and the resolved path is not exposed to the VM, so the mount fails
      with *"bind source path does not exist"*.

    The literal ``/private/tmp`` path is the one that works, so the trials
    are written there and copied back into the project afterwards.
    """
    if sys.platform == "darwin":
        return Path("/private/tmp/rsi-harbor")
    return Path(tempfile.gettempdir()) / "rsi-harbor"
