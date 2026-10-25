# The **IssueTasks** family base image.
#
# This family is NOT the plain ubuntu image.  The published IssueTasks
# rollouts ran on a **Debian** container with a **source-built python 3.10**
# in /usr/local, and the difference is visible in the rollouts themselves:
#
#   * ``pytest`` answers ``platform linux -- Python 3.10.20, ... --
#     /usr/local/bin/python3.10`` -- a python *compiled into /usr/local*,
#     which is what the official ``python:3.10`` image ships, not what
#     ubuntu's ``python3`` package does;
#   * ``apt-get update`` hits ``deb.debian.org`` and answers
#     ``Hit:1 http://deb.debian.org/debian trixie InRelease`` -- Debian 13
#     ("trixie"), not bookworm.  The SuperUser and Tezos rollouts hit
#     ``archive.ubuntu.com`` (7,895 and 5,307 mentions) instead.
#
# So the IssueTasks tasks were generated from a Debian-based image, and the
# tezos/superuser ones from ``ubuntu:24.04``.  They are two different
# environments that share a task family name; the rollout is what
# distinguishes them, and the rollout says Debian trixie.
#
# The base image is the official ``python:3.10-slim-trixie``:
#
#   * it is Debian trixie, with Debian's own apt sources;
#   * ``python3`` lands in ``/usr/local/bin`` -- the same place the
#     published rollouts found it;
#   * it is the *floating* ``3.10`` tag's current content: ``3.10-slim``
#     and ``3.10-slim-trixie`` share one digest on Docker Hub.
#
# The point release drifts, the same way the plain family's does: the
# published rollouts reported 3.10.20 and the tag resolves to a later
# 3.10.x today, so the ``pytest`` banner reads
# ``platform linux -- Python 3.10.22`` where the original read ``3.10.20``.
# Pinning the point release is not an option: the pull of
# ``python:3.10.20-slim-trixie`` fails from this network with a TLS
# handshake timeout against registry-1.docker.io, exactly like the pull of
# ``ubuntu:24.04.4`` does.  The minor version is the same, which is what the
# agent's behaviour depends on.
#
# On top of it this image adds what the harness needs and the original
# container did not have:
#
#   * git, curl, ca-certificates -- Harbor's ``ensure_system_dependencies()``
#     asks for them, so baking them in makes that check a no-op instead of
#     an ``apt-get`` round trip inside every trial;
#   * tmux and asciinema -- what the original terminal harness used.
#
# **build-essential IS installed, and the reason is the harness, not the
# environment.**  An earlier version of this file deliberately left the
# compiler out, on the grounds that the published rollouts show a minimal
# container.  The published rollouts were produced by ``terminus-2``, which
# does not require a compiler; the mini-swe-agent adapter under Harbor does.
# Harbor's ``ensure_system_dependencies()`` asks for ``("curl", "bash",
# "build_tools", "git")``, and ``build_tools`` is satisfied by ``gcc`` and
# ``make`` -- so Harbor runs ``apt-get install -y build-essential`` inside
# *every* trial when they are missing.
#
# That settles the fidelity question: under this harness the agent sees
# ``gcc`` and ``make`` **whether or not the image ships them**, because the
# install happens during setup, before the agent's first step.  Baking them
# in therefore changes nothing the agent observes, and it removes a per-trial
# ``apt-get`` round trip that on ``deb.debian.org`` (0.18 MB/s from this
# host) is slow enough to stop the trials from starting at all.
#
# The measurements back this up.  Across the published rollouts:
#
#   family     gcc missing   gcc version shown
#   superuser      0.00%            3.72%
#   tezos          0.01%            0.24%
#   swesmith       0.00%            0.12%
#   issue          3.55%            7.85%
#
# The plain families' containers already had a compiler; the issue family's
# are mixed, because the agent sometimes installed one itself.  Either way
# the harness normalises it.
#
# **node is deliberately not baked in.**  The IssueTasks rollouts show
# ``node --version`` answering ``v20.19.2`` in some of them and
# ``bash: node: command not found`` in others, so the published containers
# did not have node either -- the agent installed it when a task needed it.
# Debian's nodejs is 18.19.x, which is neither of the two, so baking it in
# would put a version in the environment that no published rollout ever saw.
FROM python:3.10-slim-trixie

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The apt options are not decoration: from this host the container talks to
# the Debian mirrors directly (the host itself goes through a proxy on
# 127.0.0.1, which a container cannot reach), and deb.debian.org answers at
# about 0.18 MB/s here -- slow enough that the install takes tens of
# minutes.  The build therefore downloads from a mirror that is reachable
# (mirrors.tuna.tsinghua.edu.cn, 0.74 MB/s) and then puts the sources back
# the way Debian ships them.
#
# One mirror, one URI: apt's multi-URI fallback does not actually work the
# way it reads -- with two URIs on the line it waits for both and the
# update stalls -- so the mirror has to be one that works.  The package set
# is small enough for that to be fine: git, curl, ca-certificates, tmux and
# asciinema, and no compiler.
#
# The restore is the point: the *published* rollouts show
# ``Hit:1 http://deb.debian.org/debian trixie InRelease``, and the agent's
# own ``apt-get update`` has to hit the same mirror the original container
# did, or the trajectory records a hostname no published rollout ever saw.
RUN set -eux; \
    cp /etc/apt/sources.list.d/debian.sources /tmp/debian.sources.orig; \
    sed -i 's|URIs: http://deb.debian.org|URIs: http://mirrors.tuna.tsinghua.edu.cn|g' \
        /etc/apt/sources.list.d/debian.sources; \
    apt-get -o Acquire::Retries=5 \
            -o Acquire::http::Timeout=20 \
            -o Acquire::https::Timeout=20 \
            update -qq; \
    apt-get install -y --no-install-recommends \
            git curl ca-certificates \
            build-essential \
            tmux asciinema; \
    rm -rf /var/lib/apt/lists/*; \
    cp /tmp/debian.sources.orig /etc/apt/sources.list.d/debian.sources; \
    rm -f /tmp/debian.sources.orig; \
    grep -q 'deb.debian.org' /etc/apt/sources.list.d/debian.sources; \
    ! grep -q 'tuna' /etc/apt/sources.list.d/debian.sources

# **opencode is baked in, so one image serves both harnesses.**  The two
# harnesses are the variable in this experiment and the environment is not:
# mini-swe-agent is not in the image -- it is the toolchain bundle that gets
# bind-mounted at /root/.local -- while opencode is a standalone binary that
# has to be installed.  So the image carries opencode, and the mini side
# mounts its toolchain on top.  The two halves of the split then run in the
# *same* environment, which is the point of the split.
#
# The standalone release is used rather than the npm package, so the build
# does not depend on the npm postinstall reaching GitHub -- and so the image
# does not need node, which this family deliberately does not ship (see the
# note about node above).
#
# The ARG is declared *here*, after the apt layer, on purpose: an ARG above
# the apt layer changes its cache key, and the apt mirror is the flakiest
# part of this build.
ARG OPENCODE_VERSION=1.18.33
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64) asset=opencode-linux-x64.tar.gz ;; \
      arm64) asset=opencode-linux-arm64.tar.gz ;; \
      *) echo "unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL --retry 8 --retry-delay 3 --retry-all-errors --retry-max-time 300 \
      "https://github.com/sst/opencode/releases/download/v${OPENCODE_VERSION}/${asset}" \
      -o /tmp/opencode.tar.gz; \
    tar -xzf /tmp/opencode.tar.gz -C /usr/local/bin opencode; \
    rm -f /tmp/opencode.tar.gz; \
    chmod 0755 /usr/local/bin/opencode; \
    opencode --version

# The original container's toolchain, so the agent's observations match the
# published ones.  node/npm are deliberately absent -- see above -- so they
# are not checked here either.  WORKDIR comes first because the original
# container's working directory is /app and it starts out empty: the agent
# creates its files there from scratch, exactly as the published rollouts
# show (``ls -la /app/`` answers with nothing but ``.`` and ``..``).
WORKDIR /app

RUN python3 --version && \
    command -v python3 && \
    git --version && \
    test -d /app
