# The SWE-Smith family base image for the **Python** repositories.
#
# This is the *family* image: every SWE-Smith task whose repository is a
# Python project starts from this exact image, and the per-task difference
# (which repository, at which bug-injected commit) is applied at **runtime**
# by the task's own setup script.  Nothing is built per task.
#
# Why ubuntu 22.04 and not 24.04
# ------------------------------
#
# The published SWE-Smith rollouts ran in the official SWE-Smith images, and
# those images are **ubuntu 22.04 (jammy)** -- the published rollouts say so
# themselves::
#
#     root@8dc15b15-...:/testbed/reproduce# apt-get update && apt-get install -y cmake
#     Hit:1 http://archive.ubuntu.com/ubuntu jammy InRelease
#     Get:2 http://archive.ubuntu.com/ubuntu jammy-updates InRelease [128 kB]
#     Hit:3 http://security.ubuntu.com/ubuntu jammy-security InRelease
#
# (The three plain families are a different story: their rollouts reach
# ``archive.ubuntu.com/ubuntu noble``, i.e. 24.04.  That is why there are two
# "plain" images and not one.)
#
# The rest of the environment, taken from the published rollouts
# --------------------------------------------------------------
#
#   * the prompt reads ``(testbed) root@...:/testbed#`` -- an environment
#     named ``testbed`` is active;
#   * ``python --version`` answers 3.10.16 (3,868 of the published rollouts
#     that printed it), 3.10.18 (1,217), 3.11.11 (139) or 3.11.13 (76), so
#     both 3.10 and 3.11 are installed and the setup script picks the
#     version the task's own rollout used;
#   * the repository sits at ``/testbed`` **as the repository root**: the
#     original ``ls -la`` shows ``pypika/``, ``setup.py`` and
#     ``PyPika.egg-info`` side by side.
#
# The upstream generator agrees -- ``data/swesmith/task_templates.py`` in
# open-thoughts/OpenThoughts-Agent renders the task Dockerfile as
#
#     FROM {official image}
#     RUN git fetch && git checkout {instance_id}
#
# The official image is amd64-only and 1.37 GB, so emulating it on this
# arm64 host makes every agent command slow; this image reconstructs its
# *contents* natively and the setup script does the checkout.
#
# The conda environment is reproduced as a **venv at
# /opt/conda/envs/testbed** rather than /opt/testbed, on purpose: the shell
# prompt then reads ``(testbed)`` exactly like the original, and any command
# that reaches for ``/opt/conda/envs/testbed/bin/python`` -- the path the
# original image exposes -- keeps working.
FROM ubuntu:22.04

ARG OPENCODE_VERSION=1.18.33

ENV DEBIAN_FRONTEND=noninteractive

# python3 + python3-pip are what the upstream plain-task image installs;
# git comes with the SWE-Smith image; curl is what the upstream setup
# script uses to fetch uv.
#
# build-essential is not optional: the editable install of a repository
# that builds a C extension (pandas is built with meson) needs a compiler,
# and the official SWE-Smith image has one -- it has to, the image build
# runs the very same ``pip install -e .`` this setup script runs.  Without
# it the install dies with "Unknown compiler(s): [['cc'], ['gcc'], ...]".
#
# tmux and asciinema are what the original terminal harness used.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip \
        git curl ca-certificates \
        tmux asciinema \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# opencode: the standalone binary from the release, not the npm package, so
# the build does not depend on the npm postinstall reaching GitHub.
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64) asset=opencode-linux-x64.tar.gz ;; \
      arm64) asset=opencode-linux-arm64.tar.gz ;; \
      *) echo "unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    # --retry-all-errors: the release download from GitHub is not reliable on
    # this network (a plain curl fails with exit 56, "recv failure"), and the
    # build has to survive that.
    curl -fsSL --retry 8 --retry-delay 3 --retry-all-errors --retry-max-time 300 \
      "https://github.com/sst/opencode/releases/download/v${OPENCODE_VERSION}/${asset}" \
      -o /tmp/opencode.tar.gz; \
    tar -xzf /tmp/opencode.tar.gz -C /usr/local/bin opencode; \
    rm -f /tmp/opencode.tar.gz; \
    chmod 0755 /usr/local/bin/opencode; \
    opencode --version; \
    python3 --version

# uv 0.7.13 is the version the upstream task templates pin.
#
# ``curl ... | sh`` hides a failed download: curl's error goes to stderr, sh
# reads an empty script and exits 0, so the layer "succeeds" without uv in
# it and the next layer dies with "uv: command not found" (exit 127).  That
# is exactly what happened on the first build of this image on this network,
# so the download is retried and the pipe is checked.
ENV UV_INSTALL_DIR=/usr/local/bin
RUN set -eux; \
    curl -fsSL --retry 8 --retry-delay 3 --retry-all-errors --retry-max-time 300 \
      https://astral.sh/uv/0.7.13/install.sh -o /tmp/uv-install.sh; \
    sh /tmp/uv-install.sh; \
    rm -f /tmp/uv-install.sh; \
    command -v uv; uv --version

# Every python version the published rollouts used is installed here, so
# the setup script can pick the right one without downloading anything at
# trial time.  The versions are the ones the published rollouts answered
# ``python --version`` with: 3.10.16 (3,868 rows), 3.10.18 (1,217),
# 3.11.11 (139) and 3.11.13 (76).
RUN uv python install 3.10.16 3.10.18 3.11.11 3.11.13

# The environment named "testbed".  --seed puts pip inside the venv:
# without it ``pip`` resolves to the *system* pip, which refuses to install
# anything ("externally managed environment", PEP 668) -- the upstream conda
# environment has no such rule, so an agent that runs ``pip install pytest``
# there succeeds.
RUN uv venv --python 3.10.16 --seed /opt/conda/envs/testbed

ENV VIRTUAL_ENV=/opt/conda/envs/testbed
ENV PATH="/opt/conda/envs/testbed/bin:$PATH"

# The repository lands in /testbed at runtime; the directory has to exist
# before that, and /logs is where Harbor collects the trial's artefacts.
RUN mkdir -p /testbed /logs

WORKDIR /testbed

# uv's download cache is shared between trials through a bind mount, so the
# second task on a repository does not re-download the wheels.
ENV UV_CACHE_DIR=/var/cache/uv
RUN mkdir -p /var/cache/uv
