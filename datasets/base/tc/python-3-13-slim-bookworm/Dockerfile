# Toolchain-baked image for the base `python:3.13-slim-bookworm`.
#
# It is the *unmodified* upstream base plus:
#   * the agent toolchain (uv + python 3.12 + mini-swe-agent==2.4.6), copied
#     from `rsi/toolchain:1` so that every trial does not have to install it;
#   * the `rsi` package, so that the agent class `rsi.rhi_agent.RhiAgent`
#     and the inlined harness spec are importable from inside the isolated
#     tool venv;
#   * a regional package mirror, so that the *task's own* `apt-get install`
#     layer can actually complete.  See MIRROR_NOTE below.
FROM python:3.13-slim-bookworm

COPY --from=rsi/toolchain:1 /root/.local /root/.local

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
RUN set -eu; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|http://deb.debian.org/debian-security|http://mirrors.aliyun.com/debian-security|g; s|http://deb.debian.org/debian|http://mirrors.aliyun.com/debian|g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http://deb.debian.org/debian-security|http://mirrors.aliyun.com/debian-security|g; s|http://deb.debian.org/debian|http://mirrors.aliyun.com/debian|g' /etc/apt/sources.list; \
    fi; \
    if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g; s|http://ports.ubuntu.com/ubuntu-ports|http://mirrors.aliyun.com/ubuntu-ports|g' /etc/apt/sources.list.d/ubuntu.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g; s|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g; s|http://ports.ubuntu.com/ubuntu-ports|http://mirrors.aliyun.com/ubuntu-ports|g' /etc/apt/sources.list; \
    fi; \
    true
# pip is reached through the same regional mirror for the same reason.
RUN printf '[global]\nindex-url = https://mirrors.aliyun.com/pypi/simple/\ntrusted-host = mirrors.aliyun.com\n' > /etc/pip.conf

# PREINSTALL_NOTE ----------------------------------------------------------
# Every LiveCode task Dockerfile ends with
#
#     RUN apt-get update && apt-get install -y git tmux asciinema #         python3-pip python3-venv && rm -rf /var/lib/apt/lists/*
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
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        git \
        tmux \
        asciinema \
        python3-pip \
        python3-venv \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}" \
    PYTHONPATH=/opt/rsi \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY rsi /opt/rsi/rsi
COPY pyproject.toml /opt/rsi/pyproject.toml
