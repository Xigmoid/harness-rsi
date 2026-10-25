# The SWE-Smith family base image for the **Go** repositories.
#
# The published SWE-Smith rollouts did not all run in the same kind of
# image.  The Python repositories ran in an ubuntu 22.04 image with a conda
# environment named ``testbed``; the Go repositories ran in a **debian 12
# (bookworm)** image, and the published rollouts say so::
#
#     root@80455e96-...:/testbed# apt-get update && apt-get install -y golang-go
#     Hit:1 http://deb.debian.org/debian bookworm InRelease
#     Get:2 http://deb.debian.org/debian-security bookworm-security InRelease [48.0 kB]
#
# and they had **Go installed but not on PATH**: the same rollout answers
# ``bash: go: command not found`` and then finds ``/usr/local/go/bin/go``
# with a ``find``.  The agent had to ``export PATH=/usr/local/go/bin:$PATH``
# itself.
#
# Both details are reproduced here, because they are what the agent
# interacts with: a Go toolchain at /usr/local/go that is *not* on PATH, so
# the rollout starts the same way -- ``go: command not found`` -- and the
# agent has to discover the toolchain the same way.
#
# The Go version is 1.24.5, the version the published rollouts found at
# /usr/local/go.  It is the arm64 build, because this host is arm64: the
# original was linux/amd64, so ``go version`` prints a different platform.
#
# The apt sources use a regional mirror.  ``deb.debian.org`` does not answer
# from this network, and an agent that cannot install anything produces a
# materially different -- and much worse -- trajectory than the published
# one, which installed Go successfully.
FROM debian:12-slim

ARG OPENCODE_VERSION=1.18.33
ARG GO_VERSION=1.24.5

ENV DEBIAN_FRONTEND=noninteractive

# git comes with the SWE-Smith image; curl is what the setup script uses.
#
# tmux and asciinema are what the original terminal harness used.
RUN set -eux; \
    # The apt sources are left exactly as debian ships them.  The published
    # rollouts reached "http://deb.debian.org/debian bookworm" over plain
    # http, so this matches the original environment as well as it works --
    # and it does work: deb.debian.org answers from a container on this
    # network.
    apt-get update -qq; \
    apt-get install -y --no-install-recommends \
        ca-certificates git curl \
        tmux asciinema; \
    rm -rf /var/lib/apt/lists/*

# Go, at the path the published images used -- and deliberately *not* on
# PATH, exactly like the original.
# Go, at the path the published images used -- and deliberately *not* on
# PATH, exactly like the original: the published rollouts answered
# "go: command not found" until the agent added /usr/local/go/bin to PATH
# itself.
#
# The comment is above the RUN on purpose: a comment *inside* a
# backslash-continued RUN is joined into the shell command, where the "#"
# swallows the rest of the line -- including whatever follows.
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    case "$arch" in \
      amd64) goarch=amd64 ;; \
      arm64) goarch=arm64 ;; \
      *) echo "unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL --retry 8 --retry-delay 3 --retry-all-errors --retry-max-time 300 \
      "https://go.dev/dl/go${GO_VERSION}.linux-${goarch}.tar.gz" -o /tmp/go.tar.gz; \
    rm -rf /usr/local/go; \
    tar -C /usr/local -xzf /tmp/go.tar.gz; \
    rm -f /tmp/go.tar.gz; \
    ls -l /usr/local/go/bin/go

# opencode: the standalone binary from the release, not the npm package.
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

# The repository lands in /testbed at runtime; the directory has to exist
# before that, and /logs is where Harbor collects the trial's artefacts.
RUN mkdir -p /testbed /logs

WORKDIR /testbed
