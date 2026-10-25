# The faithful base image for the opencode re-annotation.
#
# The published dataset was annotated in an **ubuntu:24.04** container, and
# that is not a detail: the original rollouts show it.
#
#   * ``cat /etc/os-release`` in the original tezos rollout answers
#     ``PRETTY_NAME="Ubuntu 24.04.3 LTS"``;
#   * the original superuser rollout ran ``apt-get update`` and got
#     ``Hit:1 http://archive.ubuntu.com/ubuntu noble InRelease`` -- the
#     *default* Ubuntu sources, not a regional mirror;
#   * ``python3 --version`` answers 3.12.3, which is the python of noble.
#
# The upstream generator confirms it: ``data/swesmith/task_templates.py`` in
# open-thoughts/OpenThoughts-Agent builds the non-repository tasks as
#
#     FROM ubuntu:24.04
#     WORKDIR /app
#     RUN apt-get update && apt-get install -y python3 python3-pip && ...
#
# and the repository tasks on top of the official SWE-smith image, which is
# the same ubuntu plus a conda environment named ``testbed`` with python
# 3.10 and the package installed editable.
#
# So this image is that base, plus what the opencode *harness* needs and the
# original container did not have:
#
#   * nodejs + npm -- Harbor's ``ensure_system_dependencies()`` asks for
#     both.  Installing them here makes that check inside the trial a no-op
#     instead of an ``apt-get`` round trip;
#   * the opencode binary, pinned to the version the job file pins.  Baking
#     it in means the per-trial ``npm i -g opencode-ai`` never runs, which is
#     what ``RsiOpenCode.install()``'s fast path looks for.
#
# The version lives in *both* places on purpose: the image and the job file.
#
# The apt sources are left exactly as ubuntu ships them: the original agent
# installed GNU parallel itself, and that only works if ``apt-get update``
# reaches the same archive the original reached.
#
# The floating tag is what the upstream generator writes, and the published
# dataset spans two point releases of the same LTS: of the 1,474 rollouts
# that printed /etc/os-release, 1,298 answer "Ubuntu 24.04.4 LTS" and 151
# answer "24.04.3".  The tag resolves to 24.04.5 today, so the patch level
# in /etc/os-release drifts by one.  Pinning a point release is not an
# option either: Docker Hub retires those tags, and the pull of
# ubuntu:24.04.4 fails from this network with a TLS handshake timeout.
FROM ubuntu:24.04

ARG OPENCODE_VERSION=1.18.33

ENV DEBIAN_FRONTEND=noninteractive

# python3 + python3-pip are what the upstream plain-task image installs;
# git, curl and ca-certificates come with the SWE-smith image and with the
# harness.  tmux and asciinema are what the original terminal harness used.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip \
        git curl ca-certificates \
        tmux asciinema \
        nodejs npm && \
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
    command -v node; command -v npm; \
    python3 --version

WORKDIR /app
