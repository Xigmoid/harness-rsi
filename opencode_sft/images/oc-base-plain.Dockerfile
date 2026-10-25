# The **plain** family base image, shared by the three families whose
# environment is an empty /app on ubuntu 24.04: IssueTasks, Tezos and
# SuperUser.
#
# The published dataset's environment is not a detail:
#
#   * the original tezos rollout answers
#     ``PRETTY_NAME="Ubuntu 24.04.3 LTS"`` for ``cat /etc/os-release``;
#   * the original superuser rollout ran ``apt-get update`` and got
#     ``Hit:1 http://archive.ubuntu.com/ubuntu noble InRelease`` -- the
#     *default* Ubuntu sources, not a regional mirror;
#   * ``python3 --version`` answers 3.12.3, the python of noble;
#   * ``timedatectl`` and ``systemctl`` are *not found* -- this is a plain
#     container, not a systemd one.
#
# The upstream generator confirms it: ``data/swesmith/task_templates.py`` in
# open-thoughts/OpenThoughts-Agent builds the non-repository tasks as
#
#     FROM ubuntu:24.04
#     WORKDIR /app
#     RUN apt-get update && apt-get install -y python3 python3-pip
#
# and the per-task setup is nothing at all: /app starts empty.
#
# On top of that base this image adds what the opencode *harness* needs and
# the original container did not have:
#
#   * nodejs + npm -- Harbor's ``ensure_system_dependencies()`` asks for
#     both, so baking them in makes that check a no-op instead of an
#     ``apt-get`` round trip inside every trial;
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
# git, curl and ca-certificates come with the SWE-Smith image and with the
# harness.  tmux and asciinema are what the original terminal harness used.
#
# The apt sources are left exactly as ubuntu ships them: the original agent
# installed GNU parallel itself, and that only works if ``apt-get update``
# reaches the same archive the original reached.
#
# **build-essential is not optional.**  Harbor's mini-swe-agent adapter asks
# for the system dependencies ``("curl", "bash", "build_tools", "git")``, and
# ``build_tools`` is satisfied by ``gcc`` and ``make``.  When they are
# missing, Harbor runs ``apt-get update && apt-get install -y
# build-essential`` **inside every trial** -- at the speed of the distro
# archive, that is minutes of wall clock per trial before the agent has even
# started, and on the Debian-based image it is so slow that the trials never
# get going at all.
#
# The published rollouts agree: only about 4% of them mention
# ``build-essential``, i.e. the original containers already had a compiler
# and the agent rarely had to install one.  So the compiler belongs in the
# image, not in a per-trial install.
#
# The build downloads from a faster mirror and then puts the sources back
# the way ubuntu ships them: the agent's own ``apt-get update`` has to hit
# the same archive the original container did, or the trajectory records a
# hostname no published rollout ever saw.
RUN set -eux; \
    cp /etc/apt/sources.list.d/ubuntu.sources /tmp/ubuntu.sources.orig; \
    sed -i 's|http://ports.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' \
        /etc/apt/sources.list.d/ubuntu.sources; \
    apt-get -o Acquire::Retries=5 \
            -o Acquire::http::Timeout=20 \
            -o Acquire::https::Timeout=20 \
            update -qq; \
    apt-get install -y --no-install-recommends \
        python3 python3-pip \
        git curl ca-certificates \
        build-essential \
        tmux asciinema \
        nodejs npm; \
    rm -rf /var/lib/apt/lists/*; \
    cp /tmp/ubuntu.sources.orig /etc/apt/sources.list.d/ubuntu.sources; \
    rm -f /tmp/ubuntu.sources.orig; \
    grep -q 'ports.ubuntu.com' /etc/apt/sources.list.d/ubuntu.sources; \
    ! grep -q 'tuna' /etc/apt/sources.list.d/ubuntu.sources

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
