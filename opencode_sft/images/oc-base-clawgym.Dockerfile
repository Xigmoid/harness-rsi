# The **ClawGym** base image: one environment for every ClawGym task.
#
# The ClawGym tasks come with their own environment, so this image is not a
# reconstruction of some published rollout -- there is no rollout to
# reconstruct, because ``RUC-AIBOX/ClawGym-Task`` ships *tasks*, not
# trajectories.  What it ships is a contract, and this image implements it.
#
# The contract, read off all 15,402 tasks:
#
#   * every task's ``hook_code`` is the same line --
#     ``python3 reward/test.py /root/.openclaw/workspace`` -- so the verifier
#     lives at ``reward/test.py`` **relative to the working directory**, and
#     the working directory is ``/root/.openclaw/workspace``;
#   * the verifier's first argument is the workspace root, and it reads the
#     task's inputs from ``<workspace>/input/`` and writes its checks against
#     ``<workspace>/output/`` (or, for the second task family, against files
#     directly under the workspace);
#   * the tasks' ``input_files`` are paths like ``input/products.csv`` and
#     ``reward/test.py`` -- i.e. *relative to the workspace*.
#
# So the layout the image has to provide is:
#
#     /root/.openclaw/workspace/
#         input/          <- per-task input files
#         output/         <- where the agent writes its deliverables
#         reward/test.py  <- the verifier
#
# Harbor supplies the per-task part: for a task that names a pre-built
# ``docker_image`` and has no ``environment/Dockerfile``, it uploads the task's
# ``environment/`` directory into ``[environment].workdir`` before the agent
# starts.  The registry therefore stores each task's ``input/`` and
# ``reward/`` under ``environment/``, and the image only has to own the
# *skeleton* -- the empty directories, the interpreter and the tools.
#
# **Python 3.12 is not a preference.**  Six of the 15,402 verifiers contain
# an f-string with a backslash inside the expression (``f"{p.replace('\\',
# '/')}"``).  That is a syntax error before Python 3.12 and legal from 3.12
# on (PEP 701).  Probing every verifier against an empty workspace found
# exactly those six, and they run clean on 3.12.  Ubuntu 24.04 ships
# Python 3.12.3, so this image does not have to build one.
#
# **PyYAML is required.**  101 verifiers ``import yaml``; without it they
# raise at import time and the task cannot be scored at all.
#
# **build-essential is baked in for the same reason as in the plain image:**
# Harbor's mini adapter asks for ``("curl", "bash", "build_tools", "git")``
# and installs ``build-essential`` inside every trial when ``gcc``/``make``
# are missing.  Installing it at image build time costs one download and
# saves one ``apt-get`` round trip per trial.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# The apt sources are restored to the way ubuntu ships them: the agent's own
# ``apt-get update`` has to reach the same archive the original container
# did, or the trajectory records a hostname no ClawGym rollout ever saw.  The
# build itself downloads from a faster mirror -- from this host the archive
# is slow enough that a per-trial install never finishes.
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

# PyYAML: 101 of the verifiers import it.
RUN set -eux; \
    pip3 install --break-system-packages --no-cache-dir "PyYAML==6.0.2"; \
    python3 -c "import yaml; print('pyyaml', yaml.__version__)"

# **opencode is baked in, so one image serves both harnesses.**  The two
# harnesses are the variable in this experiment, and the environment is not:
# mini-swe-agent is *not* in the image -- it is the toolchain bundle that
# gets bind-mounted at /root/.local -- while opencode is a standalone binary
# that has to be installed.  So the image carries opencode, and the mini
# side mounts its toolchain on top.  The two halves of the split then run in
# the *same* environment, which is the point of the split.
#
# The version is the one the job file pins (1.18.33), and the install is
# the standalone release rather than the npm package, so the build does not
# depend on the postinstall reaching GitHub.
#
# The ARG is declared *here*, after the apt and PyYAML layers, on purpose: an
# ARG that sits above them changes their cache key, so declaring it earlier
# forces the whole apt layer to download again -- and the apt mirror is the
# flakiest part of this build.
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

# The workspace skeleton.  The directories are empty on purpose: Harbor
# uploads the task's ``environment/`` directory (which carries ``input/`` and
# ``reward/``) into this directory, and anything the image put there would
# either be shadowed by the upload or leak into the next task, because one
# image serves every task.
RUN set -eux; \
    mkdir -p /root/.openclaw/workspace/input \
             /root/.openclaw/workspace/output \
             /root/.openclaw/workspace/reward; \
    test -d /root/.openclaw/workspace/input; \
    test -d /root/.openclaw/workspace/output; \
    test -d /root/.openclaw/workspace/reward

# The working directory is the workspace, because the hook is written as
# ``python3 reward/test.py /root/.openclaw/workspace`` -- a *relative* path.
WORKDIR /root/.openclaw/workspace

RUN set -eux; \
    python3 --version; \
    python3 -c "import yaml"; \
    command -v gcc; command -v make; \
    command -v git; command -v curl; \
    command -v node; command -v npm; \
    test "$(pwd)" = "/root/.openclaw/workspace"
