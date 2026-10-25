# Base image for the Terminal-Bench family of Harbor tasks (AIME, GPQA, ...).
#
# It exists for one reason: **Harbor installs mini-swe-agent inside every
# task container**.  The install is `curl uv + uv python install 3.12 + uv
# tool install mini-swe-agent==2.4.6 --with litellm --with orjson
# --with fastapi`, which costs ~4 minutes of wall clock and needs ~1 GB of
# downloads *per trial*.  On the registry tasks that is not a cost, it is a
# wall: a 4-task AIME calibration timed out with
# `AgentSetupTimeoutError` on 4/4 trials before the agent had even started.
#
# Baking the toolchain into the image turns the install into a no-op:
#   * `ensure_system_dependencies()` finds curl/git/gcc/make and skips apt;
#   * `uv tool install` sees the pinned version already present and returns
#     without touching the network.
# The version is pinned in *both* places on purpose -- the image and the job
# file -- so the toolchain can never drift.
#
# The `rsi` package is baked in as well: the agent class
# `rsi.harbor_agent:RsiMiniSweAgent` has to be importable from inside the
# isolated `uv tool` venv that Harbor creates for mini-swe-agent.
FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/opt/rsi \
    PATH="/root/.local/bin:${PATH}"

# Exactly the packages Harbor's ensure_system_dependencies() would install.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

# The agent toolchain, pinned to the same version the job file pins.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && export PATH="/root/.local/bin:${PATH}" \
 && uv python install 3.12 \
 && uv tool install --python 3.12 mini-swe-agent==2.4.6 \
      --with litellm --with orjson --with fastapi \
 && mini-swe-agent --help

WORKDIR /app
COPY rsi /opt/rsi/rsi
COPY pyproject.toml /opt/rsi/pyproject.toml
