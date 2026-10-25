# Base image for the IFEval task set.
#
# It carries
#   * the verifier dependencies (pytest + the CTRF reporter, langdetect and
#     nltk with the punkt models), so the verifier is deterministic and does
#     not need the network at verification time, and
#   * the `rsi` package in /opt/rsi, exposed through PYTHONPATH, so that the
#     agent class rsi.harbor_agent.RsiMiniSweAgent can be imported inside the
#     isolated `uv tool` venv that Harbor creates for mini-swe-agent.
#
# It also **pre-installs the agent toolchain**.  Harbor installs
# mini-swe-agent inside every task container, which costs ~4 minutes per
# trial (uv + a Python 3.12 + litellm and friends) and dominates the
# wall-clock of a 541-task benchmark.  Baking the toolchain into the image
# turns that into a no-op check, because the pinned version is already
# there.  The version is pinned in *both* places on purpose: the image and
# the job file.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/opt/rsi \
    DEBIAN_FRONTEND=noninteractive

# Harbor's ensure_system_dependencies() installs exactly these.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      pytest==8.4.1 \
      pytest-json-ctrf==0.3.5 \
      langdetect==1.0.9 \
      nltk==3.10.3 \
      pyyaml==6.0.3 \
      numpy==2.4.6

RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt punkt_tab

# The agent toolchain, pinned to the same version the job file pins.
ENV PATH="/root/.local/bin:${PATH}"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && export PATH="/root/.local/bin:${PATH}" \
 && uv python install 3.12 \
 && uv tool install --python 3.12 mini-swe-agent==2.4.6 \
      --with litellm --with orjson --with fastapi

WORKDIR /app
COPY rsi /opt/rsi/rsi
COPY pyproject.toml /opt/rsi/pyproject.toml
