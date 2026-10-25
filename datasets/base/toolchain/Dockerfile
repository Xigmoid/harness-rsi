# The agent toolchain as a *mountable bundle*.
#
# Harbor installs mini-swe-agent inside **every** task container:
#
#     curl https://astral.sh/uv/install.sh | sh
#     uv python install 3.12
#     uv tool install --python 3.12 mini-swe-agent==2.4.6 \
#         --with litellm --with orjson --with fastapi
#
# That is ~4 minutes of wall clock and ~1 GB of downloads *per trial*, plus
# an `apt-get install build-essential` before it.  On the registry tasks that
# is not a cost, it is a wall: a 4-task AIME calibration finished with
# `AgentSetupTimeoutError` on 4/4 trials -- the agent never ran once.
#
# The fix is to build the toolchain **once**, in a container, and bind-mount
# it into every trial at the path uv expects (`/root/.local`).  The venv's
# shebangs and pyvenv.cfg are absolute, so the bundle has to be mounted at
# the *same* path it was built at -- hence `/root/.local` on both sides.
#
# The image is built for linux/arm64, the architecture the trials run on.
#
# ``--with mcp`` is not optional for the tool-call protocol: litellm imports
# its MCP handler as soon as ``tools`` is set, and the import is
# unconditional, so without the package every request dies with
# ``ModuleNotFoundError: No module named 'mcp'``.  That is why the framework
# used to pin the *text* protocol, where no ``tools`` are sent and the import
# never happens.  With the package in the bundle the tool-call protocol works
# inside Harbor too, and the trajectory carries native OpenAI tool calls
# instead of a fenced code block the model has to be taught to parse.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

# uv's defaults put the tool dir, the python install dir and the shims under
# $HOME/.local, which is exactly the directory that gets bind-mounted.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && export PATH="/root/.local/bin:${PATH}" \
 && uv python install 3.12 \
 && uv tool install --python 3.12 mini-swe-agent==2.4.6 \
      --with litellm --with orjson --with fastapi --with pyyaml \
      --with mcp \
 && mini-swe-agent --help

# The rsi package has to be importable from inside the isolated tool venv:
# the agent class is `rsi.rhi_agent.RhiAgent` and the spec is inlined into
# the mini-swe-agent config.  It travels with the bundle so that *any* task
# image can run the harness, not only the ones we baked.
COPY rsi /root/.local/rsi-pkg/rsi
COPY pyproject.toml /root/.local/rsi-pkg/pyproject.toml
