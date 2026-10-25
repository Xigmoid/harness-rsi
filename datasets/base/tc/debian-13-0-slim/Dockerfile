# Toolchain-baked image for the base `debian:13.0-slim`.
#
# It is the *unmodified* upstream base plus:
#   * the agent toolchain (uv + python 3.12 + mini-swe-agent==2.4.6), copied
#     from `rsi/toolchain:1` so that every trial does not have to install it;
#   * the `rsi` package, so that the agent class `rsi.rhi_agent.RhiAgent`
#     and the inlined harness spec are importable from inside the isolated
#     tool venv.
FROM debian:13.0-slim

COPY --from=rsi/toolchain:1 /root/.local /root/.local

ENV PATH="/root/.local/bin:${PATH}" \
    PYTHONPATH=/opt/rsi \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY rsi /opt/rsi/rsi
COPY pyproject.toml /opt/rsi/pyproject.toml
