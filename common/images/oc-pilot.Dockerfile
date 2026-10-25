# Pilot base image for the opencode re-annotation.
#
# It starts from the project's own toolchain-baked base
# (``rsi/tc-python-3-13-slim-bookworm:1``), which already carries
#
#   * the regional apt/pip mirrors -- ``deb.debian.org`` is unreachable on
#     this network and ``mirrors.aliyun.com`` serves the same content, so a
#     task's own ``apt-get install`` layer only works because the base image
#     rewrites the sources;
#   * git, tmux, asciinema, python3-pip, curl, ca-certificates;
#   * the mini-swe-agent toolchain and the ``rsi`` package (not needed by
#     opencode, but harmless).
#
# On top of that it adds what the opencode harness needs:
#
#   * nodejs + npm -- Harbor's ``ensure_system_dependencies()`` asks for
#     both.  They are installed here so that the check inside the trial is a
#     no-op instead of an ``apt-get`` round trip;
#   * opencode itself, pinned to the version the job file pins.  Baking it
#     in means the per-trial ``npm i -g opencode-ai`` never runs, which is
#     what ``RsiOpenCode.install()``'s fast path looks for.
#
# The version lives in *both* places on purpose: the image and the job file.
FROM rsi/tc-python-3-13-slim-bookworm:1

ARG OPENCODE_VERSION=1.18.33

# nodejs + npm for Harbor's dependency check and for the npm fallback path.
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends nodejs npm && \
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
    curl -fsSL "https://github.com/sst/opencode/releases/download/v${OPENCODE_VERSION}/${asset}" \
      -o /tmp/opencode.tar.gz; \
    tar -xzf /tmp/opencode.tar.gz -C /usr/local/bin opencode; \
    rm -f /tmp/opencode.tar.gz; \
    chmod 0755 /usr/local/bin/opencode; \
    opencode --version; \
    command -v node; command -v npm

WORKDIR /app
