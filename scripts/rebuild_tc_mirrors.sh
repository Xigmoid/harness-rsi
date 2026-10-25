#!/usr/bin/env bash
# Rebuild the toolchain-baked base images with a reachable package mirror.
#
# The registry task Dockerfiles keep their upstream `apt-get install` layer
# verbatim, and that layer needs the network.  `deb.debian.org` is
# unreachable from this network, so the layer either stalls for ~250 s or
# fails with "Unable to connect" and kills the trial before the agent runs.
# The derived images point apt at mirrors.aliyun.com; the task's own
# Dockerfile is untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

for base in python:3.11-slim python:3.13-slim-bookworm \
            ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624; do
    slug=$(echo "$base" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g')
    echo "==> rsi/tc-$slug:1  (base $base)"
    $PY - "$base" <<'PYEOF'
import sys
sys.path.insert(0, "scripts")
from prepare_registry_tasks import write_derived_dockerfile
print(write_derived_dockerfile(sys.argv[1]))
PYEOF
    docker build -f "datasets/base/tc/$slug/Dockerfile" -t "rsi/tc-$slug:1" .
done
echo "==> done"
docker images | grep 'rsi/tc-'
