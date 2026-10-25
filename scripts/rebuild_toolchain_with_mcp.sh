#!/usr/bin/env bash
# Rebuild the toolchain bundle with the `mcp` extra and swap it in safely.
#
# The old bundle stays usable until the new one is verified: the new bundle
# is exported to a side directory, `mcp` is imported from inside it, and only
# then does it replace the live one.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="rsi/toolchain:2"
OUT="$ROOT/.cache/rsi-local"
STAGE="$ROOT/.cache/rsi-local.new"

echo "==> building $IMAGE (this is the slow part)"
docker build -f "$ROOT/datasets/base/toolchain/Dockerfile" -t "$IMAGE" "$ROOT"

echo "==> exporting /root/.local -> $STAGE"
rm -rf "$STAGE"
mkdir -p "$(dirname "$STAGE")"
CID="$(docker create "$IMAGE")"
docker cp "$CID:/root/.local" "$STAGE"
docker rm "$CID" >/dev/null

echo "==> verifying the staged bundle"
docker run --rm -v "$STAGE:/root/.local" "$IMAGE" sh -c '
  export PATH="/root/.local/bin:$PATH"
  mini-swe-agent --version 2>&1 | head -1
  /root/.local/share/uv/tools/mini-swe-agent/bin/python -c "import mcp; print(\"mcp OK\")"
' || { echo "!! the staged bundle is not usable; the live one is untouched"; exit 1; }

echo "==> swapping in"
rm -rf "$OUT.old"
if [ -d "$OUT" ]; then mv "$OUT" "$OUT.old"; fi
mv "$STAGE" "$OUT"
rm -rf "$OUT.old"

echo "==> bundle: $(du -sh "$OUT" | cut -f1)"
ls "$OUT"
# ``bin/mini-swe-agent`` is a *symlink* to an absolute path inside the
# container (``/root/.local/...``), so ``test -x`` on the host always fails
# even when the bundle is fine -- the same reason rsi/paths.py uses
# ``lexists``.  The real check is the one above, from inside a container.
test -L "$OUT/bin/mini-swe-agent" || { echo "!! mini-swe-agent shim missing"; exit 1; }
echo "==> ok"
