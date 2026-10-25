#!/usr/bin/env bash
# Publish the finished SFT dataset as a GitHub Release.
#
# The dataset is gigabytes, so it is **not** in git: it goes out as a Release
# attachment, which is the GitHub-blessed place for a large artefact that
# people download rather than clone.
#
# The release also carries the summary the run produced: the row counts, the
# filter counts, the reward distribution, and the list of the rollouts that
# were archived instead of kept.
#
# Usage::
#
#     bash scripts/publish_release.sh                     # the ClawGym dataset
#     bash scripts/publish_release.sh openthoughts
#     TAG=clawgym-sft-v1 bash scripts/publish_release.sh
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

GITHUB_REPO="${GITHUB_REPO:-Xigmoid/harness-rsi}"
TAG="${TAG:-clawgym-sft-v1}"
which="${1:-clawgym}"

case "$which" in
  clawgym)
    DATA=opencode_sft/clawgym/dataset/clawgym-sft.jsonl
    SUMMARY=opencode_sft/clawgym/dataset/clawgym-sft.summary.json
    INVALID_DIR=opencode_sft/clawgym/dataset/invalid
    TITLE="ClawGym harness-agent SFT dataset"
    DATASET_NAME="ClawGym"
    ;;
  openthoughts)
    DATA=opencode_sft/dataset/openthoughts-sft.jsonl
    SUMMARY=opencode_sft/dataset/openthoughts-sft.summary.json
    INVALID_DIR=opencode_sft/dataset/invalid
    TITLE="OpenThoughts harness-agent SFT dataset"
    DATASET_NAME="OpenThoughts"
    ;;
  *)
    echo "usage: $0 [clawgym|openthoughts]" >&2
    exit 2
    ;;
esac

[[ -f "$DATA" ]] || { echo "$DATA does not exist -- run 'make dataset' first" >&2; exit 2; }

size=$(du -h "$DATA" | cut -f1)
rows=$(wc -l < "$DATA" | tr -d ' ')
echo "==> $DATA ($size, $rows rows)"

# The summary is the "what is in here" of the release: the row counts, the
# filter counts and the reward distribution.  The invalid archive goes with
# it, so that the rollouts which were *not* kept are still published -- they
# are not deleted, and the reason for each one is recorded.
assets=("$DATA")
[[ -f "$SUMMARY" ]] && assets+=("$SUMMARY")
if [[ -d "$INVALID_DIR" ]]; then
  for f in "$INVALID_DIR"/*; do
    [[ -f "$f" ]] && assets+=("$f")
  done
fi

echo "==> assets:"
for f in "${assets[@]}"; do
  printf '    %-70s %s\n' "$f" "$(du -h "$f" | cut -f1)"
done

# The notes go through a file rather than a heredoc inside a command
# substitution: ``"$(cat <<EOF ... EOF)"`` does not parse (the shell loses
# track of the heredoc terminator inside the nested quoting), and the notes
# themselves contain apostrophes and backticks.
NOTES_FILE="$(mktemp)"
trap 'rm -f "$NOTES_FILE"' EXIT

cat > "$NOTES_FILE" <<EOF
The $DATASET_NAME harness-agent SFT dataset.

One JSON object per line, OpenAI Chat Completions shaped.  Every row carries
the three provenance keys:

    dataset   $DATASET_NAME
    model     the model that produced it
    harness   Mini (mini-swe-agent) or Opencode

The \`messages\` key is the model-side conversation -- exactly what the agent
sent to the endpoint, plus the model's final answer.  The \`conversations\` key
is the same content in the strict user/assistant training view.

The rollouts that trip a discard rule are published as well, in the
\`invalid/\` asset, with the reason for each one.  They are archived, not
deleted.

Reproduce it: https://github.com/$GITHUB_REPO/blob/master/REPRODUCE.md
EOF

if gh release view "$TAG" --repo "$GITHUB_REPO" >/dev/null 2>&1; then
  echo "==> the release $TAG exists; adding assets to it"
  for f in "${assets[@]}"; do
    gh release upload "$TAG" "$f" --repo "$GITHUB_REPO" --clobber
  done
  gh release edit "$TAG" --repo "$GITHUB_REPO" \
    --title "$TITLE" --notes-file "$NOTES_FILE" >/dev/null
else
  echo "==> creating the release $TAG"
  gh release create "$TAG" "${assets[@]}" \
    --repo "$GITHUB_REPO" \
    --title "$TITLE" \
    --notes-file "$NOTES_FILE"
fi

echo "==> done: https://github.com/$GITHUB_REPO/releases/tag/$TAG"
