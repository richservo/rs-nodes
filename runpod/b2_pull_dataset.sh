#!/usr/bin/env bash
# Pull a specific dataset from B2 to the pod for training.
#
# Feature-film-scale workflow: B2 holds all characters' datasets
# (multi-TB total). A pod only ever has one or two characters'
# worth pulled to local disk at a time — the rest stays on B2.
# Training reads from local /workspace/datasets/<name>/ at full
# disk speed; no FUSE round-trips, no whole-bucket mirror.
#
# Usage:
#   bash b2_pull_dataset.sh <name>
#   bash b2_pull_dataset.sh character_alex
#
# Maps to:
#   b2:$B2_BUCKET/datasets/<name>/  ->  /workspace/datasets/<name>/

set -e

export RCLONE_CONFIG="${RCLONE_CONFIG:-/workspace/.rclone/rclone.conf}"
B2_REMOTE="${B2_REMOTE:-b2}"
DATASETS_PREFIX="${B2_DATASETS_PREFIX:-datasets}"
LOCAL_DATASETS="${LOCAL_DATASETS:-/workspace/datasets}"

NAME="${1:-}"
if [ -z "$NAME" ]; then
    echo "Usage: bash b2_pull_dataset.sh <character-name>"
    echo
    echo "Available on B2:"
    if [ -z "${B2_BUCKET:-}" ]; then
        echo "  (B2_BUCKET not set)"
    else
        rclone lsf --max-depth 1 "${B2_REMOTE}:${B2_BUCKET}/${DATASETS_PREFIX}" 2>/dev/null || \
            echo "  (none, or couldn't list — check 'b2_helpers.sh status')"
    fi
    exit 1
fi

if [ -z "${B2_BUCKET:-}" ]; then
    echo "ERR: B2_BUCKET env var not set on this pod."
    echo "     Set it in RunPod console -> Env Variables and stop+start the pod."
    exit 1
fi
if [ ! -f "$RCLONE_CONFIG" ]; then
    echo "ERR: $RCLONE_CONFIG missing. B2 not configured yet."
    exit 1
fi

REMOTE="${B2_REMOTE}:${B2_BUCKET}/${DATASETS_PREFIX}/${NAME}"
LOCAL="${LOCAL_DATASETS}/${NAME}"

mkdir -p "$LOCAL"

echo "[b2_pull_dataset] $REMOTE  ->  $LOCAL"
echo "[b2_pull_dataset] Incremental — only new/changed files transfer."
echo
rclone copy --progress --transfers=8 "$REMOTE" "$LOCAL"
echo
echo "[b2_pull_dataset] Done."
echo "[b2_pull_dataset] Dataset ready at: $LOCAL"
echo "[b2_pull_dataset] Point your training workflow at that path."
