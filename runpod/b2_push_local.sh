#!/usr/bin/env bash
# Push the local mirror at /workspace/b2/ back to the B2 bucket.
#
# Use this when init.sh fell back to the local-mirror approach
# (because /dev/fuse wasn't available) and you've created/modified
# files locally — e.g. `cp my_lora.safetensors /workspace/b2/loras/`.
#
# rclone copy is one-way: local -> remote. Files that exist on B2
# but not locally are NOT deleted. Use `rclone sync` flag for that.
#
# Default: push everything. Pass a subpath to scope it:
#   bash b2_push_local.sh                  # push /workspace/b2/* to b2:bucket/*
#   bash b2_push_local.sh loras/character_alex
#                                           # push only that subdir

set -e

export RCLONE_CONFIG="${RCLONE_CONFIG:-/workspace/.rclone/rclone.conf}"
B2_REMOTE="${B2_REMOTE:-b2}"

if [ -z "${B2_BUCKET:-}" ]; then
    echo "ERR: B2_BUCKET env var not set. Can't determine target bucket."
    exit 1
fi
if [ ! -f "$RCLONE_CONFIG" ]; then
    echo "ERR: $RCLONE_CONFIG missing. Configure B2 first."
    exit 1
fi
if [ ! -d /workspace/b2 ]; then
    echo "ERR: /workspace/b2 doesn't exist. Nothing to push."
    exit 1
fi

SUB="${1:-}"
if [ -n "$SUB" ]; then
    SRC="/workspace/b2/$SUB"
    DST="${B2_REMOTE}:${B2_BUCKET}/$SUB"
else
    SRC="/workspace/b2/"
    DST="${B2_REMOTE}:${B2_BUCKET}/"
fi

echo "[b2_push_local] $SRC  ->  $DST"
rclone copy --progress --transfers=8 "$SRC" "$DST"
echo "[b2_push_local] Done."
