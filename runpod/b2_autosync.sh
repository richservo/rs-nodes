#!/usr/bin/env bash
# Background daemon: watch /workspace/b2/ for local file changes and
# push each one to B2 as it lands. Started by init.sh on every boot
# when /workspace/b2 is in local-mirror mode (FUSE unavailable).
#
# Mechanism: inotifywait -m -r emits an event whenever a file is
# closed-after-write or moved into the watch tree. We pipe those
# events through a small loop that runs rclone copyto for each.
# Additive only — local deletes do NOT delete on B2 (avoids
# accidents). Use b2_helpers.sh rm to delete on B2 explicitly.
#
# Idempotent: if started while one is already running, the previous
# instance is killed (init.sh's pkill before nohup handles that).

set -e

export RCLONE_CONFIG="${RCLONE_CONFIG:-/workspace/.rclone/rclone.conf}"
B2_REMOTE="${B2_REMOTE:-b2}"
# At feature-film scale we don't auto-sync everything under
# /workspace — only the LoRA outputs (small, valuable, the thing
# you actually want to distribute). Datasets are pulled explicitly
# per-character via b2_pull_dataset.sh and don't push back.
WATCH_DIR="${B2_AUTOSYNC_WATCH:-/workspace/ComfyUI/output/loras}"
# B2_AUTOSYNC_PREFIX is the path UNDER the bucket. So a file at
# $WATCH_DIR/character_x/foo.safetensors pushes to
# b2:$B2_BUCKET/$B2_AUTOSYNC_PREFIX/character_x/foo.safetensors
B2_AUTOSYNC_PREFIX="${B2_AUTOSYNC_PREFIX:-loras}"
LOG_FILE="${B2_AUTOSYNC_LOG:-/workspace/b2_autosync.log}"

# --- preflight -----------------------------------------------------------
if [ -z "${B2_BUCKET:-}" ]; then
    echo "$(date -Is) b2_autosync: B2_BUCKET env var not set — exiting." >> "$LOG_FILE"
    exit 1
fi
if ! command -v rclone >/dev/null 2>&1; then
    echo "$(date -Is) b2_autosync: rclone not installed — exiting." >> "$LOG_FILE"
    exit 1
fi
if ! command -v inotifywait >/dev/null 2>&1; then
    echo "$(date -Is) b2_autosync: inotifywait missing (install inotify-tools)." >> "$LOG_FILE"
    exit 1
fi
if [ ! -f "$RCLONE_CONFIG" ]; then
    echo "$(date -Is) b2_autosync: $RCLONE_CONFIG missing — exiting." >> "$LOG_FILE"
    exit 1
fi

mkdir -p "$WATCH_DIR"

echo "$(date -Is) b2_autosync starting" >> "$LOG_FILE"
echo "  watch:  $WATCH_DIR" >> "$LOG_FILE"
echo "  target: $B2_REMOTE:$B2_BUCKET/$B2_AUTOSYNC_PREFIX" >> "$LOG_FILE"

# --- the loop ------------------------------------------------------------
# -m   monitor mode (don't exit on first event)
# -r   recursive
# -q   quiet (suppress watcher's setup chatter)
# -e   limit to close_write (file finished writing) + moved_to (mv into dir)
# --format '%w%f'   emit just the full filesystem path of the changed file
inotifywait -m -r -q -e close_write,moved_to "$WATCH_DIR" \
    --format '%w%f' 2>>"$LOG_FILE" | \
while read -r filepath; do
    # Skip hidden / temp / rclone-internal files so the watcher
    # doesn't push every editor's swap file or rclone's in-flight
    # .partial chunks back to B2.
    base="$(basename "$filepath")"
    case "$base" in
        .*|*.partial|*~|*.swp|*.swx|*.tmp) continue ;;
    esac

    # Convert the local path to the B2-relative path under the bucket.
    # rel is the path under the watch dir; we prepend the configured
    # bucket-prefix so it lands at b2:bucket/<prefix>/<rel>.
    rel="${filepath#$WATCH_DIR/}"
    target="$B2_REMOTE:$B2_BUCKET/$B2_AUTOSYNC_PREFIX/$rel"

    # Push. rclone copyto is idempotent (md5 check skips no-op
    # transfers), so duplicate events are cheap (just a HEAD).
    if rclone copyto "$filepath" "$target" 2>>"$LOG_FILE"; then
        echo "$(date -Is) pushed: $rel -> $B2_AUTOSYNC_PREFIX/$rel" >> "$LOG_FILE"
    else
        echo "$(date -Is) push FAILED: $rel" >> "$LOG_FILE"
    fi
done
