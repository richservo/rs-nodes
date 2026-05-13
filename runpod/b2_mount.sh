#!/usr/bin/env bash
# Mount the configured B2 bucket as a filesystem at /workspace/b2/.
# Once mounted, you can `ls`, `cat`, drag-drop into the directory and
# every operation proxies through rclone to B2. Good for browsing and
# one-off transfers.
#
# *Don't train against this mount* — every file read is a network
# round-trip to B2. Pull datasets to /workspace/datasets/ via
# b2_helpers.sh pull before training; use the mount for inspection
# and casual file work only.
#
# Subcommands:
#   bash b2_mount.sh mount       — start the mount (idempotent)
#   bash b2_mount.sh unmount     — stop the mount
#   bash b2_mount.sh status      — is it mounted? log location?
#
# Reads /workspace/.rclone/rclone.conf (set by rs-studio or b2_setup.bat).

set -e

export RCLONE_CONFIG="${RCLONE_CONFIG:-/workspace/.rclone/rclone.conf}"
B2_REMOTE="${B2_REMOTE:-b2}"
# If B2_BUCKET is set in the pod env, mount that specific bucket
# directly so /workspace/b2/datasets/ resolves immediately. Without
# it, /workspace/b2/ is a directory listing all accessible buckets
# (which is an extra hop for the common single-bucket setup).
B2_BUCKET="${B2_BUCKET:-}"
if [ -n "$B2_BUCKET" ]; then
    MOUNT_TARGET="${B2_REMOTE}:${B2_BUCKET}"
else
    MOUNT_TARGET="${B2_REMOTE}:"
fi
MOUNT_POINT="${B2_MOUNT_POINT:-/workspace/b2}"
LOG_FILE="${B2_MOUNT_LOG:-/workspace/b2_mount.log}"
PID_FILE="${B2_MOUNT_PID:-/workspace/.b2_mount.pid}"

# ---- preflight ------------------------------------------------------------
require_config() {
    if [ ! -f "$RCLONE_CONFIG" ]; then
        echo "ERR: $RCLONE_CONFIG missing — configure B2 first (rs-studio or b2_setup.bat)"
        exit 1
    fi
}

require_rclone() {
    if ! command -v rclone >/dev/null 2>&1; then
        echo "ERR: rclone not installed. Run /workspace/install_rclone.sh"
        exit 1
    fi
}

require_fuse() {
    # rclone mount needs FUSE. On most RunPod base images it's already
    # available; on minimal images we need to apt-install it. Check
    # for /dev/fuse first — that's what actually has to be present.
    if [ ! -c /dev/fuse ]; then
        echo "ERR: /dev/fuse missing — this container can't FUSE-mount."
        echo "     (Possible on some minimal base images. Use b2_helpers.sh"
        echo "     pull/push instead — no FUSE required.)"
        exit 1
    fi
    # fusermount or fusermount3 binary needs to exist for rclone to
    # cleanly unmount. Auto-install if missing.
    if ! command -v fusermount >/dev/null 2>&1 && ! command -v fusermount3 >/dev/null 2>&1; then
        echo "[b2_mount] Installing FUSE userland..."
        apt-get update -qq && apt-get install -y -qq fuse3 || \
            apt-get install -y -qq fuse || {
                echo "ERR: could not install fuse / fuse3 via apt"
                exit 1
            }
    fi
}

# ---- subcommands ----------------------------------------------------------
is_mounted() {
    # mountpoint(1) is the canonical check; fall back to grepping /proc/mounts
    if command -v mountpoint >/dev/null 2>&1; then
        mountpoint -q "$MOUNT_POINT"
    else
        grep -q " $MOUNT_POINT " /proc/mounts
    fi
}

cmd_mount() {
    require_rclone
    require_config
    require_fuse

    mkdir -p "$MOUNT_POINT"

    if is_mounted; then
        echo "[b2_mount] Already mounted at $MOUNT_POINT"
        rclone_pid=$(cat "$PID_FILE" 2>/dev/null || echo "?")
        echo "[b2_mount] rclone PID: $rclone_pid"
        echo "[b2_mount] log: $LOG_FILE"
        return 0
    fi

    echo "[b2_mount] Mounting $MOUNT_TARGET -> $MOUNT_POINT"
    echo "[b2_mount] log: $LOG_FILE"

    # --vfs-cache-mode writes  : buffer writes locally before pushing,
    #                            so partial writes don't corrupt files
    # --dir-cache-time 1m      : refresh dir listings every minute
    # --vfs-read-chunk-size 32M: balance latency vs throughput on B2
    # --allow-non-empty        : skip the safety check on /workspace/b2
    # --daemon                 : background; without this the script
    #                            would block until manual unmount
    rclone mount "$MOUNT_TARGET" "$MOUNT_POINT" \
        --vfs-cache-mode writes \
        --dir-cache-time 1m \
        --vfs-read-chunk-size 32M \
        --allow-non-empty \
        --log-file "$LOG_FILE" \
        --log-level INFO \
        --daemon

    # Wait up to 10s for the mount to actually come online before
    # reporting success — rclone's --daemon returns instantly but
    # the FUSE syscall is async.
    for _i in $(seq 1 20); do
        if is_mounted; then break; fi
        sleep 0.5
    done

    if ! is_mounted; then
        echo "ERR: rclone exited but $MOUNT_POINT isn't mounted. Check $LOG_FILE"
        tail -20 "$LOG_FILE" 2>/dev/null
        exit 1
    fi

    # Record the rclone PID so unmount knows what to kill.
    pgrep -f "rclone mount $MOUNT_TARGET $MOUNT_POINT" > "$PID_FILE" || true

    echo "[b2_mount] Mounted. Browse with:  ls $MOUNT_POINT/"
}

cmd_unmount() {
    if ! is_mounted; then
        echo "[b2_mount] Not currently mounted at $MOUNT_POINT"
        return 0
    fi

    echo "[b2_mount] Unmounting $MOUNT_POINT..."
    if command -v fusermount3 >/dev/null 2>&1; then
        fusermount3 -u "$MOUNT_POINT" || fusermount -u "$MOUNT_POINT"
    else
        fusermount -u "$MOUNT_POINT"
    fi

    # Kill the rclone process if it's still hanging around.
    if [ -f "$PID_FILE" ]; then
        kill "$(cat "$PID_FILE")" 2>/dev/null || true
        rm -f "$PID_FILE"
    fi

    echo "[b2_mount] Unmounted."
}

cmd_status() {
    if is_mounted; then
        echo "Mounted: yes"
        echo "  Mount point: $MOUNT_POINT"
        echo "  Log file:    $LOG_FILE"
        if [ -f "$PID_FILE" ]; then
            echo "  rclone PID:  $(cat "$PID_FILE")"
        fi
        echo
        echo "Contents (top level):"
        ls -la "$MOUNT_POINT" 2>/dev/null | head -10
    else
        echo "Mounted: no"
        echo "  Run: bash $0 mount"
    fi
}

# ---- dispatch -------------------------------------------------------------
cmd="${1:-status}"
case "$cmd" in
    mount)    cmd_mount ;;
    unmount|umount)  cmd_unmount ;;
    status)   cmd_status ;;
    help|-h|--help)
        cat <<EOF
Usage: bash b2_mount.sh <command>

  mount     Mount $MOUNT_TARGET at $MOUNT_POINT (idempotent)
  unmount   Unmount $MOUNT_POINT
  status    Report whether the bucket is currently mounted

Environment overrides:
  B2_REMOTE          rclone remote name              (default: b2)
  B2_MOUNT_POINT     where to mount the bucket       (default: /workspace/b2)
  RCLONE_CONFIG      path to rclone.conf             (default: /workspace/.rclone/rclone.conf)

DO NOT TRAIN against the mount — random file reads are slow over the
network. Pull datasets to /workspace/datasets/ first. The mount is
for browsing, inspection, and one-off transfers.
EOF
        ;;
    *)
        echo "Unknown command: $cmd"
        echo "Run: bash $0 help"
        exit 1
        ;;
esac
