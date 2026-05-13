#!/usr/bin/env bash
# Sourceable rclone wrappers for Backblaze B2 (and any other rclone
# remote configured in /workspace/.rclone/rclone.conf).
#
# Usage:
#   source /workspace/b2_helpers.sh
#   b2_check
#   b2_pull datasets/character_alex /workspace/datasets/character_alex
#   b2_push /workspace/ComfyUI/output/loras/alex loras/alex
#   b2_link loras/alex/final.safetensors 168h
#
# All paths after the bucket prefix are relative to the bucket root —
# the remote name (`b2:` or whatever's in rclone.conf) is supplied by
# B2_REMOTE below.
#
# Credentials live in /workspace/.rclone/rclone.conf, which rs-studio
# writes via SFTP (encrypted-at-rest using Electron safeStorage on the
# user's machine; transmitted over the existing SSH session; chmod
# 0600 once it lands on the pod). This script never sees plaintext
# secrets — it just tells rclone where to find the config.

# ---- Configuration --------------------------------------------------------
export RCLONE_CONFIG="${RCLONE_CONFIG:-/workspace/.rclone/rclone.conf}"
B2_REMOTE="${B2_REMOTE:-b2}"   # the [name] of the rclone remote stanza

# ---- High-level status (one-shot diagnostic) ------------------------------
# Shows everything that matters for "is B2 working on this pod" in
# one go: config presence, rclone reachable, bucket top-level
# listing, local mirror state, autosync daemon state, last log lines.
b2_status() {
    echo "============================================================"
    echo "  B2 status on this pod"
    echo "============================================================"
    echo
    echo "rclone.conf:  $RCLONE_CONFIG"
    if [ -f "$RCLONE_CONFIG" ]; then
        echo "  $(ls -la "$RCLONE_CONFIG" | awk '{print $1, $3, $4, $5, "bytes"}')"
    else
        echo "  MISSING — B2 not configured yet."
        echo "  Set B2_KEY_ID + B2_APP_KEY env vars and restart the pod."
        return 1
    fi
    echo
    echo "B2_BUCKET env var: ${B2_BUCKET:-(NOT SET — needed for auto-mirror)}"
    echo
    echo "--- rclone connection check ---"
    b2_check
    echo
    echo "--- bucket top level (b2:${B2_BUCKET:-}) ---"
    if [ -n "${B2_BUCKET:-}" ]; then
        rclone lsf --max-depth 1 "${B2_REMOTE}:${B2_BUCKET}" 2>&1 | head -20 || \
            echo "(bucket empty or unreachable)"
    else
        rclone lsf --max-depth 1 "${B2_REMOTE}:" 2>&1 | head -20 || \
            echo "(no buckets accessible)"
    fi
    echo
    echo "--- /workspace/b2/ local mirror (top level) ---"
    if [ -d /workspace/b2 ]; then
        ls /workspace/b2/ 2>/dev/null | head -20
        echo "  ($(du -sh /workspace/b2 2>/dev/null | cut -f1) total)"
    else
        echo "  /workspace/b2 doesn't exist"
    fi
    echo
    echo "--- autosync daemon (local edits -> B2) ---"
    if pgrep -f "b2_autosync.sh" >/dev/null 2>&1; then
        echo "  RUNNING (PID $(pgrep -f b2_autosync.sh | head -1))"
    else
        echo "  not running"
    fi
    echo
    echo "--- last 10 lines of b2_sync.log (initial pull) ---"
    tail -10 /workspace/b2_sync.log 2>/dev/null || echo "  (no log yet)"
    echo
    echo "--- last 10 lines of b2_autosync.log (live pushes) ---"
    tail -10 /workspace/b2_autosync.log 2>/dev/null || echo "  (no log yet)"
    echo "============================================================"
}

# ---- Status / preflight ---------------------------------------------------
b2_check() {
    if ! command -v rclone >/dev/null 2>&1; then
        echo "ERR: rclone not installed. Run /workspace/install_rclone.sh"
        return 1
    fi
    if [ ! -f "$RCLONE_CONFIG" ]; then
        echo "ERR: $RCLONE_CONFIG missing. rs-studio writes it on connect — open the B2 settings panel."
        return 1
    fi
    if ! rclone listremotes | grep -q "^${B2_REMOTE}:"; then
        echo "ERR: no [${B2_REMOTE}] remote in $RCLONE_CONFIG"
        return 1
    fi
    # Cheap auth/network smoke test — `about` returns bucket stats.
    if ! rclone about "${B2_REMOTE}:" >/dev/null 2>&1; then
        echo "ERR: rclone can't reach ${B2_REMOTE}: (check creds / bucket / network)"
        return 1
    fi
    echo "OK: ${B2_REMOTE}: reachable, $(rclone about "${B2_REMOTE}:" 2>/dev/null | grep -E '^(Total|Used):' | tr '\n' ' ')"
    return 0
}

# ---- Listing --------------------------------------------------------------
b2_ls() {
    # b2_ls [prefix]   — one-level listing (files + dirs)
    rclone lsf --max-depth 1 "${B2_REMOTE}:${1:-}"
}

b2_lsl() {
    # b2_lsl [prefix]  — recursive listing with sizes (slower)
    rclone lsl "${B2_REMOTE}:${1:-}"
}

# ---- Transfer -------------------------------------------------------------
b2_pull() {
    # b2_pull <remote_path> <local_path>
    #   Copy down from B2 to local. Skips files that already exist
    #   locally with matching size+mtime.
    if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
        echo "Usage: b2_pull <remote_path> <local_path>"
        return 1
    fi
    mkdir -p "$2"
    rclone copy --progress --transfers=8 "${B2_REMOTE}:$1" "$2"
}

b2_push() {
    # b2_push <local_path> <remote_path>
    #   Copy up from local to B2. Idempotent; resumable.
    if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
        echo "Usage: b2_push <local_path> <remote_path>"
        return 1
    fi
    rclone copy --progress --transfers=8 "$1" "${B2_REMOTE}:$2"
}

b2_sync() {
    # b2_sync <local_path> <remote_path>
    #   ONE-WAY mirror: makes remote IDENTICAL to local. Deletes
    #   remote files that aren't in local. Use copy unless you mean
    #   to delete things on B2.
    if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
        echo "Usage: b2_sync <local_path> <remote_path>"
        return 1
    fi
    rclone sync --progress --transfers=8 "$1" "${B2_REMOTE}:$2"
}

# ---- Share links ----------------------------------------------------------
b2_link() {
    # b2_link <remote_path> [duration]
    #   Generate a presigned HTTPS URL for the file. Hand to web
    #   services / team members for one-off downloads without giving
    #   them auth. Default expiry: 7 days.
    if [ -z "${1:-}" ]; then
        echo "Usage: b2_link <remote_path> [duration like 24h, 168h, 30d]"
        return 1
    fi
    rclone link --expire "${2:-168h}" "${B2_REMOTE}:$1"
}

# ---- Convenience ----------------------------------------------------------
b2_size() {
    # b2_size <remote_path>  — total bytes + file count under a prefix
    rclone size "${B2_REMOTE}:${1:-}"
}

b2_rm() {
    # b2_rm <remote_path>  — delete a single file. No --recursive
    # protection by design; use rclone purge by hand if you mean it.
    if [ -z "${1:-}" ]; then
        echo "Usage: b2_rm <remote_path>"
        return 1
    fi
    rclone deletefile "${B2_REMOTE}:$1"
}

# If run directly (not sourced), dispatch on $1 so rs-studio's SSH
# exec can do `bash /workspace/b2_helpers.sh pull foo /workspace/foo`
# without sourcing first.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    cmd="${1:-}"
    shift || true
    case "$cmd" in
        status) b2_status "$@" ;;
        check)  b2_check "$@" ;;
        ls)     b2_ls "$@" ;;
        lsl)    b2_lsl "$@" ;;
        pull)   b2_pull "$@" ;;
        push)   b2_push "$@" ;;
        sync)   b2_sync "$@" ;;
        link)   b2_link "$@" ;;
        size)   b2_size "$@" ;;
        rm)     b2_rm "$@" ;;
        ""|help|-h|--help)
            cat <<'EOF'
Usage: bash b2_helpers.sh <command> [args...]

  status                           One-shot summary: config, bucket, mirror, daemon
  check                            Verify rclone + config + connectivity
  ls    [prefix]                   One-level listing
  lsl   [prefix]                   Recursive listing with sizes
  pull  <remote> <local>           Copy down from B2
  push  <local> <remote>           Copy up to B2
  sync  <local> <remote>           One-way mirror (deletes remote extras!)
  link  <remote> [duration]        Generate presigned URL (default 168h)
  size  [prefix]                   Bytes + file count under a prefix
  rm    <remote>                   Delete a single file on B2
EOF
            ;;
        *)
            echo "Unknown command: $cmd"
            echo "Run: bash $0 help"
            exit 1
            ;;
    esac
fi
