#!/usr/bin/env bash
# rs-nodes pod-side CLOUD BUCKET sync (Cloudflare R2, S3, GCS, Backblaze B2, Wasabi, any
# S3-compatible). Pods have no FUSE, so nothing is mounted: the bucket is COPIED down into
# /workspace at start (restore) and copied back while running (save). Never deletes anything,
# locally or in the bucket (rclone copy only).
#
# Usage: bash bucket_sync.sh restore | save | daemon | status
#
# Env (set by Frames; RS_BUCKET_CONF arrives as a RunPod secret):
#   RS_BUCKET_CONF      the whole rclone.conf text with ONE remote named [store]
#   RS_BUCKET_PATH      "<bucket>[/<prefix>]"
#   RS_BUCKET_POD       pod name (the per-pod outputs folder)
#   RS_BUCKET_SAVE_MIN  periodic save minutes for the daemon (default 30; 0 = off)
# With RS_BUCKET_CONF / RS_BUCKET_PATH unset every subcommand is a no-op (exit 0).
#
# Bucket layout:  <path>/models/ <-> /workspace/ComfyUI/models   (LoRAs included, shared)
#                 <path>/input/  <-> /workspace/ComfyUI/input    (shared)
#                 <path>/ollama/ <-> /workspace/.ollama/models   (shared)
#                 <path>/outputs/<pod>/ <-> /workspace/ComfyUI/output (per pod, same pod only)
#
# Supersedes b2_helpers.sh's rclone setup: rclone lives in /workspace/.bin (cached on the volume).
#
# State: /workspace/.rs/bucket_state.json
#   {"phase": "restoring"|"saving"|"idle"|"error", "lastRestore": epoch, "lastSave": epoch,
#    "error": string|null, "updatedAt": epoch}
# Log:   /workspace/.rs/bucket_sync.log (never contains the conf or any secret)

set -u
umask 077
STATE_DIR=/workspace/.rs
STATE_FILE="$STATE_DIR/bucket_state.json"
LOG_FILE="$STATE_DIR/bucket_sync.log"
LOCK_FILE="$STATE_DIR/bucket_sync.lock"
DAEMON_PID="$STATE_DIR/bucket_daemon.pid"
BIN_DIR=/workspace/.bin
RCLONE_DIR=/workspace/.rclone
export RCLONE_CONFIG="$RCLONE_DIR/rclone.conf"
RCLONE_URL="https://downloads.rclone.org/rclone-current-linux-amd64.zip"

mkdir -p "$STATE_DIR" "$BIN_DIR"
SELF="$0"
CMD="${1:-status}"

log() { echo "[bucket $(date '+%F %T')] $*" >> "$LOG_FILE"; echo "[bucket] $*"; }

# ---- no bucket configured: do nothing ----------------------------------------------------
if [ -z "${RS_BUCKET_CONF:-}" ] || [ -z "${RS_BUCKET_PATH:-}" ]; then
    case "$CMD" in
        restore|save|daemon|status) log "no bucket configured"; exit 0 ;;
        *) echo "usage: bucket_sync.sh restore|save|daemon|status"; exit 2 ;;
    esac
fi

BPATH="${RS_BUCKET_PATH%/}"
BPATH="${BPATH#/}"
POD="${RS_BUCKET_POD:-}"
POD=$(printf '%s' "$POD" | tr -c 'A-Za-z0-9._-' '_')
SAVE_MIN="${RS_BUCKET_SAVE_MIN:-30}"
SAVE_MIN=$(printf '%s' "$SAVE_MIN" | tr -dc '0-9')
SAVE_MIN=${SAVE_MIN:-30}

# ---- state file (atomic) ------------------------------------------------------------------
state_num() { grep -o "\"$1\": [0-9]*" "$STATE_FILE" 2>/dev/null | head -1 | tr -dc '0-9'; }
write_state() { # phase [error] [restored|saved]
    local phase="$1" err="${2:-}" lr ls errj now
    lr=$(state_num lastRestore); lr=${lr:-0}
    ls=$(state_num lastSave); ls=${ls:-0}
    [ "${3:-}" = "restored" ] && lr=$(date +%s)
    [ "${3:-}" = "saved" ] && ls=$(date +%s)
    if [ -n "$err" ]; then
        err=$(printf '%s' "$err" | tr -d '"\\\n\r' | cut -c1-300)
        errj="\"$err\""
    else
        errj=null
    fi
    now=$(date +%s)
    printf '{"phase": "%s", "lastRestore": %d, "lastSave": %d, "error": %s, "updatedAt": %d}\n' \
        "$phase" "$lr" "$ls" "$errj" "$now" > "$STATE_FILE.tmp" && mv -f "$STATE_FILE.tmp" "$STATE_FILE"
}

# ---- rclone: cached on the volume, installed once -----------------------------------------
RCLONE=""
ensure_rclone() {
    if [ -x "$BIN_DIR/rclone" ]; then
        RCLONE="$BIN_DIR/rclone"
    elif command -v rclone >/dev/null 2>&1; then
        RCLONE=$(command -v rclone)
    else
        log "installing rclone to $BIN_DIR"
        local tmp="/workspace/.rclone_dl.$$"
        rm -rf "$tmp"; mkdir -p "$tmp"
        if ! curl -fsSL --retry 3 "$RCLONE_URL" -o "$tmp/rclone.zip"; then
            log "rclone download failed"; rm -rf "$tmp"; return 1
        fi
        if command -v unzip >/dev/null 2>&1; then
            unzip -q -o "$tmp/rclone.zip" -d "$tmp/x" || { log "unzip failed"; rm -rf "$tmp"; return 1; }
        else
            python3 -c 'import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$tmp/rclone.zip" "$tmp/x" \
                || { log "python3 unzip failed"; rm -rf "$tmp"; return 1; }
        fi
        local found
        found=$(find "$tmp/x" -type f -name rclone | head -1)
        if [ -z "$found" ]; then log "rclone binary not found in the zip"; rm -rf "$tmp"; return 1; fi
        cp -f "$found" "$BIN_DIR/rclone.new" && chmod 755 "$BIN_DIR/rclone.new" && mv -f "$BIN_DIR/rclone.new" "$BIN_DIR/rclone"
        rm -rf "$tmp"
        RCLONE="$BIN_DIR/rclone"
    fi
    if ! "$RCLONE" version >/dev/null 2>&1; then
        log "rclone does not run ($RCLONE)"; return 1
    fi
    return 0
}

write_conf() {
    mkdir -p "$RCLONE_DIR" && chmod 700 "$RCLONE_DIR"
    printf '%s\n' "$RS_BUCKET_CONF" > "$RCLONE_CONFIG.tmp" && chmod 600 "$RCLONE_CONFIG.tmp" && mv -f "$RCLONE_CONFIG.tmp" "$RCLONE_CONFIG"
}

# ---- lock: restore and save never run at the same time ------------------------------------
LOCK_KIND=""
take_lock() { # wait-seconds (0 = do not wait)
    local wait="$1"
    if command -v flock >/dev/null 2>&1; then
        exec 9>"$LOCK_FILE"
        if [ "$wait" -gt 0 ]; then flock -w "$wait" 9 || return 1; else flock -n 9 || return 1; fi
        LOCK_KIND=flock
        return 0
    fi
    local t=0 old
    while ! mkdir "$LOCK_FILE.d" 2>/dev/null; do
        old=$(cat "$LOCK_FILE.d/pid" 2>/dev/null || true)
        if [ -n "$old" ] && ! kill -0 "$old" 2>/dev/null; then rm -rf "$LOCK_FILE.d"; continue; fi
        [ "$t" -ge "$wait" ] && return 1
        sleep 1; t=$((t + 1))
    done
    echo $$ > "$LOCK_FILE.d/pid"
    LOCK_KIND=dir
    return 0
}
drop_lock() { if [ "$LOCK_KIND" = dir ]; then rm -rf "$LOCK_FILE.d"; fi; return 0; }

# ---- copy one folder ----------------------------------------------------------------------
EXCL=(--exclude '*.tmp' --exclude '*.part' --exclude '*.incomplete' --exclude '.incomplete' --exclude '__pycache__/**' --exclude '.cache/**')
COMMON=(--transfers 8 --checkers 16 --fast-list --stats 30s --stats-one-line --log-level NOTICE --retries 3 --low-level-retries 10)
FAILS=""
run_copy() { # src dst [extra flags...]
    local src="$1" dst="$2" rc
    shift 2
    log "copy $src -> $dst"
    "$RCLONE" copy "$src" "$dst" "${COMMON[@]}" "${EXCL[@]}" "$@" >> "$LOG_FILE" 2>&1
    rc=$?
    # rc 3 = source directory not found (nothing in the bucket yet): not an error
    if [ "$rc" -ne 0 ] && [ "$rc" -ne 3 ]; then
        log "copy failed (rclone exit $rc): $src"
        FAILS="$FAILS $(basename "$dst")"
    fi
    return 0
}

save_dir() { # local-dir remote
    if [ -d "$1" ]; then run_copy "$1" "$2" --update; fi
    return 0
}

do_restore() {
    ensure_rclone || { write_state error "rclone unavailable"; return 1; }
    write_conf
    write_state restoring
    log "restore start"
    FAILS=""
    run_copy "store:$BPATH/models" /workspace/ComfyUI/models
    run_copy "store:$BPATH/input" /workspace/ComfyUI/input
    run_copy "store:$BPATH/ollama" /workspace/.ollama/models
    if [ -n "$POD" ]; then run_copy "store:$BPATH/outputs/$POD" /workspace/ComfyUI/output; fi
    if [ -n "$FAILS" ]; then
        log "restore finished with errors:$FAILS"; write_state error "restore failed:$FAILS" restored; return 1
    fi
    log "restore done"; write_state idle "" restored
    return 0
}

do_save() {
    ensure_rclone || { write_state error "rclone unavailable"; return 1; }
    write_conf
    write_state saving
    log "save start"
    FAILS=""
    save_dir /workspace/ComfyUI/models "store:$BPATH/models"
    save_dir /workspace/ComfyUI/input "store:$BPATH/input"
    save_dir /workspace/.ollama/models "store:$BPATH/ollama"
    if [ -n "$POD" ]; then save_dir /workspace/ComfyUI/output "store:$BPATH/outputs/$POD"; fi
    if [ -n "$FAILS" ]; then
        log "save finished with errors:$FAILS"; write_state error "save failed:$FAILS" saved; return 1
    fi
    log "save done"; write_state idle "" saved
    return 0
}

case "$CMD" in
    restore)
        take_lock 1800 || { log "another sync holds the lock - restore skipped"; exit 1; }
        do_restore; rc=$?; drop_lock; exit $rc ;;
    save)
        w=1800
        if [ "${2:-}" = "--nowait" ]; then w=0; fi
        take_lock "$w" || { log "another sync is running - save skipped"; exit 0; }
        do_save; rc=$?; drop_lock; exit $rc ;;
    daemon)
        if [ "$SAVE_MIN" -le 0 ]; then log "periodic save off (RS_BUCKET_SAVE_MIN=0)"; exit 0; fi
        if [ -f "$DAEMON_PID" ]; then
            old=$(cat "$DAEMON_PID" 2>/dev/null || true)
            if [ -n "$old" ] && [ "$old" != "$$" ] && kill -0 "$old" 2>/dev/null; then kill "$old" 2>/dev/null || true; fi
        fi
        echo $$ > "$DAEMON_PID"
        log "daemon started: save every $SAVE_MIN min"
        while true; do
            sleep $((SAVE_MIN * 60))
            bash "$SELF" save --nowait >/dev/null 2>&1 || true
        done ;;
    status)
        if [ -f "$STATE_FILE" ]; then cat "$STATE_FILE"; else echo '{"phase": "idle", "lastRestore": 0, "lastSave": 0, "error": null, "updatedAt": 0}'; fi
        exit 0 ;;
    *) echo "usage: bucket_sync.sh restore|save|daemon|status"; exit 2 ;;
esac
