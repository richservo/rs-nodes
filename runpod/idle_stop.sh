#!/usr/bin/env bash
# rs-nodes pod-side GPU IDLE auto-stop. Started by init.sh in the background on every boot of a
# GPU pod (ComfyUI and NO_COMFY alike). STOPS the pod (never terminates: /workspace survives)
# after the GPUs have been idle for the set number of minutes, so a forgotten pod does not run
# all night. Works with Frames closed - it runs on the pod.
#
# Limit (minutes, 0 = off). Any write to the limit file (even the same value: Frames'
# "Keep running") restarts the idle count.
#   /workspace/.rs/idle_stop_min   - the live value; Frames writes it over SSH to change the timer
#                                    without a restart (changing pod env vars restarts the pod)
#   RS_IDLE_STOP_MIN               - env; seeds the file only when the file does not exist yet
# State for Frames' countdown: /workspace/.rs/idle_state.json
#   {"idleMin": N, "limitMin": N, "gpuUtil": N, "busy": 0|1, "updatedAt": epochSeconds}
# Idle = the busiest GPU at or under RS_IDLE_GPU_UTIL percent (default 5). Minutes spent while a
# bootstrap / node-pack install / model download runs never count as idle.
# Stops itself with the pod-scoped RUNPOD_API_KEY RunPod sets on every pod.

set -u
STATE_DIR=/workspace/.rs
LIMIT_FILE="$STATE_DIR/idle_stop_min"
STATE_FILE="$STATE_DIR/idle_state.json"
LOG_FILE="$STATE_DIR/idle_stop.log"
PID_FILE="$STATE_DIR/idle_stop.pid"
THRESH="${RS_IDLE_GPU_UTIL:-5}"

mkdir -p "$STATE_DIR"

# One watchdog per pod: retire a previous instance (init.sh can run again on a container restart).
if [ -f "$PID_FILE" ]; then
    old=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$old" ] && [ "$old" != "$$" ] && kill -0 "$old" 2>/dev/null; then kill "$old" 2>/dev/null || true; fi
fi
echo $$ > "$PID_FILE"

[ -f "$LIMIT_FILE" ] || echo "${RS_IDLE_STOP_MIN:-0}" > "$LIMIT_FILE"

log() { echo "[idle-stop $(date '+%F %T')] $*" >> "$LOG_FILE"; }

stop_pod() {
    log "GPU idle for $1 min (limit $2) - stopping pod ${RUNPOD_POD_ID:-?}"
    if [ -z "${RUNPOD_POD_ID:-}" ]; then log "RUNPOD_POD_ID not set - cannot stop"; return 1; fi
    if command -v runpodctl >/dev/null 2>&1; then
        runpodctl pod stop "$RUNPOD_POD_ID" >> "$LOG_FILE" 2>&1 && return 0
        runpodctl stop pod "$RUNPOD_POD_ID" >> "$LOG_FILE" 2>&1 && return 0
    fi
    if [ -n "${RUNPOD_API_KEY:-}" ]; then
        curl -fsS -X POST "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}/action" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" -H "Content-Type: application/json" \
            -d '{"action":"stop"}' >> "$LOG_FILE" 2>&1 && return 0
    fi
    log "stop failed (no runpodctl / RUNPOD_API_KEY, or the API refused)"
    return 1
}

log "started (limit $(cat "$LIMIT_FILE" 2>/dev/null) min, idle at <= ${THRESH}% GPU)"
idle=0
limit_mtime=$(stat -c %Y "$LIMIT_FILE" 2>/dev/null || echo 0)
while true; do
    m=$(stat -c %Y "$LIMIT_FILE" 2>/dev/null || echo 0)
    if [ "$m" != "$limit_mtime" ]; then limit_mtime=$m; idle=0; log "limit set to $(cat "$LIMIT_FILE" 2>/dev/null) min - idle count restarted"; fi
    limit=$(tr -dc '0-9' < "$LIMIT_FILE" 2>/dev/null)
    limit=${limit:-0}
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9\n' | sort -n | tail -1)
    util=${util:-0}
    busy=0
    if pgrep -f "bootstrap.sh|install_extras.sh|download_models.sh|install_ollama.sh|redo_bootstrap.sh" >/dev/null 2>&1; then busy=1; fi
    if [ "$busy" = "1" ] || [ "$util" -gt "$THRESH" ]; then idle=0; else idle=$((idle + 1)); fi
    printf '{"idleMin": %d, "limitMin": %d, "gpuUtil": %d, "busy": %d, "updatedAt": %d}\n' \
        "$idle" "$limit" "$util" "$busy" "$(date +%s)" > "$STATE_FILE.tmp" && mv -f "$STATE_FILE.tmp" "$STATE_FILE"
    if [ "$limit" -gt 0 ] && [ "$idle" -ge "$limit" ]; then
        stop_pod "$idle" "$limit" && exit 0
        idle=0  # stop failed: try again after another full idle period, not every minute
    fi
    sleep 60
done
