// RSPromptRelayTimeline — interactive canvas timeline.
//
// User-facing UX: type a Style line, click "+ Add", type each segment's prompt,
// drag blocks on a single timeline track. JSON is the output, hidden from the UI.

import { app } from "../../scripts/app.js";

// ----- Constants ---------------------------------------------------------
const PADDING_X     = 12;
const AXIS_Y        = 18;        // axis line y; labels render below this
const AXIS_AREA_H   = 40;        // axis area above the segment track
const TRACK_H       = 38;        // segment track height (proportional, not dominating)
const CANVAS_H      = AXIS_AREA_H + TRACK_H + 8;
const EDGE_HOT_PX   = 9;
const HANDLE_PX     = 4;
const ADJACENT_EPS  = 1e-3;     // segments are "touching" if their boundaries are within this many seconds
const SEG_RADIUS    = 4;
const BG            = "#1a1d22";
const SEG_FILL      = "rgba(120, 180, 255, 0.45)";
const SEG_FILL_SEL  = "rgba(120, 200, 255, 0.85)";
const SEG_STROKE    = "rgba(120, 180, 255, 1.0)";
const SEG_STROKE_SEL= "rgba(255, 255, 255, 1.0)";
const AXIS_COLOR    = "#888";
const TICK_COLOR    = "#555";
const TEXT_COLOR    = "#ddd";
const TEXT_DIM      = "#888";
const HINT_COLOR    = "#666";
const MIN_SEG_SEC   = 0.04;

// ----- State helpers -----------------------------------------------------
function getDataWidget(node) {
    return node.widgets?.find((w) => w.name === "timeline_data");
}
function getNumberWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function readState(node) {
    const w = getDataWidget(node);
    let raw = w?.value ?? "";
    try {
        const obj = JSON.parse(raw || "{}");
        return {
            global: typeof obj.global === "string" ? obj.global : "",
            segments: (Array.isArray(obj.segments) ? obj.segments : [])
                .filter(
                    (s) => s && typeof s === "object" && typeof s.prompt === "string"
                        && Number.isFinite(s.t_start) && Number.isFinite(s.t_end)
                        && s.t_end > s.t_start
                )
                .map((s) => ({
                    t_start: Number(s.t_start),
                    t_end: Number(s.t_end),
                    prompt: String(s.prompt),
                })),
        };
    } catch {
        return { global: "", segments: [] };
    }
}

function writeState(node, state) {
    const w = getDataWidget(node);
    if (!w) return;
    const segs = [...state.segments].sort((a, b) => a.t_start - b.t_start);
    const out = {
        global: state.global || "",
        segments: segs.map((s) => ({
            t_start: Math.round(s.t_start * 1000) / 1000,
            t_end:   Math.round(s.t_end   * 1000) / 1000,
            prompt:  s.prompt,
        })),
    };
    const json = JSON.stringify(out, null, 2);
    if (w.value !== json) {
        w.value = json;
        if (typeof w.callback === "function") {
            try { w.callback(w.value); } catch {}
        }
    }
}

function snapToFrame(t, fps, enabled) {
    if (!enabled || !Number.isFinite(fps) || fps <= 0) return t;
    return Math.round(t * fps) / fps;
}

// Re-tile segments to fill exactly [0, totalSec], preserving each segment's
// relative width. Mutates segments in place.
function rescaleToFill(segments, totalSec) {
    if (!segments.length || totalSec <= 0) return;
    const sorted = [...segments].sort((a, b) => a.t_start - b.t_start);
    const totalSpan = sorted.reduce((s, x) => s + Math.max(x.t_end - x.t_start, 0), 0);
    let cursor = 0;
    if (totalSpan <= 0) {
        const w = totalSec / sorted.length;
        for (const s of sorted) {
            s.t_start = cursor;
            cursor += w;
            s.t_end = cursor;
        }
    } else {
        const scale = totalSec / totalSpan;
        for (const s of sorted) {
            const w = Math.max((s.t_end - s.t_start) * scale, 0);
            s.t_start = cursor;
            cursor += w;
            s.t_end = cursor;
        }
    }
    // Pin last edge to totalSec to absorb floating-point drift.
    sorted[sorted.length - 1].t_end = totalSec;
}

// ----- Geometry ----------------------------------------------------------
function timeToX(t, totalSec, cssW) {
    const innerW = Math.max(cssW - PADDING_X * 2, 1);
    return PADDING_X + (t / Math.max(totalSec, 1e-6)) * innerW;
}
function xToTime(x, totalSec, cssW) {
    const innerW = Math.max(cssW - PADDING_X * 2, 1);
    return ((x - PADDING_X) / innerW) * Math.max(totalSec, 1e-6);
}

function hitTest(state, totalSec, cssW, mx, my) {
    if (my < AXIS_AREA_H || my > AXIS_AREA_H + TRACK_H) return null;
    // First pass: prefer edge hits (resize) over body hits (move) — even if a body
    // overlaps another segment's edge, the edge wins for grab-the-trim feel.
    let bodyHit = null;
    const ordered = state.segments
        .map((s, i) => ({ s, i }))
        .sort((a, b) => a.s.t_start - b.s.t_start);
    for (const { s, i } of ordered) {
        const x0 = timeToX(s.t_start, totalSec, cssW);
        const x1 = timeToX(s.t_end, totalSec, cssW);
        if (mx >= x0 - EDGE_HOT_PX && mx <= x0 + EDGE_HOT_PX) return { index: i, region: "left" };
        if (mx >= x1 - EDGE_HOT_PX && mx <= x1 + EDGE_HOT_PX) return { index: i, region: "right" };
        if (mx > x0 + EDGE_HOT_PX && mx < x1 - EDGE_HOT_PX) bodyHit = { index: i, region: "body" };
    }
    return bodyHit;
}

// Find rolling-trim neighbour: returns the adjacent segment whose boundary
// touches `seg`'s edge on the given side, or null if there's a gap.
function findAdjacent(state, seg, side) {
    const sorted = [...state.segments].sort((a, b) => a.t_start - b.t_start);
    const idx = sorted.indexOf(seg);
    if (idx < 0) return null;
    if (side === "right") {
        const next = sorted[idx + 1];
        if (next && Math.abs(next.t_start - seg.t_end) <= ADJACENT_EPS) return next;
    } else if (side === "left") {
        const prev = sorted[idx - 1];
        if (prev && Math.abs(prev.t_end - seg.t_start) <= ADJACENT_EPS) return prev;
    }
    return null;
}


// ----- Drawing -----------------------------------------------------------
function draw(canvas, node, runtime) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 600;
    const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);
    const fps = Number(getNumberWidget(node, "frame_rate")?.value ?? 25);

    if (canvas.width !== Math.floor(cssW * dpr) || canvas.height !== Math.floor(CANVAS_H * dpr)) {
        canvas.width = Math.floor(cssW * dpr);
        canvas.height = Math.floor(CANVAS_H * dpr);
        canvas.style.height = CANVAS_H + "px";
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Background
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, cssW, CANVAS_H);

    // Pick a tick stride that keeps labels readable (>= ~50px apart).
    const innerW = Math.max(cssW - PADDING_X * 2, 1);
    const pxPerSec = innerW / Math.max(totalSec, 1e-6);
    const candidates = [0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60];
    let majorStep = candidates[candidates.length - 1];
    for (const c of candidates) {
        if (c * pxPerSec >= 50) { majorStep = c; break; }
    }
    const minorStep = majorStep >= 1 ? majorStep / (majorStep >= 5 ? 5 : 2) : majorStep / 2;
    const frameSec = 1 / Math.max(fps, 0.01);

    // Minor ticks: minorStep, plus per-frame faint ticks if dense enough.
    ctx.strokeStyle = TICK_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let t = 0; t <= totalSec + 1e-6; t += minorStep) {
        const x = timeToX(t, totalSec, cssW);
        ctx.moveTo(x + 0.5, AXIS_Y);
        ctx.lineTo(x + 0.5, AXIS_Y + 5);
    }
    ctx.stroke();
    if (frameSec * pxPerSec >= 4) {
        ctx.strokeStyle = "#3a3d44";
        ctx.beginPath();
        for (let t = 0; t <= totalSec + 1e-6; t += frameSec) {
            const x = timeToX(t, totalSec, cssW);
            ctx.moveTo(x + 0.5, AXIS_Y);
            ctx.lineTo(x + 0.5, AXIS_Y + 3);
        }
        ctx.stroke();
    }

    // Axis line
    ctx.strokeStyle = AXIS_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PADDING_X, AXIS_Y + 0.5);
    ctx.lineTo(cssW - PADDING_X, AXIS_Y + 0.5);
    ctx.stroke();

    // Major ticks + labels
    ctx.fillStyle = TEXT_COLOR;
    ctx.font = "11px sans-serif";
    ctx.textBaseline = "top";
    ctx.textAlign = "center";
    ctx.strokeStyle = AXIS_COLOR;
    ctx.beginPath();
    const dec = majorStep >= 1 ? 0 : (majorStep >= 0.1 ? 1 : 2);
    for (let t = 0; t <= totalSec + 1e-6; t += majorStep) {
        const x = timeToX(t, totalSec, cssW);
        ctx.moveTo(x + 0.5, AXIS_Y);
        ctx.lineTo(x + 0.5, AXIS_Y + 9);
        const label = `${t.toFixed(dec)}s`;
        ctx.fillText(label, x, AXIS_Y + 11);
    }
    ctx.stroke();

    // Track baseline
    const trackY = AXIS_AREA_H;
    ctx.strokeStyle = "#2a2d33";
    ctx.lineWidth = 1;
    ctx.strokeRect(PADDING_X + 0.5, trackY + 0.5, innerW - 1, TRACK_H - 1);

    // Segments — all on the SAME track row, time-ordered. Overlaps are
    // visualised by translucent fill stacking; selected drawn on top.
    const sel = runtime.getSelected();
    const renderOrder = runtime.state.segments
        .map((s, i) => ({ s, i }))
        .sort((a, b) => a.s.t_start - b.s.t_start)
        .filter(({ s }) => s !== sel);
    if (sel && runtime.state.segments.indexOf(sel) >= 0) {
        renderOrder.push({ s: sel, i: runtime.state.segments.indexOf(sel) });
    }

    for (const { s, i } of renderOrder) {
        const isSel = s === sel;
        const x0 = timeToX(s.t_start, totalSec, cssW);
        const x1 = timeToX(s.t_end, totalSec, cssW);
        const w = Math.max(x1 - x0, 2);

        ctx.fillStyle = isSel ? SEG_FILL_SEL : SEG_FILL;
        roundRect(ctx, x0, trackY + 2, w, TRACK_H - 4, SEG_RADIUS);
        ctx.fill();
        ctx.strokeStyle = isSel ? SEG_STROKE_SEL : SEG_STROKE;
        ctx.lineWidth = isSel ? 2 : 1;
        roundRect(ctx, x0 + 0.5, trackY + 2.5, w - 1, TRACK_H - 5, SEG_RADIUS);
        ctx.stroke();
        ctx.lineWidth = 1;

        // Always-visible trim handles (so users see where to grab).
        const handleColor = isSel ? "rgba(255,255,255,0.95)" : "rgba(255,255,255,0.55)";
        ctx.fillStyle = handleColor;
        ctx.fillRect(x0, trackY + 2, HANDLE_PX, TRACK_H - 4);
        ctx.fillRect(x1 - HANDLE_PX, trackY + 2, HANDLE_PX, TRACK_H - 4);

        // Label clipped to body
        ctx.save();
        ctx.beginPath();
        ctx.rect(x0 + 4, trackY, Math.max(w - 8, 0), TRACK_H);
        ctx.clip();
        ctx.fillStyle = TEXT_COLOR;
        ctx.font = "11px sans-serif";
        ctx.textBaseline = "middle";
        ctx.textAlign = "left";
        const label = s.prompt || `(segment ${i + 1})`;
        ctx.fillText(label, x0 + 6, trackY + TRACK_H / 2 - 4);
        ctx.fillStyle = TEXT_DIM;
        ctx.font = "9px sans-serif";
        ctx.fillText(`${s.t_start.toFixed(2)}–${s.t_end.toFixed(2)}s`, x0 + 6, trackY + TRACK_H - 6);
        ctx.restore();
    }

    if (runtime.state.segments.length === 0) {
        ctx.fillStyle = HINT_COLOR;
        ctx.font = "11px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("Click “+ Add” to start • drag body = move • drag edges = resize",
                     cssW / 2, trackY + TRACK_H / 2);
    }
}

function roundRect(ctx, x, y, w, h, r) {
    const rad = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    ctx.moveTo(x + rad, y);
    ctx.arcTo(x + w, y, x + w, y + h, rad);
    ctx.arcTo(x + w, y + h, x, y + h, rad);
    ctx.arcTo(x, y + h, x, y, rad);
    ctx.arcTo(x, y, x + w, y, rad);
    ctx.closePath();
}

// ----- Widget wiring -----------------------------------------------------
function hideTimelineDataWidget(node) {
    const w = getDataWidget(node);
    if (!w) return;
    // Multiple belts to stop ComfyUI from rendering the textarea while still
    // serializing the value. Different ComfyUI versions honour different ones.
    w.computeSize = () => [0, -4];
    try { Object.defineProperty(w, "type", { value: "hidden", configurable: true, writable: true }); }
    catch { w.type = "hidden"; }
    w.draw = () => {};
    if (w.element) {
        w.element.style.display = "none";
        if (w.element.parentElement) w.element.parentElement.style.display = "none";
    }
    if (w.inputEl) {
        w.inputEl.style.display = "none";
        if (w.inputEl.parentElement) w.inputEl.parentElement.style.display = "none";
    }
}

function attachTimeline(node) {
    if (node._rsTimelineAttached) return;
    node._rsTimelineAttached = true;

    hideTimelineDataWidget(node);

    // Fixed pixel heights for each row so flex-shrink can't collapse textareas.
    const H_LABEL  = 14;
    const H_STYLE  = 50;
    const H_TOOL   = 28;
    const H_EDITOR = 64;
    const GAP      = 6;
    const PADV     = 8;        // top + bottom padding inside root
    const TOTAL_H  = PADV + H_LABEL + GAP + H_STYLE + GAP + H_TOOL + GAP + CANVAS_H + GAP + H_LABEL + GAP + H_EDITOR;

    const childCSS = "flex-shrink:0;flex-grow:0;";

    // Build DOM
    const root = document.createElement("div");
    root.style.cssText =
        "width:100%;display:flex;flex-direction:column;gap:" + GAP + "px;" +
        "padding:" + (PADV / 2) + "px 6px;box-sizing:border-box;font-family:sans-serif;";

    const styleLabel = document.createElement("div");
    styleLabel.textContent = "STYLE  (always-on context)";
    styleLabel.style.cssText = childCSS +
        "height:" + H_LABEL + "px;line-height:" + H_LABEL + "px;" +
        "color:#aaa;font-size:10px;font-weight:bold;letter-spacing:0.5px;";

    const styleArea = document.createElement("textarea");
    styleArea.placeholder = "e.g. cinematic 35mm, warm tungsten light, woman in red dress";
    styleArea.style.cssText = childCSS +
        "width:100%;height:" + H_STYLE + "px;box-sizing:border-box;background:#222;color:#eee;" +
        "border:1px solid #444;border-radius:3px;padding:4px 6px;font:11px sans-serif;resize:none;";

    const toolbar = document.createElement("div");
    toolbar.style.cssText = childCSS +
        "height:" + H_TOOL + "px;display:flex;gap:6px;align-items:center;";
    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.textContent = "+ Add";
    addBtn.style.cssText =
        "background:#2c5;color:#fff;border:none;border-radius:3px;padding:4px 10px;cursor:pointer;font:11px sans-serif;";
    const delBtn = document.createElement("button");
    delBtn.type = "button";
    delBtn.textContent = "× Delete";
    delBtn.style.cssText =
        "background:#623;color:#fff;border:none;border-radius:3px;padding:4px 10px;cursor:pointer;font:11px sans-serif;";
    delBtn.disabled = true;
    delBtn.style.opacity = "0.5";
    toolbar.appendChild(addBtn);
    toolbar.appendChild(delBtn);

    const canvas = document.createElement("canvas");
    canvas.tabIndex = 0;
    canvas.style.cssText = childCSS +
        "width:100%;height:" + CANVAS_H + "px;display:block;border-radius:4px;outline:none;";
    canvas.width = 600;
    canvas.height = CANVAS_H;

    const editorLabel = document.createElement("div");
    editorLabel.textContent = "SELECTED SEGMENT";
    editorLabel.style.cssText = childCSS +
        "height:" + H_LABEL + "px;line-height:" + H_LABEL + "px;" +
        "color:#aaa;font-size:10px;font-weight:bold;letter-spacing:0.5px;";

    const editorArea = document.createElement("textarea");
    editorArea.placeholder = "(select a segment to edit its prompt)";
    editorArea.disabled = true;
    editorArea.style.cssText = childCSS +
        "width:100%;height:" + H_EDITOR + "px;box-sizing:border-box;background:#222;color:#eee;" +
        "border:1px solid #444;border-radius:3px;padding:4px 6px;font:11px sans-serif;resize:none;opacity:0.5;";

    root.appendChild(styleLabel);
    root.appendChild(styleArea);
    root.appendChild(toolbar);
    root.appendChild(canvas);
    root.appendChild(editorLabel);
    root.appendChild(editorArea);

    const widget = node.addDOMWidget("rs_timeline_ui", "ui", root, { serialize: false });
    // Fixed total height — ComfyUI sums widget computeSize values to lay out the node.
    widget.computeSize = function (width) {
        return [width || 240, TOTAL_H];
    };

    // ----- Runtime state --------------------------------------------------
    const runtime = {
        state: readState(node),
        selectedRef: null,    // direct reference to a segment object (stable across edits)
        drag: null,
        getSelected() { return runtime.selectedRef; },
    };
    styleArea.value = runtime.state.global || "";

    function commit() {
        writeState(node, runtime.state);
        redraw();
    }
    function redraw() {
        const sel = runtime.getSelected();
        if (sel && runtime.state.segments.indexOf(sel) >= 0) {
            if (document.activeElement !== editorArea || editorArea.value !== sel.prompt) {
                editorArea.value = sel.prompt;
            }
            editorArea.disabled = false;
            editorArea.style.opacity = "1";
            delBtn.disabled = false;
            delBtn.style.opacity = "1";
        } else {
            runtime.selectedRef = null;
            editorArea.value = "";
            editorArea.disabled = true;
            editorArea.style.opacity = "0.5";
            delBtn.disabled = true;
            delBtn.style.opacity = "0.5";
        }
        draw(canvas, node, runtime);
    }

    // Resize observer just triggers a canvas redraw when the node width changes.
    const ro = new ResizeObserver(() => { draw(canvas, node, runtime); });
    ro.observe(root);

    // ----- Listeners ------------------------------------------------------
    styleArea.addEventListener("input", () => {
        runtime.state.global = styleArea.value;
        writeState(node, runtime.state);
    });

    editorArea.addEventListener("input", () => {
        const sel = runtime.getSelected();
        if (!sel) return;
        sel.prompt = editorArea.value;
        commit();
    });

    addBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);

        const N = runtime.state.segments.length;
        if (N === 0) {
            const seg = { prompt: "", t_start: 0, t_end: totalSec };
            runtime.state.segments.push(seg);
            runtime.selectedRef = seg;
            commit();
            editorArea.focus();
            return;
        }

        // New segment gets the equal-share default size. Existing segments
        // preserve their relative widths and rescale to fit the remaining space.
        const newLen = totalSec / (N + 1);
        rescaleToFill(runtime.state.segments, totalSec - newLen);
        const newSeg = { prompt: "", t_start: totalSec - newLen, t_end: totalSec };
        runtime.state.segments.push(newSeg);
        runtime.selectedRef = newSeg;
        commit();
        editorArea.focus();
    });

    delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const sel = runtime.getSelected();
        if (!sel) return;
        const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);
        runtime.state.segments = runtime.state.segments.filter((s) => s !== sel);
        runtime.selectedRef = null;
        rescaleToFill(runtime.state.segments, totalSec);
        commit();
    });

    function canvasMouse(e) {
        // Convert from VISUAL pixels (post-transform — workflow zoom etc.) to
        // LAYOUT pixels (the coordinate space the drawing code uses).
        const r = canvas.getBoundingClientRect();
        const sx = r.width  > 0 ? canvas.clientWidth  / r.width  : 1;
        const sy = r.height > 0 ? canvas.clientHeight / r.height : 1;
        return {
            x: (e.clientX - r.left) * sx,
            y: (e.clientY - r.top)  * sy,
        };
    }

    canvas.addEventListener("mousedown", (e) => {
        canvas.focus();
        const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);
        const cssW = canvas.clientWidth || 600;
        const { x, y } = canvasMouse(e);
        const hit = hitTest(runtime.state, totalSec, cssW, x, y);
        if (hit) {
            const seg = runtime.state.segments[hit.index];
            runtime.selectedRef = seg;
            const t = xToTime(x, totalSec, cssW);
            let partner = null;
            // Snapshot all fixed pivots / widths up-front for the body-drag swap logic.
            const fixedPivots = new Map();
            const fixedWidths = new Map();
            for (const s of runtime.state.segments) {
                fixedPivots.set(s, (s.t_start + s.t_end) / 2);
                fixedWidths.set(s, s.t_end - s.t_start);
            }
            if (hit.region === "left") {
                partner = findAdjacent(runtime.state, seg, "left");
            } else if (hit.region === "right") {
                partner = findAdjacent(runtime.state, seg, "right");
            }
            runtime.drag = {
                kind: hit.region,
                seg,
                mouseT0: t,
                segT0: seg.t_start,
                segT1: seg.t_end,
                origLen: seg.t_end - seg.t_start,
                partner,
                partnerT0: partner ? partner.t_start : null,
                partnerT1: partner ? partner.t_end : null,
                fixedPivots,
                fixedWidths,
            };
            redraw();
            e.preventDefault();
        } else {
            runtime.selectedRef = null;
            redraw();
        }
    });

    function onMove(e) {
        if (!runtime.drag) {
            const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);
            const { x, y } = canvasMouse(e);
            const hit = hitTest(runtime.state, totalSec, canvas.clientWidth || 600, x, y);
            canvas.style.cursor = !hit ? "default" :
                hit.region === "left" || hit.region === "right" ? "ew-resize" : "move";
            return;
        }
        const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);
        const fps = Number(getNumberWidget(node, "frame_rate")?.value ?? 25);
        const cssW = canvas.clientWidth || 600;
        const { x } = canvasMouse(e);
        const t = xToTime(x, totalSec, cssW);
        const dt = t - runtime.drag.mouseT0;
        const seg = runtime.drag.seg;
        if (runtime.state.segments.indexOf(seg) < 0) { runtime.drag = null; return; }
        const snap = !e.shiftKey;
        const sorted = [...runtime.state.segments].sort((a, b) => a.t_start - b.t_start);
        const idx = sorted.indexOf(seg);

        if (runtime.drag.kind === "body") {
            // Body drag: slide-through-with-swap.
            //   - Dragged seg keeps its size.
            //   - IMMEDIATE neighbors' touching edges follow it (prev's right = seg's left,
            //     next's left = seg's right).
            //   - NON-immediate segments sit at their CANONICAL widths starting from
            //     either end (so they appear unchanged).
            //   - When the dragged seg's pivot crosses a canonical pivot of another seg,
            //     they swap order — the "passed" seg pops back to canonical size on the
            //     other side of the dragged seg.
            const len = runtime.drag.origLen;
            const tentativeStart = runtime.drag.segT0 + dt;
            const tentativePivot = tentativeStart + len / 2;

            const others = runtime.state.segments.filter((s) => s !== seg);
            others.sort((a, b) => runtime.drag.fixedPivots.get(a) - runtime.drag.fixedPivots.get(b));

            let segIdx = 0;
            for (const o of others) {
                if (runtime.drag.fixedPivots.get(o) < tentativePivot) segIdx++;
            }
            const newOrder = [...others.slice(0, segIdx), seg, ...others.slice(segIdx)];

            // Sum canonical widths of NON-immediate segments on each side. (Immediate
            // neighbours are at indices segIdx-1 and segIdx+1 in newOrder.)
            let leftSum = 0;
            for (let i = 0; i < segIdx - 1; i++) {
                leftSum += runtime.drag.fixedWidths.get(newOrder[i]);
            }
            let rightSum = 0;
            for (let i = segIdx + 2; i < newOrder.length; i++) {
                rightSum += runtime.drag.fixedWidths.get(newOrder[i]);
            }
            const hasLeftN  = segIdx > 0;
            const hasRightN = segIdx < newOrder.length - 1;

            // Clamp tentative start so neighbours can't shrink below MIN.
            const minStart = leftSum + (hasLeftN ? MIN_SEG_SEC : 0);
            const maxStart = totalSec - rightSum - len - (hasRightN ? MIN_SEG_SEC : 0);
            let xs = Math.max(minStart, Math.min(tentativeStart, maxStart));
            xs = snapToFrame(xs, fps, snap);
            xs = Math.max(minStart, Math.min(xs, maxStart));
            const xe = xs + len;

            // Layout in newOrder.
            let cursor = 0;
            for (let i = 0; i < newOrder.length; i++) {
                const s = newOrder[i];
                if (s === seg) {
                    s.t_start = xs;
                    s.t_end   = xe;
                    cursor = xe;
                } else if (i === segIdx - 1) {
                    // Immediate left: from cursor (= leftSum after non-imm left) to xs.
                    s.t_start = cursor;
                    s.t_end   = xs;
                    cursor = xs;
                } else if (i === segIdx + 1) {
                    // Immediate right: from xe to (totalSec - rightSum).
                    s.t_start = cursor;
                    s.t_end   = totalSec - rightSum;
                    cursor = s.t_end;
                } else {
                    // Non-immediate: canonical width, tiled from cursor.
                    const w = runtime.drag.fixedWidths.get(s);
                    s.t_start = cursor;
                    s.t_end   = cursor + w;
                    cursor = s.t_end;
                }
            }
        } else if (runtime.drag.kind === "left") {
            // Don't allow crossing the previous segment (regardless of partner status).
            const prev = sorted[idx - 1];
            const minStart = prev ? prev.t_start + MIN_SEG_SEC : 0; // partner can't shrink below MIN
            const partnerCap = runtime.drag.partner
                ? Math.max(runtime.drag.partner.t_start + MIN_SEG_SEC, prev ? prev.t_start + MIN_SEG_SEC : 0)
                : (prev ? prev.t_end : 0);
            let ns = runtime.drag.segT0 + dt;
            ns = Math.max(partnerCap, Math.min(ns, seg.t_end - MIN_SEG_SEC));
            ns = snapToFrame(ns, fps, snap);
            ns = Math.max(partnerCap, Math.min(ns, seg.t_end - MIN_SEG_SEC));
            seg.t_start = ns;
            // Rolling trim: partner (touching prev neighbour) follows our left edge.
            if (runtime.drag.partner) {
                runtime.drag.partner.t_end = ns;
            }
        } else if (runtime.drag.kind === "right") {
            const next = sorted[idx + 1];
            const partnerCap = runtime.drag.partner
                ? Math.min(runtime.drag.partner.t_end - MIN_SEG_SEC, totalSec)
                : (next ? next.t_start : totalSec);
            let ne = runtime.drag.segT1 + dt;
            ne = Math.min(partnerCap, Math.max(ne, seg.t_start + MIN_SEG_SEC));
            ne = snapToFrame(ne, fps, snap);
            ne = Math.min(partnerCap, Math.max(ne, seg.t_start + MIN_SEG_SEC));
            seg.t_end = ne;
            // Rolling trim: partner (touching next neighbour) follows our right edge.
            if (runtime.drag.partner) {
                runtime.drag.partner.t_start = ne;
            }
        }
        commit();
    }
    function onUp() {
        if (runtime.drag) { runtime.drag = null; redraw(); }
    }
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);

    canvas.addEventListener("keydown", (e) => {
        if ((e.key === "Delete" || e.key === "Backspace") && runtime.getSelected()) {
            const sel = runtime.getSelected();
            const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);
            runtime.state.segments = runtime.state.segments.filter((s) => s !== sel);
            runtime.selectedRef = null;
            rescaleToFill(runtime.state.segments, totalSec);
            commit();
            e.preventDefault();
        }
    });

    canvas.addEventListener("contextmenu", (e) => {
        const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);
        const { x, y } = canvasMouse(e);
        const hit = hitTest(runtime.state, totalSec, canvas.clientWidth || 600, x, y);
        if (hit) {
            const seg = runtime.state.segments[hit.index];
            runtime.state.segments = runtime.state.segments.filter((s) => s !== seg);
            if (runtime.selectedRef === seg) runtime.selectedRef = null;
            rescaleToFill(runtime.state.segments, totalSec);
            commit();
            e.preventDefault();
        }
    });

    // total_duration_sec changes: rescale all segments proportionally to fit the new total.
    {
        const w = getNumberWidget(node, "total_duration_sec");
        if (w) {
            const orig = w.callback;
            w.callback = function (...args) {
                const r = orig?.apply(this, args);
                const newTotal = Number(w.value ?? 4);
                if (runtime.state.segments.length > 0 && newTotal > 0) {
                    rescaleToFill(runtime.state.segments, newTotal);
                    writeState(node, runtime.state);
                }
                redraw();
                return r;
            };
        }
    }
    // frame_rate changes: just redraw (tick density may change).
    {
        const w = getNumberWidget(node, "frame_rate");
        if (w) {
            const orig = w.callback;
            w.callback = function (...args) {
                const r = orig?.apply(this, args);
                redraw();
                return r;
            };
        }
    }

    const origConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
        const r = origConfigure?.apply(this, args);
        requestAnimationFrame(() => {
            hideTimelineDataWidget(node);
            runtime.state = readState(node);
            styleArea.value = runtime.state.global || "";
            runtime.selectedRef = null;
            redraw();
        });
        return r;
    };

    // Initial sizing pass + first render
    requestAnimationFrame(() => {
        hideTimelineDataWidget(node);
        redraw();
    });

    node._rsTimeline = { canvas, redraw };
}

app.registerExtension({
    name: "rs-nodes.PromptRelayTimeline",
    nodeCreated(node) {
        if (node.comfyClass === "RSPromptRelayTimeline") {
            attachTimeline(node);
        }
    },
});
