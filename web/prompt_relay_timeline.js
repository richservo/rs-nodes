// RSPromptRelayTimeline — canvas timeline widget.
//
// Step G2: read-only canvas. Renders the time axis + tick marks + segment
// rectangles parsed from the underlying timeline_data STRING. No interaction
// yet (G3 adds drag/resize/add).
//
// Architecture: the node has a hidden-ish `timeline_data` STRING widget that
// stores the canonical JSON. The canvas renders from it. Future interaction
// steps will write back to it.

import { app } from "../../scripts/app.js";

// ----- Layout & style ----------------------------------------------------
const CANVAS_HEIGHT_BASE = 60;     // axis area
const ROW_HEIGHT = 26;             // per-segment row
const ROW_GAP = 4;
const PADDING_X = 12;
const AXIS_Y = 20;
const TICK_LONG_H = 6;
const TICK_SHORT_H = 3;
const SEG_FILL = "rgba(120, 180, 255, 0.55)";
const SEG_STROKE = "rgba(120, 180, 255, 1.0)";
const BG = "#1a1d22";
const AXIS_COLOR = "#888";
const TICK_COLOR = "#555";
const TEXT_COLOR = "#ddd";
const TEXT_DIM = "#888";

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
        const segs = Array.isArray(obj.segments) ? obj.segments : [];
        return {
            global: typeof obj.global === "string" ? obj.global : "",
            segments: segs
                .filter(
                    (s) =>
                        s &&
                        typeof s === "object" &&
                        typeof s.prompt === "string" &&
                        Number.isFinite(s.t_start) &&
                        Number.isFinite(s.t_end) &&
                        s.t_end > s.t_start
                )
                .map((s) => ({
                    t_start: Number(s.t_start),
                    t_end: Number(s.t_end),
                    prompt: String(s.prompt),
                })),
        };
    } catch (e) {
        return { global: "", segments: [] };
    }
}

// ----- Drawing -----------------------------------------------------------
function timeToX(t, totalSec, width) {
    const inner = Math.max(width - PADDING_X * 2, 1);
    return PADDING_X + (t / Math.max(totalSec, 1e-6)) * inner;
}

function draw(canvas, node) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 600;
    const state = readState(node);
    const totalSec = Number(getNumberWidget(node, "total_duration_sec")?.value ?? 4);
    const fps = Number(getNumberWidget(node, "frame_rate")?.value ?? 25);
    const cssH = CANVAS_HEIGHT_BASE + Math.max(state.segments.length, 1) * (ROW_HEIGHT + ROW_GAP);

    // Resize buffer to match CSS size at devicePixelRatio
    if (canvas.width !== Math.floor(cssW * dpr) || canvas.height !== Math.floor(cssH * dpr)) {
        canvas.width = Math.floor(cssW * dpr);
        canvas.height = Math.floor(cssH * dpr);
        canvas.style.height = cssH + "px";
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Background
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, cssW, cssH);

    // Frame snap interval (used as minor tick spacing too)
    const frameSec = 1 / Math.max(fps, 0.01);

    // Minor ticks (1/fps) — only if reasonable density
    const innerW = Math.max(cssW - PADDING_X * 2, 1);
    const minorPx = (frameSec / Math.max(totalSec, 1e-6)) * innerW;
    if (minorPx >= 4) {
        ctx.strokeStyle = TICK_COLOR;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let t = 0; t <= totalSec + 1e-6; t += frameSec) {
            const x = timeToX(t, totalSec, cssW);
            ctx.moveTo(x + 0.5, AXIS_Y);
            ctx.lineTo(x + 0.5, AXIS_Y + TICK_SHORT_H);
        }
        ctx.stroke();
    }

    // Major ticks (1s)
    ctx.strokeStyle = AXIS_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PADDING_X, AXIS_Y + 0.5);
    ctx.lineTo(cssW - PADDING_X, AXIS_Y + 0.5);
    ctx.stroke();

    ctx.fillStyle = TEXT_DIM;
    ctx.font = "10px sans-serif";
    ctx.textBaseline = "top";
    ctx.textAlign = "center";
    const sec_step = totalSec > 30 ? 5 : totalSec > 10 ? 1 : 0.5;
    ctx.beginPath();
    for (let t = 0; t <= totalSec + 1e-6; t += sec_step) {
        const x = timeToX(t, totalSec, cssW);
        ctx.moveTo(x + 0.5, AXIS_Y);
        ctx.lineTo(x + 0.5, AXIS_Y + TICK_LONG_H);
        const label = sec_step >= 1 ? `${t.toFixed(0)}s` : `${t.toFixed(1)}s`;
        ctx.fillText(label, x, AXIS_Y + TICK_LONG_H + 2);
    }
    ctx.stroke();

    // Segment rows
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    state.segments.forEach((seg, i) => {
        const y = CANVAS_HEIGHT_BASE - 8 + i * (ROW_HEIGHT + ROW_GAP);
        const x0 = timeToX(seg.t_start, totalSec, cssW);
        const x1 = timeToX(seg.t_end, totalSec, cssW);
        const w = Math.max(x1 - x0, 2);

        ctx.fillStyle = SEG_FILL;
        ctx.fillRect(x0, y, w, ROW_HEIGHT);

        ctx.strokeStyle = SEG_STROKE;
        ctx.lineWidth = 1;
        ctx.strokeRect(x0 + 0.5, y + 0.5, w - 1, ROW_HEIGHT - 1);

        // Label (clipped to segment width)
        ctx.save();
        ctx.beginPath();
        ctx.rect(x0 + 4, y, Math.max(w - 8, 0), ROW_HEIGHT);
        ctx.clip();
        ctx.fillStyle = TEXT_COLOR;
        ctx.font = "11px sans-serif";
        const label = seg.prompt || `(segment ${i + 1})`;
        ctx.fillText(label, x0 + 6, y + ROW_HEIGHT / 2);
        ctx.restore();

        // Time range bottom-right
        ctx.fillStyle = TEXT_DIM;
        ctx.font = "9px sans-serif";
        ctx.textAlign = "right";
        const range = `${seg.t_start.toFixed(2)}–${seg.t_end.toFixed(2)}s`;
        ctx.fillText(range, x1 - 4, y + ROW_HEIGHT - 6);
        ctx.textAlign = "left";
    });

    // Empty-state hint
    if (state.segments.length === 0) {
        ctx.fillStyle = TEXT_DIM;
        ctx.font = "11px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        const y = CANVAS_HEIGHT_BASE - 8 + ROW_HEIGHT / 2;
        ctx.fillText(
            "No segments yet — paste JSON into 'timeline_data' (interactions land in G3)",
            cssW / 2,
            y
        );
    }

    // Update node size if our height changed
    const totalNeeded = cssH + 8;
    if (canvas._lastReportedHeight !== totalNeeded) {
        canvas._lastReportedHeight = totalNeeded;
        // Trigger node to recompute size; ComfyUI does this when widgets resize.
        if (node.setDirtyCanvas) node.setDirtyCanvas(true, true);
    }
}

// ----- Widget wiring -----------------------------------------------------
function attachTimeline(node) {
    if (node._rsTimelineAttached) return;
    node._rsTimelineAttached = true;

    const wrapper = document.createElement("div");
    wrapper.style.cssText = "width:100%;box-sizing:border-box;padding:0 4px;";

    const canvas = document.createElement("canvas");
    canvas.style.cssText = "width:100%;display:block;border-radius:4px;";
    canvas.width = 600;
    canvas.height = CANVAS_HEIGHT_BASE + ROW_HEIGHT + ROW_GAP;
    wrapper.appendChild(canvas);

    // Insert the canvas widget BEFORE the timeline_data textarea so the JSON
    // sits at the bottom as a fallback view.
    const widget = node.addDOMWidget(
        "rs_timeline_canvas",
        "canvas",
        wrapper,
        { serialize: false }
    );

    // Reorder: place canvas widget right above timeline_data
    const dataWidget = getDataWidget(node);
    if (dataWidget) {
        const ci = node.widgets.indexOf(widget);
        const di = node.widgets.indexOf(dataWidget);
        if (ci !== -1 && di !== -1 && ci !== di - 1) {
            node.widgets.splice(ci, 1);
            const newDi = node.widgets.indexOf(dataWidget);
            node.widgets.splice(newDi, 0, widget);
        }
    }

    // Re-render hooks: any of these widget changes should redraw.
    const triggerDraw = () => draw(canvas, node);
    for (const name of ["timeline_data", "total_duration_sec", "frame_rate"]) {
        const w = node.widgets?.find((w) => w.name === name);
        if (!w) continue;
        const orig = w.callback;
        w.callback = function (...args) {
            const r = orig?.apply(this, args);
            triggerDraw();
            return r;
        };
    }

    // Workflow-load redraw
    const origConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
        const r = origConfigure?.apply(this, args);
        // Defer one frame so widgets are populated
        requestAnimationFrame(triggerDraw);
        return r;
    };

    // Initial draw (deferred so layout has settled)
    requestAnimationFrame(triggerDraw);

    // Expose for future steps
    node._rsTimeline = { canvas, draw: triggerDraw };
}

app.registerExtension({
    name: "rs-nodes.PromptRelayTimeline",
    nodeCreated(node) {
        if (node.comfyClass === "RSPromptRelayTimeline") {
            attachTimeline(node);
        }
    },
});
