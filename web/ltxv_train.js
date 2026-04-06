import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const PREPARE_SECTIONS = [
    ["Dataset", "media_folder"],
    ["Model Paths", "model_path"],
    ["Captioning", "caption_mode"],
    ["Options", "resolution_buckets"],
    ["Face Detection", "face_detection"],
    ["IC-LoRA", "conditioning_folder"],
];

const TRAIN_SECTIONS = [
    ["Model", "model"],
    ["Preset", "preset"],
    ["LoRA Config", "lora_rank"],
    ["Module Selection", "video_self_attention"],
    ["Training", "learning_rate"],
    ["Quantization", "quantization"],
    ["Strategy", "strategy"],
    ["Validation", "clip"],
    ["Checkpoints", "checkpoint_interval"],
    ["Resume", "resume_checkpoint"],
];

// Preset definitions: { modules: {toggle: bool}, rank, alpha }
const PRESETS = {
    "subject": {
        video_self_attention: true,
        video_cross_attention: true,
        video_feed_forward: false,
        audio_self_attention: false,
        audio_cross_attention: false,
        audio_feed_forward: false,
        video_attends_to_audio: false,
        audio_attends_to_video: false,
        lora_rank: 16,
        lora_alpha: 16,
    },
    "style": {
        video_self_attention: true,
        video_cross_attention: false,
        video_feed_forward: true,
        audio_self_attention: false,
        audio_cross_attention: false,
        audio_feed_forward: false,
        video_attends_to_audio: false,
        audio_attends_to_video: false,
        lora_rank: 16,
        lora_alpha: 16,
    },
    "motion": {
        video_self_attention: true,
        video_cross_attention: false,
        video_feed_forward: true,
        audio_self_attention: false,
        audio_cross_attention: false,
        audio_feed_forward: false,
        video_attends_to_audio: false,
        audio_attends_to_video: false,
        lora_rank: 16,
        lora_alpha: 16,
    },
    "subject + style": {
        video_self_attention: true,
        video_cross_attention: true,
        video_feed_forward: true,
        audio_self_attention: false,
        audio_cross_attention: false,
        audio_feed_forward: false,
        video_attends_to_audio: false,
        audio_attends_to_video: false,
        lora_rank: 32,
        lora_alpha: 32,
    },
    "all video": {
        video_self_attention: true,
        video_cross_attention: true,
        video_feed_forward: true,
        audio_self_attention: false,
        audio_cross_attention: false,
        audio_feed_forward: false,
        video_attends_to_audio: false,
        audio_attends_to_video: false,
        lora_rank: 32,
        lora_alpha: 32,
    },
    "audio + video": {
        video_self_attention: true,
        video_cross_attention: true,
        video_feed_forward: true,
        audio_self_attention: true,
        audio_cross_attention: true,
        audio_feed_forward: true,
        video_attends_to_audio: true,
        audio_attends_to_video: true,
        lora_rank: 32,
        lora_alpha: 32,
    },
};

function applyPreset(node, presetName) {
    const preset = PRESETS[presetName];
    if (!preset) return; // "custom" — don't touch anything

    for (const [key, value] of Object.entries(preset)) {
        const widget = node.widgets.find((w) => w.name === key);
        if (widget) {
            widget.value = value;
        }
    }
}

function createSectionHeader(label) {
    const bar = document.createElement("div");
    bar.style.cssText =
        "width:100%;display:flex;align-items:center;gap:6px;" +
        "padding:6px 4px 2px;box-sizing:border-box;";

    const line1 = document.createElement("div");
    line1.style.cssText = "flex:0 0 8px;height:1px;background:#666;";

    const text = document.createElement("span");
    text.textContent = label;
    text.style.cssText =
        "color:#aaa;font-size:11px;font-weight:bold;text-transform:uppercase;letter-spacing:0.5px;white-space:nowrap;";

    const line2 = document.createElement("div");
    line2.style.cssText = "flex:1;height:1px;background:#666;";

    bar.appendChild(line1);
    bar.appendChild(text);
    bar.appendChild(line2);
    return bar;
}

function addSections(node, sections) {
    for (const [label, firstWidget] of sections) {
        const target = node.widgets.find((w) => w.name === firstWidget);
        if (!target) continue;

        const header = node.addDOMWidget(
            "section_" + label.toLowerCase().replace(/\s+/g, "_"),
            "custom",
            createSectionHeader(label),
            { serialize: false }
        );

        const hdrIdx = node.widgets.indexOf(header);
        node.widgets.splice(hdrIdx, 1);
        const targetIdx = node.widgets.indexOf(target);
        node.widgets.splice(targetIdx, 0, header);
    }
}

// ---------------------------------------------------------------------------
// Live loss chart
// ---------------------------------------------------------------------------
const lossData = {};  // node_id -> { steps: [], losses: [], lrs: [] }

function createLossChart() {
    const container = document.createElement("div");
    container.style.cssText = "width:100%;padding:4px;box-sizing:border-box;";

    const canvas = document.createElement("canvas");
    canvas.width = 400;
    canvas.height = 180;
    canvas.style.cssText = "width:100%;height:180px;background:#1a1a1a;border-radius:4px;";
    container.appendChild(canvas);

    return { container, canvas };
}

function drawLossChart(canvas, data) {
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    const pad = { top: 20, right: 12, bottom: 28, left: 52 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, W, H);

    if (!data || data.steps.length < 2) {
        ctx.fillStyle = "#666";
        ctx.font = "12px monospace";
        ctx.textAlign = "center";
        ctx.fillText(
            data && data.steps.length === 1
                ? "Waiting for more data..."
                : "Training loss will appear here",
            W / 2, H / 2
        );
        return;
    }

    const steps = data.steps;
    const losses = data.losses;
    const minStep = steps[0];
    const maxStep = steps[steps.length - 1];
    const stepRange = maxStep - minStep || 1;

    // Use EMA-smoothed losses for display, raw for min/max
    const smoothed = [];
    const alpha = Math.min(0.95, 1 - 2 / (Math.max(losses.length / 5, 2) + 1));
    smoothed[0] = losses[0];
    for (let i = 1; i < losses.length; i++) {
        smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * losses[i];
    }

    // Y range with padding
    let minLoss = Infinity, maxLoss = -Infinity;
    for (const v of smoothed) {
        if (v < minLoss) minLoss = v;
        if (v > maxLoss) maxLoss = v;
    }
    const lossRange = maxLoss - minLoss || 0.01;
    minLoss -= lossRange * 0.05;
    maxLoss += lossRange * 0.05;
    const yRange = maxLoss - minLoss;

    // Grid lines
    ctx.strokeStyle = "#333";
    ctx.lineWidth = 0.5;
    const numGridY = 4;
    ctx.font = "9px monospace";
    ctx.fillStyle = "#888";
    ctx.textAlign = "right";
    for (let i = 0; i <= numGridY; i++) {
        const y = pad.top + (plotH * i) / numGridY;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(pad.left + plotW, y);
        ctx.stroke();
        const val = maxLoss - (yRange * i) / numGridY;
        ctx.fillText(val.toFixed(3), pad.left - 4, y + 3);
    }

    // X axis labels
    ctx.textAlign = "center";
    ctx.fillStyle = "#888";
    const numGridX = 4;
    for (let i = 0; i <= numGridX; i++) {
        const x = pad.left + (plotW * i) / numGridX;
        const stepVal = minStep + (stepRange * i) / numGridX;
        ctx.fillText(Math.round(stepVal).toString(), x, H - pad.bottom + 14);
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, pad.top + plotH);
        ctx.stroke();
    }

    // Plot raw losses as faint dots
    ctx.fillStyle = "rgba(100, 160, 255, 0.15)";
    for (let i = 0; i < steps.length; i++) {
        const x = pad.left + ((steps[i] - minStep) / stepRange) * plotW;
        const y = pad.top + ((maxLoss - losses[i]) / yRange) * plotH;
        ctx.beginPath();
        ctx.arc(x, y, 1.2, 0, Math.PI * 2);
        ctx.fill();
    }

    // Plot smoothed line
    ctx.strokeStyle = "#4a9eff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < steps.length; i++) {
        const x = pad.left + ((steps[i] - minStep) / stepRange) * plotW;
        const y = pad.top + ((maxLoss - smoothed[i]) / yRange) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Linear trend line (least-squares regression on smoothed values)
    if (steps.length >= 10) {
        let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
        const n = steps.length;
        for (let i = 0; i < n; i++) {
            sumX += steps[i];
            sumY += smoothed[i];
            sumXY += steps[i] * smoothed[i];
            sumXX += steps[i] * steps[i];
        }
        const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;

        const x0 = pad.left;
        const x1 = pad.left + plotW;
        const y0 = pad.top + ((maxLoss - (slope * minStep + intercept)) / yRange) * plotH;
        const y1 = pad.top + ((maxLoss - (slope * maxStep + intercept)) / yRange) * plotH;

        ctx.strokeStyle = slope < 0 ? "rgba(80, 220, 120, 0.6)" : "rgba(255, 100, 80, 0.6)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Title
    ctx.fillStyle = "#ccc";
    ctx.font = "10px monospace";
    ctx.textAlign = "left";
    const lastLoss = losses[losses.length - 1];
    const lastSmooth = smoothed[smoothed.length - 1];
    ctx.fillText(
        `Step ${maxStep}/${data.totalSteps || "?"}  loss=${lastLoss.toFixed(4)}  smooth=${lastSmooth.toFixed(4)}`,
        pad.left, pad.top - 6
    );
}

// Listen for training updates from the backend
api.addEventListener("rs-training-update", (event) => {
    const { node_id, step, total_steps, loss, lr } = event.detail;
    if (!node_id) return;

    if (!lossData[node_id]) {
        lossData[node_id] = { steps: [], losses: [], lrs: [], totalSteps: total_steps };
    }
    const d = lossData[node_id];
    d.totalSteps = total_steps;
    d.steps.push(step);
    d.losses.push(loss);
    d.lrs.push(lr);

    // Redraw chart if the node has one
    const node = app.graph.getNodeById(Number(node_id));
    if (node && node._lossCanvas) {
        drawLossChart(node._lossCanvas, d);
    }
});

app.registerExtension({
    name: "rs-nodes.LTXVTrain",

    nodeCreated(node) {
        if (node.comfyClass === "RSLTXVPrepareDataset") {
            addSections(node, PREPARE_SECTIONS);
        } else if (node.comfyClass === "RSLTXVTrainLoRA") {
            addSections(node, TRAIN_SECTIONS);

            // Wire up preset dropdown to apply settings on change
            const presetWidget = node.widgets.find((w) => w.name === "preset");
            if (presetWidget) {
                const origCallback = presetWidget.callback;
                presetWidget.callback = function (value) {
                    if (origCallback) origCallback.call(this, value);
                    applyPreset(node, value);
                };
                // Apply initial preset
                applyPreset(node, presetWidget.value);
            }

            // Add live loss chart widget at the bottom
            const { container, canvas } = createLossChart();
            const chartWidget = node.addDOMWidget(
                "loss_chart", "custom", container,
                { serialize: false, hideOnZoom: false }
            );
            node._lossCanvas = canvas;
            node._lossChartWidget = chartWidget;

            // Add spacer widgets below the chart so the node is tall enough.
            // ComfyUI sizes the node based on widget count, not DOM height,
            // so the canvas overflows without extra slots to claim the space.
            for (let i = 0; i < 3; i++) {
                const spacer = document.createElement("div");
                spacer.style.cssText = "width:100%;height:1px;";
                node.addDOMWidget(
                    "chart_spacer_" + i, "custom", spacer,
                    { serialize: false }
                );
            }

            node.size[0] = Math.max(node.size[0] || 0, 480);

            // Initial draw (placeholder)
            requestAnimationFrame(() => {
                // Match canvas resolution to display size
                const rect = canvas.getBoundingClientRect();
                if (rect.width > 0) {
                    canvas.width = Math.round(rect.width * (window.devicePixelRatio || 1));
                    canvas.height = Math.round(180 * (window.devicePixelRatio || 1));
                }
                drawLossChart(canvas, lossData[node.id]);
            });
        }
    },
});
