import { app } from "../../scripts/app.js";

const MAX_CLIPS = 20;

// Widgets only visible when run_inference is on (mode is always visible)
const INFERENCE_WIDGETS = new Set([
    "language",
    "seed",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "max_new_tokens",
    "head_handle",
    "tail_handle",
]);

function getClipIndex(name) {
    const m = name ? name.match(/(\d+)$/) : null;
    return m ? parseInt(m[1], 10) : 0;
}

app.registerExtension({
    name: "rs-nodes.MossTTSSave",

    nodeCreated(node) {
        if (node.comfyClass !== "RSMossTTSSave") return;

        // Track the current clip count (updated after execution)
        let currentClipCount = 0;

        // --- Per-clip header labels ---
        const headers = {};
        for (let i = MAX_CLIPS; i >= 1; i--) {
            const bar = document.createElement("div");
            bar.style.cssText =
                "width:100%;display:flex;align-items:center;gap:4px;" +
                "padding:4px;box-sizing:border-box;border-bottom:1px solid #555;";

            const label = document.createElement("span");
            label.textContent = "Clip " + i;
            label.style.cssText =
                "flex:1;color:#ccc;font-size:12px;font-weight:bold;";

            bar.appendChild(label);

            const headerWidget = node.addDOMWidget(
                "clip_header_" + i,
                "custom",
                bar,
                { serialize: false }
            );
            headers[i] = { widget: headerWidget, label: label };

            // Position header before the first control widget for this clip
            const firstControl =
                i === 1
                    ? node.widgets.find((w) => w.name === "pause_before_1")
                    : node.widgets.find((w) => w.name === "start_time_" + i);
            if (firstControl) {
                const hdrIdx = node.widgets.indexOf(headerWidget);
                node.widgets.splice(hdrIdx, 1);
                const ctrlIdx = node.widgets.indexOf(firstControl);
                node.widgets.splice(ctrlIdx, 0, headerWidget);
            }
        }

        // --- Per-clip inline audio previews (between header and start_time) ---
        const previews = {};
        for (let i = MAX_CLIPS; i >= 1; i--) {
            const container = document.createElement("div");
            container.style.cssText =
                "width:100%;padding:0 4px;box-sizing:border-box;";

            const audioEl = document.createElement("audio");
            audioEl.controls = true;
            audioEl.preload = "metadata";
            audioEl.style.cssText = "width:100%;height:32px;";
            container.appendChild(audioEl);

            const domWidget = node.addDOMWidget(
                "audio_preview_" + i,
                "custom",
                container,
                { serialize: false }
            );
            previews[i] = { widget: domWidget, audioEl: audioEl };

            // Position preview before the first control widget (after the header)
            const firstCtrl =
                i === 1
                    ? node.widgets.find((w) => w.name === "pause_before_1")
                    : node.widgets.find((w) => w.name === "start_time_" + i);
            if (firstCtrl) {
                const prevIdx = node.widgets.indexOf(domWidget);
                node.widgets.splice(prevIdx, 1);
                const ctrlIdx = node.widgets.indexOf(firstCtrl);
                node.widgets.splice(ctrlIdx, 0, domWidget);
            }
        }

        // --- Full Clip preview (always at the bottom) ---
        const fullClipBar = document.createElement("div");
        fullClipBar.style.cssText =
            "width:100%;display:flex;align-items:center;gap:4px;" +
            "padding:4px;box-sizing:border-box;border-bottom:1px solid #555;";
        const fullClipLabel = document.createElement("span");
        fullClipLabel.textContent = "Full Clip";
        fullClipLabel.style.cssText =
            "flex:1;color:#ccc;font-size:12px;font-weight:bold;";
        fullClipBar.appendChild(fullClipLabel);

        node.addDOMWidget("full_clip_header", "custom", fullClipBar, {
            serialize: false,
        });

        const fullClipContainer = document.createElement("div");
        fullClipContainer.style.cssText =
            "width:100%;padding:0 4px;box-sizing:border-box;";
        const fullClipAudio = document.createElement("audio");
        fullClipAudio.controls = true;
        fullClipAudio.preload = "metadata";
        fullClipAudio.style.cssText = "width:100%;height:32px;";
        fullClipContainer.appendChild(fullClipAudio);

        node.addDOMWidget("full_clip_preview", "custom", fullClipContainer, {
            serialize: false,
        });

        // --- Snapshot all widgets for visibility management ---
        let snapshotDone = false;

        function hideWidget(w) {
            if (w.element) {
                w.element.hidden = true;
                w.element.style.display = "none";
            }
            if (!w._origType) w._origType = w.type;
            w.type = "hidden";
            w.hidden = true;
            w.computeSize = () => [0, -4];
        }

        function showWidget(w) {
            if (w.element) {
                w.element.hidden = false;
                w.element.style.display = "";
            }
            if (w._origType) {
                w.type = w._origType;
                delete w._origType;
            }
            w.hidden = false;
            delete w.computeSize;
        }

        function updateVisibility() {
            if (!snapshotDone) {
                snapshotDone = true;
            }

            const runInferenceWidget = node.widgets.find(
                (w) => w.name === "run_inference"
            );
            const inferencing = runInferenceWidget
                ? runInferenceWidget.value
                : true;

            const modeWidget = node.widgets.find((w) => w.name === "mode");
            const currentMode = modeWidget ? modeWidget.value : "all";

            for (const w of node.widgets) {
                const clipIdx = getClipIndex(w.name);
                const isClipWidget =
                    clipIdx > 0 &&
                    (w.name.startsWith("start_time_") ||
                        w.name.startsWith("end_time_") ||
                        w.name.startsWith("pause_after_") ||
                        w.name.startsWith("pause_before_") ||
                        w.name.startsWith("clip_header_") ||
                        w.name.startsWith("audio_preview_"));

                let show = true;
                if (isClipWidget) {
                    show = clipIdx <= currentClipCount;
                } else if (w.name === "select_index") {
                    show = inferencing && currentMode === "single";
                } else if (INFERENCE_WIDGETS.has(w.name)) {
                    show = inferencing;
                }

                if (show) {
                    showWidget(w);
                } else {
                    hideWidget(w);
                }
            }

            const sz = node.computeSize();
            node.setSize([Math.max(sz[0], node.size[0], 350), sz[1]]);
            node.setDirtyCanvas(true, true);
        }

        // --- Wire run_inference toggle ---
        const runInferenceWidget = node.widgets.find(
            (w) => w.name === "run_inference"
        );
        if (runInferenceWidget) {
            const origCb = runInferenceWidget.callback;
            runInferenceWidget.callback = function (value) {
                if (origCb) origCb.call(this, value);
                updateVisibility();
            };
        }

        // --- Wire mode toggle ---
        const modeWidget = node.widgets.find((w) => w.name === "mode");
        if (modeWidget) {
            const origModeCb = modeWidget.callback;
            modeWidget.callback = function (value) {
                if (origModeCb) origModeCb.call(this, value);
                updateVisibility();
            };
        }

        // --- onExecuted: update inline previews + clip count ---
        const origOnExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            if (origOnExecuted) origOnExecuted.call(this, output);

            // Update clip count from execution result
            if (output && output.clip_count && output.clip_count[0] != null) {
                currentClipCount = output.clip_count[0];
            }

            // Update clip header labels with dialogue text
            if (output && output.clip_labels) {
                for (let i = 1; i <= MAX_CLIPS; i++) {
                    if (!headers[i]) continue;
                    const text = output.clip_labels[i - 1];
                    if (text) {
                        const truncated = text.length > 40 ? text.slice(0, 40) + "\u2026" : text;
                        headers[i].label.textContent = "Clip " + i + " \u2014 " + truncated;
                    } else {
                        headers[i].label.textContent = "Clip " + i;
                    }
                }
            }

            // Update per-clip inline audio sources
            if (output && output.audio) {
                for (let i = 1; i <= MAX_CLIPS; i++) {
                    if (!previews[i]) continue;
                    const audioEl = previews[i].audioEl;
                    const info = output.audio[i - 1];

                    if (info) {
                        const url =
                            "/view?filename=" +
                            encodeURIComponent(info.filename) +
                            "&type=" +
                            encodeURIComponent(info.type || "input") +
                            (info.subfolder
                                ? "&subfolder=" +
                                  encodeURIComponent(info.subfolder)
                                : "") +
                            "&t=" +
                            Date.now();
                        audioEl.src = url;
                    } else {
                        audioEl.removeAttribute("src");
                        audioEl.load();
                    }
                }
            }

            // Update full clip preview
            if (output && output.full_audio && output.full_audio[0]) {
                const info = output.full_audio[0];
                const url =
                    "/view?filename=" +
                    encodeURIComponent(info.filename) +
                    "&type=" +
                    encodeURIComponent(info.type || "input") +
                    (info.subfolder
                        ? "&subfolder=" + encodeURIComponent(info.subfolder)
                        : "") +
                    "&t=" +
                    Date.now();
                fullClipAudio.src = url;
            }

            updateVisibility();

            requestAnimationFrame(() => {
                const sz = node.computeSize();
                node.setSize([Math.max(sz[0], node.size[0], 350), sz[1]]);
                node.setDirtyCanvas(true, true);
            });
        };

        // --- Initial visibility (hide all clip widgets, they appear after first run) ---
        requestAnimationFrame(() => {
            updateVisibility();
            if (node.size[0] < 350) {
                node.setSize([350, node.size[1]]);
            }
        });
    },
});
