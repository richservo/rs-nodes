import { app } from "../../scripts/app.js";

const MAX_CLIPS = 20;

function getClipIndex(name) {
    const m = name ? name.match(/(\d+)$/) : null;
    return m ? parseInt(m[1], 10) : 0;
}

app.registerExtension({
    name: "rs-nodes.AudioConcat",

    nodeCreated(node) {
        if (node.comfyClass !== "RSAudioConcat") return;

        const clipCountWidget = node.widgets.find((w) => w.name === "clip_count");
        if (!clipCountWidget) return;

        // --- Upload button ---
        const uploadContainer = document.createElement("div");
        uploadContainer.style.cssText =
            "width:100%;padding:4px;box-sizing:border-box;";

        const uploadBtn = document.createElement("button");
        uploadBtn.textContent = "Upload Audio Clips";
        uploadBtn.style.cssText =
            "width:100%;padding:6px 12px;cursor:pointer;border-radius:4px;" +
            "border:1px solid #666;background:#333;color:#ddd;font-size:13px;";

        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.multiple = true;
        fileInput.accept = "audio/*,.wav,.mp3,.flac,.ogg,.m4a,.aac";
        fileInput.style.display = "none";

        uploadBtn.onclick = () => fileInput.click();

        fileInput.onchange = async () => {
            if (!fileInput.files.length) return;
            uploadBtn.textContent = "Uploading...";
            uploadBtn.disabled = true;

            for (const file of fileInput.files) {
                const formData = new FormData();
                formData.append("image", file);
                formData.append("type", "input");
                formData.append("overwrite", "true");

                try {
                    const resp = await fetch("/upload/image", {
                        method: "POST",
                        body: formData,
                    });
                    if (resp.ok) {
                        const result = await resp.json();
                        const name = result.name;
                        // Add the new file to every file combo widget's options.
                        if (allWidgets) {
                            for (const w of allWidgets) {
                                if (
                                    w.name &&
                                    w.name.startsWith("audio_file_") &&
                                    w.options &&
                                    w.options.values
                                ) {
                                    if (!w.options.values.includes(name)) {
                                        w.options.values.push(name);
                                        w.options.values.sort();
                                    }
                                }
                            }
                        }
                    }
                } catch (e) {
                    console.error("Upload failed:", e);
                }
            }

            uploadBtn.textContent = "Upload Audio Clips";
            uploadBtn.disabled = false;
            fileInput.value = "";
        };

        uploadContainer.appendChild(fileInput);
        uploadContainer.appendChild(uploadBtn);

        node.addDOMWidget("upload_audio", "custom", uploadContainer, {
            serialize: false,
        });

        // --- Per-clip audio preview elements ---
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
            previews[i] = {
                widget: domWidget,
                audioEl: audioEl,
                container: container,
            };

            // Move preview to just before pause_after_i.
            const pauseWidget = node.widgets.find(
                (w) => w.name === "pause_after_" + i
            );
            if (pauseWidget) {
                const prevIdx = node.widgets.indexOf(domWidget);
                node.widgets.splice(prevIdx, 1);
                const newPauseIdx = node.widgets.indexOf(pauseWidget);
                node.widgets.splice(newPauseIdx, 0, domWidget);
            }
        }

        // --- Wire file widget changes to preview sources ---
        for (let i = 1; i <= MAX_CLIPS; i++) {
            const fileWidget = node.widgets.find(
                (w) => w.name === "audio_file_" + i
            );
            if (!fileWidget || !previews[i]) continue;

            const updateSource = () => {
                const name = fileWidget.value;
                const audioEl = previews[i].audioEl;
                if (name && name !== "none") {
                    const url =
                        "/view?filename=" +
                        encodeURIComponent(name) +
                        "&type=input";
                    if (audioEl.getAttribute("src") !== url) {
                        audioEl.src = url;
                    }
                } else {
                    audioEl.removeAttribute("src");
                    audioEl.load();
                }
            };

            const origCb = fileWidget.callback;
            fileWidget.callback = function (value) {
                if (origCb) origCb.call(this, value);
                updateSource();
            };

            updateSource();
        }

        // --- Visibility: rebuild widget list based on clip_count ---
        let allWidgets = null;

        function updateVisibility() {
            const count = clipCountWidget.value;

            if (!allWidgets) {
                allWidgets = [...node.widgets];
            }

            const visible = [];
            for (const w of allWidgets) {
                const idx = getClipIndex(w.name);
                if (idx === 0) {
                    // Non-clip widget (clip_count, sample_rate, upload) — always visible.
                    visible.push(w);
                    if (w.element) w.element.style.display = "";
                } else if (idx <= count) {
                    visible.push(w);
                    if (w.element) w.element.style.display = "";
                } else {
                    if (w.element) w.element.style.display = "none";
                }
            }

            node.widgets.length = 0;
            for (const w of visible) {
                node.widgets.push(w);
            }

            const sz = node.computeSize();
            node.setSize([Math.max(sz[0], node.size[0]), sz[1]]);
            node.setDirtyCanvas(true, true);
        }

        const origClipCb = clipCountWidget.callback;
        clipCountWidget.callback = function (value) {
            if (origClipCb) origClipCb.call(this, value);
            updateVisibility();
        };

        requestAnimationFrame(() => {
            updateVisibility();
        });
    },
});
