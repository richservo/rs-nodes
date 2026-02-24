import { app } from "../../scripts/app.js";

const MAX_CLIPS = 20;
const CLIP_FIELDS = ["audio_file", "start_time", "end_time", "pause_after"];

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

        // Helpers to get/set all values for a clip slot.
        function getClipValues(idx) {
            const vals = {};
            for (const field of CLIP_FIELDS) {
                const w = allWidgets
                    ? allWidgets.find((x) => x.name === field + "_" + idx)
                    : null;
                vals[field] = w ? w.value : field === "audio_file" ? "none" : 0.0;
            }
            return vals;
        }

        function setClipValues(idx, vals) {
            if (!allWidgets) return;
            for (const field of CLIP_FIELDS) {
                const w = allWidgets.find((x) => x.name === field + "_" + idx);
                if (w) {
                    w.value = vals[field];
                    if (w.callback) w.callback(vals[field]);
                }
            }
        }

        function defaultClipValues() {
            return {
                audio_file: "none",
                start_time: 0.0,
                end_time: 0.0,
                pause_after: 0.0,
            };
        }

        // Swap two clip slots.
        function swapClips(a, b) {
            const valsA = getClipValues(a);
            const valsB = getClipValues(b);
            setClipValues(a, valsB);
            setClipValues(b, valsA);
        }

        // Remove clip at index and shift everything after it up.
        function removeClip(idx) {
            const count = clipCountWidget.value;
            for (let i = idx; i < count; i++) {
                setClipValues(i, getClipValues(i + 1));
            }
            setClipValues(count, defaultClipValues());

            if (count > 1) {
                clipCountWidget.value = count - 1;
                if (clipCountWidget.callback)
                    clipCountWidget.callback(count - 1);
            }
            updateLabels();
        }

        // Update clip header labels after reorder.
        function updateLabels() {
            for (let i = 1; i <= MAX_CLIPS; i++) {
                if (headers[i] && headers[i].label) {
                    headers[i].label.textContent = "Clip " + i;
                }
            }
        }

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

            let nextSlot = 1;
            if (allWidgets) {
                for (let i = 1; i <= MAX_CLIPS; i++) {
                    const fw = allWidgets.find(
                        (w) => w.name === "audio_file_" + i
                    );
                    if (fw && fw.value && fw.value !== "none") {
                        nextSlot = i + 1;
                    }
                }
            }

            const uploadedNames = [];

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
                        uploadedNames.push(result.name);

                        if (allWidgets) {
                            for (const w of allWidgets) {
                                if (
                                    w.name &&
                                    w.name.startsWith("audio_file_") &&
                                    w.options &&
                                    w.options.values
                                ) {
                                    if (!w.options.values.includes(result.name)) {
                                        w.options.values.push(result.name);
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

            for (const name of uploadedNames) {
                if (nextSlot > MAX_CLIPS) break;
                const fw = allWidgets
                    ? allWidgets.find((w) => w.name === "audio_file_" + nextSlot)
                    : null;
                if (fw) {
                    fw.value = name;
                    if (fw.callback) fw.callback(name);
                }
                nextSlot++;
            }

            const newCount = Math.min(nextSlot - 1, MAX_CLIPS);
            if (newCount > clipCountWidget.value) {
                clipCountWidget.value = newCount;
                if (clipCountWidget.callback)
                    clipCountWidget.callback(newCount);
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

        // --- Per-clip header bars (label, move up/down, remove) ---
        const btnStyle =
            "padding:2px 6px;cursor:pointer;border-radius:3px;" +
            "border:1px solid #555;background:#444;color:#ddd;font-size:12px;" +
            "min-width:24px;text-align:center;";

        const headers = {};
        for (let i = MAX_CLIPS; i >= 1; i--) {
            const bar = document.createElement("div");
            bar.style.cssText =
                "width:100%;display:flex;align-items:center;gap:4px;" +
                "padding:4px;box-sizing:border-box;border-bottom:1px solid #555;";

            const label = document.createElement("span");
            label.textContent = "Clip " + i;
            label.style.cssText = "flex:1;color:#ccc;font-size:12px;font-weight:bold;";

            const upBtn = document.createElement("button");
            upBtn.textContent = "\u25B2";
            upBtn.title = "Move up";
            upBtn.style.cssText = btnStyle;

            const downBtn = document.createElement("button");
            downBtn.textContent = "\u25BC";
            downBtn.title = "Move down";
            downBtn.style.cssText = btnStyle;

            const removeBtn = document.createElement("button");
            removeBtn.textContent = "\u2715";
            removeBtn.title = "Remove clip";
            removeBtn.style.cssText =
                btnStyle + "background:#633;border-color:#855;";

            const clipIdx = i;
            upBtn.onclick = () => {
                if (clipIdx <= 1) return;
                swapClips(clipIdx, clipIdx - 1);
            };
            downBtn.onclick = () => {
                if (clipIdx >= clipCountWidget.value) return;
                swapClips(clipIdx, clipIdx + 1);
            };
            removeBtn.onclick = () => {
                removeClip(clipIdx);
            };

            bar.appendChild(label);
            bar.appendChild(upBtn);
            bar.appendChild(downBtn);
            bar.appendChild(removeBtn);

            const headerWidget = node.addDOMWidget(
                "clip_header_" + i,
                "custom",
                bar,
                { serialize: false }
            );
            headers[i] = { widget: headerWidget, label: label };

            // Insert header just before audio_file_i.
            const fileWidget = node.widgets.find(
                (w) => w.name === "audio_file_" + i
            );
            if (fileWidget) {
                const hdrIdx = node.widgets.indexOf(headerWidget);
                node.widgets.splice(hdrIdx, 1);
                const fileIdx = node.widgets.indexOf(fileWidget);
                node.widgets.splice(fileIdx, 0, headerWidget);
            }
        }

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

            // Move preview to just before start_time_i.
            const startWidget = node.widgets.find(
                (w) => w.name === "start_time_" + i
            );
            if (startWidget) {
                const prevIdx = node.widgets.indexOf(domWidget);
                node.widgets.splice(prevIdx, 1);
                const startIdx = node.widgets.indexOf(startWidget);
                node.widgets.splice(startIdx, 0, domWidget);
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
