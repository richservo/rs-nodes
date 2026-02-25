import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "rs-nodes.AudioSave",

    nodeCreated(node) {
        if (node.comfyClass !== "RSAudioSave") return;

        const container = document.createElement("div");
        container.style.cssText =
            "width:calc(100% - 16px);padding:4px;box-sizing:border-box;margin:0 auto;";

        const audioEl = document.createElement("audio");
        audioEl.controls = true;
        audioEl.preload = "metadata";
        audioEl.style.cssText = "width:100%;height:32px;";
        container.appendChild(audioEl);

        const label = document.createElement("div");
        label.style.cssText =
            "color:#999;font-size:11px;text-align:center;padding-top:2px;";
        label.textContent = "No audio yet";
        container.appendChild(label);

        node.addDOMWidget("audio_preview", "custom", container, {
            serialize: false,
            getMinHeight: () => 54,
        });

        const origOnExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            if (origOnExecuted) origOnExecuted.call(this, output);

            if (output && output.audio && output.audio.length > 0) {
                const info = output.audio[0];
                const url =
                    "/view?filename=" +
                    encodeURIComponent(info.filename) +
                    "&type=" +
                    encodeURIComponent(info.type || "input") +
                    (info.subfolder
                        ? "&subfolder=" + encodeURIComponent(info.subfolder)
                        : "") +
                    "&t=" + Date.now();
                audioEl.src = url;
                label.textContent = info.filename;
            }
        };
    },
});
