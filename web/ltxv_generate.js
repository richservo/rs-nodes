import { app } from "../../scripts/app.js";

// Section definitions: [label, first_widget_name]
const GENERATE_SECTIONS = [
    ["Generation", "width"],
    ["Frame Injection", "first_strength"],
    ["Audio", "audio_cfg"],
    ["Guidance", "stg_scale"],
    ["Efficiency", "attention_mode"],
    ["Upscale", "upscale"],
    ["Output", "decode"],
    ["Scheduler", "max_shift"],
];

const EXTEND_SECTIONS = [
    ["Extension", "num_new_frames"],
    ["Frame Injection", "last_strength"],
    ["Efficiency", "attention_mode"],
    ["Upscale", "upscale"],
    ["Output", "decode"],
    ["Scheduler", "max_shift"],
];

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

        // Move header before the target widget
        const hdrIdx = node.widgets.indexOf(header);
        node.widgets.splice(hdrIdx, 1);
        const targetIdx = node.widgets.indexOf(target);
        node.widgets.splice(targetIdx, 0, header);
    }
}

app.registerExtension({
    name: "rs-nodes.LTXVGenerate",

    nodeCreated(node) {
        if (node.comfyClass === "RSLTXVGenerate") {
            addSections(node, GENERATE_SECTIONS);
        } else if (node.comfyClass === "RSLTXVExtend") {
            addSections(node, EXTEND_SECTIONS);
        }
    },
});
