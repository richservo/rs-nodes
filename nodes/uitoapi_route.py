# Server-side UI workflow → API prompt converter for rs-studio.
#
# ComfyUI's frontend has graphToPrompt() in JavaScript that converts a
# UI-format workflow (the one "Save" produces) into the API prompt
# /prompt expects. There's no equivalent on the Python side, so any
# external tool that wants to drive runs from a UI workflow has to
# reimplement the conversion — and stay in sync with every quirk of
# ComfyUI's widget system.
#
# This endpoint runs the conversion ON the pod, where it has direct
# access to NODE_CLASS_MAPPINGS. That means the canonical widget
# ordering for any installed node is authoritative — no positional
# guessing from the workflow JSON, no need for clients to hardcode
# auto-widget patterns like control_after_generate.
#
# POST /rs/uitoapi
#   request body: the raw UI workflow JSON (the same shape as a
#                 saved .json from ComfyUI's File → Save menu)
#   response:     { "prompt": <api prompt>, "warnings": [...] }
#
# Backward-compatible: rs-studio falls back to its local converter
# when this endpoint isn't available (older rs-nodes on the pod).

import os

from aiohttp import web

import server
import nodes

# Per-node widget-walk diagnostics are noisy (one multi-kB line per
# node per submission). Gate behind RS_UITOAPI_DEBUG=1 so day-to-day
# runs stay quiet; flip the env var when actively debugging the
# UI→API conversion.
_DEBUG = os.environ.get("RS_UITOAPI_DEBUG", "0") == "1"

# Bumped whenever the conversion logic changes — printed on import so
# the operator can confirm which version Comfy actually loaded.
_VERSION = "3"
print(f"[rs/uitoapi] version {_VERSION} loaded (debug={_DEBUG})")


# Frontend-only nodes that have no class_type and shouldn't appear
# in the API prompt. Mirrors workflow-converter.ts's SKIP_TYPES list.
SKIP_TYPES = {
    "Note",
    "PrimitiveNode",
    "Reroute",
    "GroupHeader",
    "MarkdownNote",
}

# Connection-typed inputs — never widgets, value comes from a link.
CONNECTION_TYPES = {
    "MODEL",
    "CLIP",
    "VAE",
    "CONDITIONING",
    "LATENT",
    "IMAGE",
    "MASK",
    "CONTROL_NET",
    "STYLE_MODEL",
    "CLIP_VISION",
    "CLIP_VISION_OUTPUT",
    "GLIGEN",
    "SAMPLER",
    "SIGMAS",
    "NOISE",
    "GUIDER",
    "AUDIO",
    "VIDEO",
}


# Primitive widget types in ComfyUI. Anything OUTSIDE this set that's
# also an uppercase string is treated as a connection. This is the
# positive-allowlist pattern Seth Robinson's converter uses (and the
# one ComfyUI's own frontend uses) — necessary to handle custom
# connection types like DA3_MODEL / FLUX_GUIDER that aren't in
# CONNECTION_TYPES but also aren't widgets.
WIDGET_PRIMITIVES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}

def _is_widget_type(declared):
    """True when the declared type is a widget (consumes a slot in
    widgets_values), False when it's a connection-only input.

    Uses a positive allowlist instead of a CONNECTION_TYPES blocklist,
    because custom node connection types (DA3_MODEL, FLUX_GUIDER,
    etc.) aren't in any standard blocklist. Comfy's own widget vs
    connection distinction comes down to:

      - list / tuple of strings           → legacy combo widget
      - "INT" / "FLOAT" / "STRING" / etc. → primitive widget
      - "COMBO"                            → v3 combo widget
      - "COMFY_*COMBO*"                    → v3 dynamic combo
      - any other lowercase string         → custom widget type
      - any other UPPERCASE string         → connection (skip)

    The widget_source markers ("__WIDGET__" / "__CONNECTION__")
    used by the v3-schema path bypass this and answer directly.
    """
    if declared == "__WIDGET__":
        return True
    if declared == "__CONNECTION__":
        return False
    if isinstance(declared, (list, tuple)):
        return True
    if isinstance(declared, str):
        if declared in WIDGET_PRIMITIVES:
            return True
        if declared.startswith("COMFY_") and "COMBO" in declared:
            return True
        # Lowercase custom widget types (rare but allowed by Comfy).
        if not declared.isupper():
            return True
        return False
    return False


def _has_auto_control_after_generate(input_name):
    """ComfyUI's frontend auto-injects a control_after_generate combo
    widget after any widget named seed / noise_seed. The auto-widget is
    NOT in the node class's INPUT_TYPES but IS serialized into
    widgets_values, so the cursor must skip past it."""
    return input_name in ("seed", "noise_seed")


def _get_v3_schema(node_class):
    """Return the v3-IO schema for a node class, or None for v2 nodes
    (or v3 nodes whose schema can't be loaded)."""
    schema = getattr(node_class, "SCHEMA", None)
    if schema is not None:
        return schema
    get_schema = getattr(node_class, "GET_SCHEMA", None) or getattr(
        node_class, "FINALIZE_SCHEMA", None
    )
    if get_schema is None:
        return None
    try:
        return get_schema()
    except Exception:
        return None


def _v3_input_is_widget(inp):
    """True when a v3 IO Input is a widget (has a widget_type, or is
    a WidgetInput subclass). Connection inputs (IMAGE, MASK, MODEL,
    custom types like DA3_MODEL) return False — they don't consume a
    widgets_values slot.

    The distinguishing trait: WidgetInput subclasses define a `default`
    attribute set in __init__ (lines 198-201 in comfy_api/latest/_io.py).
    Plain Input doesn't. We use that as a duck-type check so this works
    regardless of which exact subclass is in use (FloatInput, ComboInput,
    BooleanInput, custom widget input subclasses, etc.).
    """
    # WidgetInput has these attrs set in __init__; plain Input doesn't.
    return hasattr(inp, "widget_type") or hasattr(inp, "default")


def _canonical_input_order(node_class):
    """Return [(name, declared_type), ...] in the canonical order
    ComfyUI's frontend uses to lay out widgets / connections.

    For v3-IO nodes, the schema's `inputs` list is in user-declared
    order — that's what the frontend uses to build the widgets array,
    and therefore the order widgets_values is serialized in.
    INPUT_TYPES() re-groups required/optional, which corrupts that
    order whenever required and optional aren't strictly contiguous
    in declaration order. So we read the schema directly when
    available and fall back to INPUT_TYPES() only for v2 nodes.

    Each entry's declared_type is either:
      - the literal string "__WIDGET__" — meaning the cursor walker
        should consume a widgets_values slot for this input.
      - the literal string "__CONNECTION__" — connection-only, no
        widget value.
      - or a raw type string from INPUT_TYPES (for the v2 fallback).
    The walker uses _is_widget_type on the type to decide, so any of
    these work, but the explicit __WIDGET__/__CONNECTION__ tokens
    skip the string-based classification which can't tell custom
    connection types (DA3_MODEL, FLUX_GUIDER, etc.) from widgets.
    """
    schema = _get_v3_schema(node_class)
    schema_inputs = getattr(schema, "inputs", None) if schema is not None else None
    if isinstance(schema_inputs, list) and schema_inputs:
        order = []
        for inp in schema_inputs:
            name = getattr(inp, "id", None)
            if not name:
                continue
            order.append((name, "__WIDGET__" if _v3_input_is_widget(inp) else "__CONNECTION__"))
        return order

    # v2 / legacy path: pull from INPUT_TYPES(). Required first, then
    # optional — matches how the legacy frontend laid widgets out.
    try:
        spec = node_class.INPUT_TYPES()
    except Exception:
        return None
    order = []
    for section in ("required", "optional"):
        section_dict = spec.get(section, {}) or {}
        for name, decl in section_dict.items():
            declared_type = decl[0] if isinstance(decl, (list, tuple)) and decl else None
            order.append((name, declared_type))
    return order


def _index_links_by_dest(workflow):
    """Map (dest_node_id, dest_slot) -> (src_node_id, src_slot)."""
    by_dest = {}
    for link in workflow.get("links", []) or []:
        if not isinstance(link, (list, tuple)) or len(link) < 6:
            continue
        # ComfyUI link tuple: [link_id, src_node, src_slot, dst_node, dst_slot, type]
        _, src_node, src_slot, dst_node, dst_slot, _ = link[:6]
        by_dest[(dst_node, dst_slot)] = (src_node, src_slot)
    return by_dest


def _convert_node(node, link_by_dest, warnings):
    """Convert one workflow node to its API prompt entry. Returns
    (key, value) or None when the node should be omitted."""
    node_id = node.get("id")
    node_type = node.get("type")
    if node_id is None or not node_type:
        return None
    if node_type in SKIP_TYPES:
        return None
    # mode 2 = muted, 4 = bypass; we skip both. Real bypass should
    # passthrough the wires, but that's a layer of complexity we
    # don't need yet — same behavior as workflow-converter.ts.
    if node.get("mode") in (2, 4):
        return None

    widgets_values = node.get("widgets_values") or []
    workflow_inputs = node.get("inputs") or []

    api_inputs = {}
    node_class = nodes.NODE_CLASS_MAPPINGS.get(node_type)

    # ----- 1. Resolve connection inputs from the link table -----
    # Use workflow inputs[] order for slot indices because that's the
    # order links reference (link tuple stores dst_slot which is the
    # workflow-side slot index, not the canonical INPUT_TYPES index).
    for slot_idx, inp in enumerate(workflow_inputs):
        name = inp.get("name")
        if not name:
            continue
        link_id = inp.get("link")
        if link_id is None:
            continue
        src = link_by_dest.get((node_id, slot_idx))
        if src is not None:
            api_inputs[name] = [str(src[0]), int(src[1])]

    # ----- 2. Determine the widget order -----
    # In the modern Comfy frontend (v3+), every widget-bound input is
    # serialized into inputs[] with a `widget: {name}` tag, in the
    # SAME order as widgets_values was written. That's the canonical
    # source of truth for cursor walking: it's what the frontend
    # actually wrote out.
    #
    # Older saves (or some custom nodes) DON'T tag widget inputs in
    # inputs[]. For those we fall back to NODE_CLASS_MAPPINGS'
    # INPUT_TYPES() iteration order — that's how the legacy frontend
    # decided widget order too.
    #
    # Using INPUT_TYPES() as the primary order was the previous bug:
    # for v3-IO custom nodes (DepthAnything_V3 et al.) the schema
    # iteration order differs from the order the frontend actually
    # serializes widgets in, so widget values landed on the wrong
    # input names (normalization_mode getting resize_method's value,
    # tile_size getting overlap's value, etc.).
    tagged_widgets = [
        inp for inp in workflow_inputs if inp.get("widget") is not None
    ]
    # widget_order entries are (name, is_promoted, declared_type).
    # declared_type is needed for the type-aware padding skip below
    # — ComfyUI v3-IO custom nodes serialize widgets_values with
    # extra '' padding slots scattered between real widget values,
    # and we need the expected type to know when to skip them.
    widget_order = []
    widget_source = "tagged_widgets"
    if tagged_widgets:
        for inp in tagged_widgets:
            name = inp.get("name") or (inp.get("widget") or {}).get("name")
            if not name:
                continue
            widget_order.append(
                (name, inp.get("link") is not None, inp.get("type"))
            )
    else:
        # Legacy fallback: derive widget order from INPUT_TYPES.
        widget_source = "input_types_fallback"
        canonical = _canonical_input_order(node_class) if node_class else None
        if canonical is None:
            widget_source = "input_types_missing_walked_workflow_inputs"
            # No class registered AND no widget tags — derive from
            # widget-typed entries in workflow_inputs as a last resort.
            canonical = []
            for inp in workflow_inputs:
                name = inp.get("name")
                if not name:
                    continue
                if _is_widget_type(inp.get("type")):
                    canonical.append((name, inp.get("type")))
        # Build promotion lookup from workflow inputs[].
        promoted = {}
        for inp in workflow_inputs:
            name = inp.get("name")
            if not name:
                continue
            promoted[name] = inp.get("link") is not None
        for name, declared_type in canonical:
            if not _is_widget_type(declared_type):
                continue
            widget_order.append(
                (name, promoted.get(name, False), declared_type)
            )

    if _DEBUG:
        print(
            f"[rs/uitoapi] node {node_id} ({node_type}) "
            f"widget_source={widget_source} "
            f"widgets_values={widgets_values} "
            f"widget_order={widget_order} "
            f"workflow_inputs={[{'name': i.get('name'), 'type': i.get('type'), 'widget': i.get('widget'), 'link': i.get('link')} for i in workflow_inputs]}"
        )

    # ----- 3. Walk widget_order against widgets_values -----
    # Two shapes to handle:
    #
    # 1) LIST (legacy + most nodes): widgets_values is a positional
    #    array. Widgets promoted to inputs (link != None) still
    #    occupy a slot — the frontend writes the widget's current
    #    value there even though the runtime value comes from the
    #    link — so the cursor advances either way.
    #
    #    ComfyUI v3-IO custom nodes (RSLTXVGenerate etc.) serialize
    #    extra '' padding slots between real widget values —
    #    observed 9 extras in a 54-entry array for a 45-widget
    #    node. To stay aligned we type-check each slot: when the
    #    expected widget type is numeric / boolean / combo, an
    #    empty string is treated as padding and skipped.
    #
    #    Also handles ComfyUI's auto-injected control_after_generate
    #    widget that gets inserted after every seed / noise_seed INT
    #    widget but is NOT tagged in inputs[].
    #
    # 2) DICT (newer VHS releases — VHS_LoadVideo and friends):
    #    widgets_values is keyed by widget name. Cursor walking
    #    doesn't apply; we just look each name up directly.
    if isinstance(widgets_values, dict):
        for name, is_promoted, _declared_type in widget_order:
            if is_promoted:
                continue
            if name in widgets_values:
                api_inputs[name] = widgets_values[name]
            else:
                warnings.append(
                    f"node {node_id} ({node_type}): widget '{name}' "
                    f"missing from dict-shaped widgets_values "
                    f"(keys: {sorted(widgets_values.keys())})"
                )
    else:
        cursor = 0
        for name, is_promoted, declared_type in widget_order:
            # Skip over any padding slots that can't be this widget's value.
            while cursor < len(widgets_values) and not _value_matches_type(
                widgets_values[cursor], declared_type
            ):
                cursor += 1
            if cursor < len(widgets_values):
                if not is_promoted:
                    api_inputs[name] = widgets_values[cursor]
                cursor += 1
            elif not is_promoted:
                warnings.append(
                    f"node {node_id} ({node_type}): widget '{name}' "
                    f"requested cursor {cursor} but widgets_values has "
                    f"only {len(widgets_values)} entries"
                )
            # Auto-injected control_after_generate sits after every seed /
            # noise_seed widget in widgets_values regardless of whether
            # it's tagged in inputs[]. Always skip the slot — the value is
            # a frontend-only "fixed / increment / decrement / randomize"
            # control that the runner doesn't need.
            if _has_auto_control_after_generate(name):
                cursor += 1

    return str(node_id), {
        "class_type": node_type,
        "inputs": api_inputs,
    }


def _value_matches_type(value, declared_type):
    """Return True when `value` could plausibly be the saved widget
    value for an input of `declared_type`. Used to step the
    widgets_values cursor past padding slots that v3-IO nodes
    sometimes serialize between real widget values.

    The padding pattern we've observed: empty strings ('') wedged
    between numeric / boolean / combo widgets. Real values for
    those types are never empty strings, so '' is a strong padding
    signal. STRING-typed widgets legitimately allow empty values
    (guide_index_list defaults to ''), so we accept anything there.
    """
    # Booleans must be Python bool — empty strings / numbers / etc.
    # are all padding.
    if declared_type == "BOOLEAN":
        return isinstance(value, bool)
    # Numeric — int / float (but NOT bool, which is a numeric
    # subtype in Python).
    if declared_type in ("INT", "FLOAT"):
        if isinstance(value, bool):
            return False
        return isinstance(value, (int, float))
    # Combo (inline list of strings OR "COMBO" string from v3 IO).
    # Real combo values are non-empty strings.
    if isinstance(declared_type, (list, tuple)) or declared_type == "COMBO":
        if isinstance(value, str) and value == "":
            return False
        return True
    # STRING — anything (including '') is valid.
    if declared_type == "STRING":
        return True
    # Custom widget types (lowercase strings) — accept anything,
    # we don't have enough type info to discriminate.
    return True


def convert_ui_to_api(workflow):
    """Convert a UI-format workflow JSON (the saved Comfy graph) into
    the API prompt /prompt expects."""
    warnings = []
    if not isinstance(workflow, dict):
        raise ValueError("workflow must be a JSON object")
    nodes_list = workflow.get("nodes")
    if not isinstance(nodes_list, list):
        raise ValueError("workflow.nodes is missing or not a list")

    link_by_dest = _index_links_by_dest(workflow)
    api_prompt = {}

    for node in nodes_list:
        if not isinstance(node, dict):
            continue
        result = _convert_node(node, link_by_dest, warnings)
        if result is None:
            continue
        key, value = result
        api_prompt[key] = value

    return api_prompt, warnings


# Register on the running ComfyUI server. Tolerates being imported in a
# context where PromptServer isn't initialised yet (e.g. unit-test
# import of rs-nodes) by guarding against attribute errors.
try:
    _server = server.PromptServer.instance
except Exception:
    _server = None

if _server is not None:

    @_server.routes.post("/rs/uitoapi")
    async def _ui_to_api_route(request):
        try:
            workflow = await request.json()
        except Exception as err:
            return web.json_response(
                {"error": f"invalid JSON body: {err}"}, status=400
            )
        try:
            api_prompt, warnings = convert_ui_to_api(workflow)
        except ValueError as err:
            return web.json_response({"error": str(err)}, status=400)
        except Exception as err:
            # Log full traceback server-side, return a short message to
            # the client — the renderer doesn't need a stack.
            import traceback
            traceback.print_exc()
            return web.json_response(
                {"error": f"conversion failed: {err}"}, status=500
            )
        return web.json_response({"prompt": api_prompt, "warnings": warnings})
