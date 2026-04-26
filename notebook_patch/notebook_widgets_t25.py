"""T2.5 NIM-flavored widgets for the SMMG-style notebook.

Mirrors the field names accepted by the **NVIDIA Cosmos Transfer 2.5 NIM**
(``cosmos-transfer2.5-2b``) — `guidance_scale` / `steps` (not `guidance` /
`num_steps`), no `sigma_max` (NIM uses an internal default).
"""

from __future__ import annotations

import os
from pathlib import Path

import ipywidgets as widgets


def _list_videos_in(d: str) -> list[str]:
    p = Path(d)
    if not p.exists():
        return []
    return sorted(f.name for f in p.iterdir() if f.suffix.lower() == ".mp4")


def create_t25_params(isaaclab_output_dir: str = "_isaaclab_out") -> dict[str, widgets.Widget]:
    """Build the T2.5 NIM parameter widget set.

    Returns dict[name] → widget. Read each widget's `.value` at submit time.
    """
    videos = _list_videos_in(isaaclab_output_dir)

    return {
        "input_video": widgets.Dropdown(
            options=videos or ["(no video — generate one in the previous step)"],
            description="Input video:",
            style={"description_width": "initial"},
            layout={"width": "700px"},
        ),
        "seed": widgets.IntText(
            value=2025, description="seed:",
            style={"description_width": "initial"},
        ),
        "guidance": widgets.IntSlider(
            value=3, min=0, max=7, step=1,
            description="guidance:", style={"description_width": "initial"},
        ),
        "num_steps": widgets.IntSlider(
            value=35, min=4, max=80, step=1,
            description="num_steps:", style={"description_width": "initial"},
        ),
        "sigma_max": widgets.FloatSlider(
            value=70.0, min=0.0, max=80.0, step=1.0,
            description="sigma_max:", style={"description_width": "initial"},
        ),
        "resolution": widgets.Dropdown(
            options=["256", "480", "512", "720"], value="720",
            description="resolution:", style={"description_width": "initial"},
        ),
        "edge_weight": widgets.FloatSlider(
            value=0.6, min=0.0, max=1.0, step=0.05,
            description="edge.control_weight:",
            style={"description_width": "initial"},
            layout={"width": "500px"},
        ),
        "depth_weight": widgets.FloatSlider(
            value=0.0, min=0.0, max=1.0, step=0.05,
            description="depth.control_weight:",
            style={"description_width": "initial"},
            layout={"width": "500px"},
        ),
        "seg_weight": widgets.FloatSlider(
            value=0.0, min=0.0, max=1.0, step=0.05,
            description="seg.control_weight:",
            style={"description_width": "initial"},
            layout={"width": "500px"},
        ),
        "vis_weight": widgets.FloatSlider(
            value=0.0, min=0.0, max=1.0, step=0.05,
            description="vis.control_weight (blur):",
            style={"description_width": "initial"},
            layout={"width": "500px"},
        ),
    }


def widgets_to_t25_call_kwargs(params: dict[str, widgets.Widget]) -> dict:
    """Convert the widget set into kwargs you can pass straight into
    `cosmos_t25_client.transfer(...)`.

    Branches with weight == 0.0 are dropped so NIM doesn't enable them.
    """
    controls: dict[str, dict] = {}
    for branch, key in (
        ("edge", "edge_weight"),
        ("depth", "depth_weight"),
        ("seg", "seg_weight"),
        ("vis", "vis_weight"),
    ):
        # Tolerate widget dicts built by an older version that didn't have
        # this branch — silently skip rather than KeyError.
        widget = params.get(key)
        if widget is None:
            continue
        w = float(widget.value)
        if w > 0.0:
            controls[branch] = {"control_weight": w}

    out: dict = {
        "resolution": str(params["resolution"].value),
        "controls": controls,
        "_input_video_filename": params["input_video"].value,
    }
    # Optional knobs (only include when widget is present in this version)
    for key in ("seed", "guidance", "num_steps", "sigma_max"):
        w = params.get(key)
        if w is None:
            continue
        out[key] = float(w.value) if isinstance(w.value, float) else int(w.value)
    return out
