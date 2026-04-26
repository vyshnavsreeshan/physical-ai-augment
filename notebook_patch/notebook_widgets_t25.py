"""T2.5-flavored widgets for the SMMG-style notebook.

Replaces SMMG's `notebook_widgets.create_cosmos_params` (which exposed T1
parameters: control_weight, sigma_max, canny_strength) with widgets that
match Cosmos Transfer 2.5's InferenceArguments schema.
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
    """Build the T2.5 parameter widget set.

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
            value=2025, description="seed:", style={"description_width": "initial"}
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
            value=70.0, min=0.0, max=200.0, step=1.0,
            description="sigma_max:", style={"description_width": "initial"},
        ),
        "resolution": widgets.Dropdown(
            options=["720", "480"], value="720",
            description="resolution:", style={"description_width": "initial"},
        ),
        "edge_weight": widgets.FloatSlider(
            value=0.6, min=0.0, max=1.0, step=0.05,
            description="edge.control_weight:", style={"description_width": "initial"},
            layout={"width": "500px"},
        ),
        "depth_weight": widgets.FloatSlider(
            value=0.5, min=0.0, max=1.0, step=0.05,
            description="depth.control_weight:", style={"description_width": "initial"},
            layout={"width": "500px"},
        ),
        "seg_weight": widgets.FloatSlider(
            value=0.0, min=0.0, max=1.0, step=0.05,
            description="seg.control_weight:", style={"description_width": "initial"},
            layout={"width": "500px"},
        ),
    }


def widgets_to_t25_call_kwargs(params: dict[str, widgets.Widget]) -> dict:
    """Convert the widget set into kwargs you can pass straight into
    `cosmos_t25_client.transfer(...)`.

    Branches with weight == 0.0 are dropped so T2.5 doesn't enable them.
    """
    controls: dict[str, dict] = {}
    for branch, key in (("edge", "edge_weight"), ("depth", "depth_weight"), ("seg", "seg_weight")):
        w = float(params[key].value)
        if w > 0.0:
            controls[branch] = {"control_weight": w}

    return {
        "seed": int(params["seed"].value),
        "guidance": int(params["guidance"].value),
        "num_steps": int(params["num_steps"].value),
        "sigma_max": float(params["sigma_max"].value),
        "resolution": str(params["resolution"].value),
        "controls": controls,
        "_input_video_filename": params["input_video"].value,
    }
