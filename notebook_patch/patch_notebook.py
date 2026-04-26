"""Patch SMMG's generate_dataset.ipynb so its Cosmos cells call our T2.5 API.

Reads the upstream notebook, replaces the three Cosmos-related code cells:
  1. URL widget
  2. Variable dropdowns + cosmos_params widgets (was T1)
  3. process_video call (was cosmos_request)

…with T2.5-native equivalents that import from cosmos_t25_client +
notebook_widgets_t25. Writes the result to a new file.

Run from project root::

    python notebook_patch/patch_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "third_party" / "synthetic-manipulation-motion-generation" / "notebook" / "generate_dataset.ipynb"
DST = ROOT / "notebook_patch" / "generate_dataset_t25.ipynb"


URL_CELL = """\
import ipywidgets as widgets
import os

# Default points at the NVIDIA Cosmos Transfer 2.5 NIM (port 8000).
# Override at the Jupyter prompt if NIM lives on a different host.
default_url = os.environ.get("COSMOS_API_URL", "http://cosmos-nim:8000")
url_widget = widgets.Text(
    value=default_url,
    placeholder="http://host:port",
    description="Cosmos NIM URL:",
    style={"description_width": "initial"},
    layout={"width": "700px"},
)
display(url_widget)

# Quick liveness check
try:
    from cosmos_t25_client import healthz
    print("Cosmos NIM health:", healthz(url_widget.value))
except Exception as e:
    print(f"(Cosmos NIM unreachable yet — set the URL above and re-run this cell): {e}")
"""

PARAMS_CELL = """\
from notebook_widgets import create_variable_dropdowns
from notebook_widgets_t25 import create_t25_params
from notebook_utils import ISAACLAB_OUTPUT_DIR

# Prompt mixer (cube × table × location) — reuse the original templates.
prompt_manager = create_variable_dropdowns("stacking_prompt.toml")

# Cosmos Transfer 2.5 inference parameters.
cosmos_params = create_t25_params(ISAACLAB_OUTPUT_DIR)
for w in cosmos_params.values():
    display(w)
"""

SUBMIT_CELL = """\
import os
from cosmos_t25_client import transfer
from notebook_widgets_t25 import widgets_to_t25_call_kwargs
from notebook_utils import ISAACLAB_OUTPUT_DIR, COSMOS_OUTPUT_DIR
from IPython.display import Video, clear_output

if not url_widget.value:
    raise ValueError("Cosmos API URL is empty.")

call = widgets_to_t25_call_kwargs(cosmos_params)
input_video_name = call.pop("_input_video_filename")
video_filepath = os.path.join(ISAACLAB_OUTPUT_DIR, input_video_name)

os.makedirs(COSMOS_OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(COSMOS_OUTPUT_DIR, f"cosmos_t25_seed{call['seed']}.mp4")

result = transfer(
    api_url=url_widget.value,
    video_path=video_filepath,
    output_path=output_path,
    prompt=prompt_manager.prompt,
    **call,
)

clear_output(wait=True)
print(f"Done — job_id={result['job_id']}")
print(f"Saved → {result['output_path']}")
display(Video(result["output_path"], width=900))
"""


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"t25-{abs(hash(src))%(10**8)}",
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def main() -> int:
    nb = json.loads(SRC.read_text())
    cells = nb["cells"]

    # We surgically replace 3 code cells. They're identifiable by content
    # markers from the upstream notebook.
    swaps = [
        ("create url widget",
         lambda src: "url_widget = widgets.Text" in src and "Cosmos URL" in src,
         URL_CELL),
        ("create cosmos params",
         lambda src: "create_variable_dropdowns" in src and "create_cosmos_params" in src,
         PARAMS_CELL),
        ("submit + download",
         lambda src: "from cosmos_request import process_video" in src,
         SUBMIT_CELL),
    ]

    new_cells = []
    swap_results = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            new_cells.append(cell)
            continue
        src = "".join(cell.get("source", []))
        replaced = False
        for label, predicate, replacement in swaps:
            if predicate(src):
                new_cells.append(code_cell(replacement))
                swap_results.append(label)
                replaced = True
                break
        if not replaced:
            new_cells.append(cell)

    nb["cells"] = new_cells
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(nb, indent=1))

    print(f"Replaced cells: {swap_results}")
    print(f"Wrote: {DST}")
    if len(swap_results) != len(swaps):
        missing = [s[0] for s in swaps if s[0] not in swap_results]
        print(f"WARNING: did NOT find/replace: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
