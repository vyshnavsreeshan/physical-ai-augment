"""Patch IsaacLab's stack_ik_rel_blueprint_env_cfg.image() to skip the
PNG-save block when the camera buffer is empty.

Background
----------
At ManagerBasedEnv init, ObservationManager._prepare_terms calls each obs
function once to determine its output shape — but at that point the camera
hasn't rendered, so `sensor.data.output["rgb"]` is shape (0, 0, 0, 0).
The original env_cfg unconditionally writes a PNG, and torchvision/Pillow
crashes on the empty tensor with "tile cannot extend outside image".

Fix
---
Wrap the `if save_image_to_file:` block to first check `images.numel() > 0`.
The shape probe will return the empty tensor (its shape is what matters);
real saves happen on subsequent calls when the camera has rendered.

Run inside the smmg-lab container:
    docker exec smmg-lab python3.11 /smoke/smmg_patches/skip_empty_image_save.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TARGET = Path(
    "/opt/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/"
    "stack/config/franka/stack_ik_rel_blueprint_env_cfg.py"
)


def main() -> int:
    src = TARGET.read_text()
    needle = "    if save_image_to_file:\n"
    if needle not in src:
        print(f"FAIL: anchor not found in {TARGET}", file=sys.stderr)
        return 1
    if "# PATCH: skip on empty image buffer" in src:
        print("already patched")
        return 0
    replacement = (
        "    # PATCH: skip on empty image buffer (observation manager init)\n"
        "    if save_image_to_file and images.numel() > 0:\n"
    )
    new = src.replace(needle, replacement, 1)
    TARGET.write_text(new)
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
