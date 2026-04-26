"""Force headless=True in the AppLauncher config cell.

The upstream notebook leaves args_cli.headless at its default (False), so
AppLauncher loads the display-rendering experience file. Then it auto-falls
back to --no-window because DISPLAY isn't set, but the experience file is
still wrong — the offscreen camera capture pipeline isn't initialized
correctly, and `sensor.data.output["rgb"]` returns the same buffer every
step (causing stationary PNG dumps even though the env is actually moving).

Forcing headless=True makes AppLauncher load isaaclab.python.headless.rendering.kit
which has the correct offscreen-camera config.
"""
import json
import sys
from pathlib import Path


def main(notebook_path: str) -> int:
    p = Path(notebook_path)
    nb = json.loads(p.read_text())
    patched = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "args_cli.enable_cameras = True" in src and "args_cli.headless" not in src:
            new_src = src.replace(
                "args_cli.enable_cameras = True",
                "args_cli.enable_cameras = True\nargs_cli.headless = True",
                1,
            )
            cell["source"] = new_src.splitlines(keepends=True)
            patched += 1
    p.write_text(json.dumps(nb, indent=1))
    print(f"patched {patched} cell(s) in {p}")
    return 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/workspace/generate_dataset.ipynb"
    sys.exit(main(target))
