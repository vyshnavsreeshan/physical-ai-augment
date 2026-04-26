"""Patch the notebook's env_loop call to match IsaacLab 2.2.1's signature.

Old (2.0.2):
    env_loop(env, action_queue, info_pool, event_loop)

New (2.2.1):
    env_loop(env, reset_queue, action_queue, info_pool, event_loop)
"""

import json
import sys
from pathlib import Path

OLD = "env_loop(env, async_components['action_queue'], \n            async_components['info_pool'], async_components['event_loop'])"
NEW = "env_loop(env, async_components['reset_queue'], async_components['action_queue'], \n            async_components['info_pool'], async_components['event_loop'])"


def main(notebook_path: str) -> int:
    p = Path(notebook_path)
    nb = json.loads(p.read_text())
    patched = 0
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "env_loop(env," in src and "async_components['reset_queue']" not in src:
            new_src = src.replace(OLD, NEW, 1)
            if new_src != src:
                cell["source"] = new_src.splitlines(keepends=True)
                patched += 1
    p.write_text(json.dumps(nb, indent=1))
    print(f"patched {patched} cell(s) in {p}")
    return 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/workspace/generate_dataset.ipynb"
    sys.exit(main(target))
