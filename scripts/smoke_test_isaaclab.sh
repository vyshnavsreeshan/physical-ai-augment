#!/usr/bin/env bash
# Quick smoke test: verify Isaac Sim headless app boots and list Stack-Cube envs.
set -e
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/activate_env.sh"

python -u - <<'PY'
import sys

print("[1/4] Launching Isaac Sim headless via AppLauncher...")
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app
print("       app launched")

print("[2/4] Import isaaclab after kit is live")
import isaaclab
print("       isaaclab", isaaclab.__version__ if hasattr(isaaclab, "__version__") else "imported")

print("[3/4] Register envs from isaaclab_tasks")
import gymnasium as gym
import isaaclab_tasks  # noqa: F401 — triggers env registration
total = len(gym.envs.registry)
print(f"       total registered envs: {total}")

print("[4/4] Search for candidate visuomotor stack-cube envs")
patterns = ["Stack-Cube", "Galbot", "Visuomotor"]
hits = sorted(
    n for n in gym.envs.registry.keys()
    if any(p in n for p in patterns)
)
for n in hits:
    print("   -", n)
print(f"       candidates: {len(hits)}")

simulation_app.close()
print("OK")
PY
