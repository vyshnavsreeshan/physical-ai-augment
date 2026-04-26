"""Diagnose: did Mimic actually produce motion, or stuck frames?"""
import os, re
from collections import defaultdict
import numpy as np
from PIL import Image

ROOT = "/workspace/_isaaclab_out"
PAT = re.compile(r"^(.*)_semantic_segmentation_trial_(\d+)_tile_(\d+)_step_(\d+)\.png$")

counts = defaultdict(list)
for f in os.listdir(ROOT):
    m = PAT.match(f)
    if m:
        cam, trial, tile, step = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        counts[(cam, trial, tile)].append(step)

for k, steps in sorted(counts.items()):
    steps.sort()
    print(f"{k[0]}  trial={k[1]} tile={k[2]}  frames={len(steps)}  range=[{steps[0]}..{steps[-1]}]")

print()
print("=== motion check ===")
for (cam, trial, tile), steps in sorted(counts.items()):
    if len(steps) < 60:
        continue
    steps = sorted(steps)
    f1, f2, f3 = steps[0], steps[len(steps)//2], steps[-1]
    def load(s):
        return np.array(Image.open(f"{ROOT}/{cam}_semantic_segmentation_trial_{trial}_tile_{tile}_step_{s}.png"))
    a = load(f1); b = load(f2); c = load(f3)
    d12 = np.abs(a.astype(int) - b.astype(int)).mean()
    d13 = np.abs(a.astype(int) - c.astype(int)).mean()
    print(f"{cam} trial={trial}  step{f1}↔{f2}: diff={d12:.2f}  step{f1}↔{f3}: diff={d13:.2f}  "
          f"({'MOVING' if d13 > 1.0 else 'STATIONARY'})")
