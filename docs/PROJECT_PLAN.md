# Project Plan — Augmented Imitation Learning Demo

**Source:** NVIDIA's reference pipeline, documented at
<https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/augmented_imitation.html>

**Target task environment:** `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0`

**Goal:** Train two Robomimic BC policies on the same Franka cube-stacking task
— one on NVIDIA's 1k Mimic dataset, one on the Mimic+Cosmos-augmented 2k
dataset — and reproduce the published robustness gap across six evaluation
settings (lighting, textures, etc.).

## Phases

0. **Pre-flight verification** — confirm deps, env, scripts, budget. [COMPLETE]
1. **Acquire baseline data** — pull `mimic_dataset_1k.hdf5` (and optionally
   `cosmos_dataset_1k.hdf5` as a fallback shortcut) from
   `nvidia/PhysicalAI-Robotics-Manipulation-Augmented` on HF.
2. **HDF5 → MP4** — `scripts/tools/hdf5_to_mp4.py`.
3. **Adapt Cosmos workflow to Transfer2.5** — port the Transfer1 controlnet
   spec (`sigma_max=50`, `control_weight="0.3,0.3,0.6,0.7"`,
   `hint_key="blur,canny,depth,segmentation"`) to Transfer2.5's JSON format;
   get a single clip working end-to-end.
4. **Batch Cosmos augmentation** — 1,000 demos × 1 prompt each, matching
   NVIDIA's experimental scale.
5. **MP4 → HDF5 + merge** — `mp4_to_hdf5.py` then `merge_hdf5_datasets.py` →
   `mimic_cosmos_dataset.hdf5` (2,000 demos).
6. **Train baseline (Policy A)** — Robomimic BC on `mimic_dataset_1k.hdf5`.
7. **Train augmented (Policy B)** — identical hyperparameters on
   `mimic_cosmos_dataset.hdf5`.
8. **Robust eval** — `scripts/imitation_learning/robomimic/robust_eval.py`
   across six settings (Vanilla, Light Intensity, Light Color, Light Texture,
   Table Texture, Robot Arm Texture).
9. **Demo assets** — split-screen video, results table, writeup, reproducible
   code.

## Reference commands (from NVIDIA's doc)

```bash
# HDF5 → MP4
python scripts/tools/hdf5_to_mp4.py \
  --input_file datasets/mimic_dataset_1k.hdf5 \
  --output_dir datasets/mimic_dataset_1k_mp4

# Prompt generation
python scripts/tools/cosmos/cosmos_prompt_gen.py \
  --templates_path scripts/tools/cosmos/transfer1_templates.json \
  --num_prompts 10

# MP4 → HDF5 (after Cosmos augmentation)
python scripts/tools/mp4_to_hdf5.py \
  --input_file datasets/mimic_dataset_1k.hdf5 \
  --videos_dir datasets/cosmos_dataset_1k_mp4 \
  --output_file datasets/cosmos_dataset_1k.hdf5

# Merge
python scripts/tools/merge_hdf5_datasets.py \
  --input_files datasets/mimic_dataset_1k.hdf5 datasets/cosmos_dataset_1k.hdf5 \
  --output_file datasets/mimic_cosmos_dataset.hdf5

# Training (Robomimic BC)
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0 --algo bc

# Robust evaluation
./isaaclab.sh -p scripts/imitation_learning/robomimic/robust_eval.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0
```

## Reference success-rate targets

| Setting             | Mimic 1k | Mimic 2k | Cosmos-Mimic 2k |
|---------------------|---------:|---------:|----------------:|
| Vanilla             |    62.0% |    96.6% |           86.6% |
| Light Intensity     |    11.1% |    20.0% |           62.2% |
| Light Color         |    24.6% |    30.0% |           77.7% |
| Light Texture       |    16.6% |    20.0% |           68.8% |
| Table Texture       |     0.0% |     0.0% |           20.0% |
| Robot Arm Texture   |     0.0% |     0.0% |            4.4% |

The headline is the Cosmos-Mimic column — this is Policy B's target.

## MVP fallback

If batch Cosmos inference proves too slow or Transfer2.5 porting drags, skip
Phases 2–5 and download `cosmos_dataset_1k.hdf5` directly from HF (it's
published alongside the baseline). Lose the "we ran Cosmos ourselves" part of
the story but keep the full training and evaluation comparison.
