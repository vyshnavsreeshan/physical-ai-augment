# physical-ai-augment

NVIDIA Isaac Lab Mimic + **Cosmos Transfer 2.5 NIM**, dockerized for
**Blackwell / Hopper**. Generate synthetic robot manipulation trajectories
with Mimic, photorealistically augment them with Cosmos Transfer 2.5, and
use the result to fine-tune visuomotor imitation-learning policies.

This is a port of NVIDIA's [Synthetic Manipulation Motion Generation
blueprint](https://github.com/NVIDIA-Omniverse-blueprints/synthetic-manipulation-motion-generation)
that:

- Replaces the NGC-shipped Isaac Sim 4.5 / IsaacLab 2.0.2 / torch 2.5.1+cu118
  base (incompatible with Blackwell `sm_120`) with **Isaac Sim 5.1 + Isaac Lab
  v2.2.1 + torch 2.7.0+cu128**
- Replaces the upstream Cosmos Transfer 1 client with a **native
  Cosmos Transfer 2.5 NIM** integration (single sync `POST /v1/infer`,
  persistent models, concurrent requests)
- Patches IsaacLab 2.2.1 / Blackwell-specific drift (env_loop signature,
  empty-buffer image-save guard, headless experience-file selection,
  software h264 instead of broken NVENC)

---

## Architecture

```
┌─────────────────────────── smmg-lab container ────────────────────────────┐
│  Isaac Sim 5.1 + Isaac Lab v2.2.1 + Jupyter Lab + patched notebook       │
│                                                                          │
│  Generates Mimic trajectories from `samples/annotated_dataset.hdf5`.     │
│  Encodes shaded-segmentation MP4 (Warp kernel + libx264).                │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │ POST /v1/infer (base64 video + prompt)
                                  ▼
┌────────────────────────── cosmos-nim container ──────────────────────────┐
│  nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:1.0.0                          │
│                                                                          │
│  Persistent models (Transfer 2.5 + Predict 2.5 + Reason 1 + Guardrail).  │
│  Triton-backed, handles concurrent requests internally.                  │
│  Returns photoreal MP4 inline (base64).                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

The two services are decoupled: SMMG side does dataset generation, NIM does
photoreal augmentation. They communicate over HTTP. You can run them on the
same machine OR on separate machines (recommended — see "Two-machine
deployment" below).

## Quick start (single machine)

### 0. Prerequisites

- **GPU(s)** — any cu128-compatible NVIDIA (Ampere / Ada / Hopper / Blackwell), 24+ GB VRAM
- **RAM** — 32+ GB (Cosmos NIM peaks at ~25 GB during model load)
- **Disk** — ~150 GB free (datasets + checkpoints + Docker images + NIM cache)
- **Docker** with `nvidia-container-toolkit`
- **NGC account** with API key —
  [https://org.ngc.nvidia.com/setup/api-key](https://org.ngc.nvidia.com/setup/api-key)
  (free with NVIDIA Developer Program)
- **HuggingFace account** with read access to
  [`nvidia/PhysicalAI-Robotics-Manipulation-Augmented`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Augmented)
  (CC-BY-4.0; only required if you plan to download the public Mimic
  trajectories or pre-trained policy checkpoints)

### 1. Clone + grab upstream sources

```bash
git clone https://github.com/vyshnavsreeshan/physical-ai-augment.git
cd physical-ai-augment

# Upstream code we mount/COPY at build time. Both pinned to specific tags
# so the build is reproducible — main branches drift, tags don't.
mkdir -p third_party
git -C third_party clone --depth 1 --branch v2.2.1 \
    https://github.com/isaac-sim/IsaacLab.git
git -C third_party clone --depth 1 --branch v1.0 \
    https://github.com/NVIDIA-Omniverse-blueprints/synthetic-manipulation-motion-generation.git
```

### 2. Authenticate to NGC and pull the Cosmos NIM

```bash
export NGC_API_KEY=nvapi-...
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
docker pull nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:1.0.0
# 30-60 GB pull, takes 10-20 min on a fast link
```

### 3. Build the SMMG image

```bash
docker build -f docker/smmg-blackwell.Dockerfile \
    -t physical-ai-lab/smmg-blackwell:1.0 .
# ~30 min the first time (Isaac Sim wheel download is the bulk)
```

### 4. Bring up the stack

```bash
export NGC_API_KEY=nvapi-...               # required for the cosmos-nim service
docker compose -f docker/docker-compose.yml up -d
```

- Jupyter Lab → http://localhost:8888/lab (notebook auto-opens)
- Cosmos NIM → http://localhost:8000/v1/health/ready

First start of cosmos-nim takes ~5-10 min as it downloads model weights into
`~/.cache/nim/`. Watch:
```bash
docker logs -f cosmos-nim 2>&1 | grep -E "ready|error|Application"
```

### 5. Run the notebook

In Jupyter:

1. Cell 1: pick `num_trials = 1` for a fast first end-to-end test
2. Cell 2: launch Isaac Sim (Mimic env) — ~5 min first time
3. Cell 3: accept default randomization sliders
4. Cell 4: generate trajectory — watch for `1/N successful demos`
5. Cells 5–7: pick camera (`table_cam`), encode shaded-segmentation MP4
6. Cell 8: URL widget — auto-filled with `http://cosmos-nim:8000`
7. Cells 9–10: prompt mixer + T2.5 params (defaults are fine)
8. Cell 11: submit — ~5–14 min, returns photoreal MP4 inline (depends on
   `num_steps`, resolution, video length)

## Two-machine deployment (recommended)

When the GPU is shared between Mimic generation and Cosmos inference, peak
RAM is tight. Running NIM on a dedicated Hopper / Blackwell box gives both
services breathing room and is closer to the deployment topology NVIDIA's
SMMG blueprint assumes.

```bash
# on the dedicated Cosmos host (e.g. an H100/H200 instance)
export NGC_API_KEY=nvapi-...
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
docker pull nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:1.0.0

mkdir -p ~/.cache/nim && chmod -R 777 ~/.cache/nim 2>/dev/null || true
docker run -d --name cosmos-nim --restart unless-stopped \
    --runtime=nvidia --gpus all \
    --shm-size=32GB --ulimit nofile=65536:65536 \
    -e NGC_API_KEY=$NGC_API_KEY \
    -v ~/.cache/nim:/opt/nim/.cache \
    -p 8000:8000 \
    nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:1.0.0

# on the Isaac Sim host: edit docker/docker-compose.yml so the smmg-lab
# service has  COSMOS_API_URL: "http://<cosmos-host>:8000",
# remove the cosmos-nim service block, and:
docker compose -f docker/docker-compose.yml up -d smmg-lab
```

## Repository layout

```
physical-ai-augment/
├── docker/
│   ├── smmg-blackwell.Dockerfile        # Isaac Sim 5.1 + Lab 2.2.1 + Jupyter
│   └── docker-compose.yml               # smmg-lab + cosmos-nim
├── notebook_patch/
│   ├── cosmos_t25_client.py             # NIM client (POST /v1/infer)
│   ├── notebook_widgets_t25.py          # ipywidgets matching NIM's request schema
│   ├── patch_notebook.py                # Patches upstream notebook
│   └── generate_dataset_t25.ipynb       # Pre-patched notebook (output of patcher)
├── smmg_patches/                        # Runtime patches for IsaacLab/notebook drift
│   ├── force_headless.py                #   Loads correct offscreen experience file
│   ├── patch_env_loop_call.py           #   Updates env_loop signature for IsaacLab 2.2.1
│   ├── skip_empty_image_save.py         #   Avoids Pillow crash on init-time empty obs
│   ├── imageio_encode.py                #   Replaces NVENC encode with software h264
│   └── check_motion.py / check_hdf5.py  #   Diagnostic helpers used while debugging
├── scripts/
│   ├── activate_env.sh                  # Source to activate local env (non-Docker)
│   ├── smoke_test_isaaclab.sh           # 4-stage smoke test for the SMMG image
│   ├── smmg_smoke.py                    #   Same, run inside container
│   └── robust_eval_record.py            # Patched robust_eval that saves rollout MP4s
├── legacy/                              # Pre-NIM Flask wrapper (kept for offline use)
│   └── README.md                        #   See for details
├── docs/
│   └── PROJECT_PLAN.md                  # The 9-phase plan
└── README.md
```
## References

- [NVIDIA NIM for Cosmos](https://docs.nvidia.com/nim/cosmos/latest/quickstart-guide.html)
- [NVIDIA Synthetic Manipulation Motion Generation
  blueprint](https://github.com/NVIDIA-Omniverse-blueprints/synthetic-manipulation-motion-generation)
- [Isaac Lab — Augmented Imitation
  Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/augmented_imitation.html)
- [Cosmos Transfer 2.5 (source)](https://github.com/nvidia-cosmos/cosmos-transfer2.5)
- [Hugging Face — `nvidia/PhysicalAI-Robotics-Manipulation-Augmented`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Augmented)

## License

Mirrors NVIDIA's source licenses (Apache 2.0 / BSD-3-Clause for the
underlying frameworks). Cube-stacking dataset is CC-BY-4.0 (NVIDIA).
NVIDIA NIM container is governed by the NVIDIA Software License Agreement.
See individual upstream repos for terms.
