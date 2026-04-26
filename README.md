# physical-ai-augment

NVIDIA Isaac Lab Mimic + Cosmos Transfer 2.5, dockerized for **Blackwell / Hopper**.
Generate synthetic robot manipulation trajectories with Mimic, photorealistically
augment them with Cosmos Transfer 2.5, and use the result to fine-tune
visuomotor imitation-learning policies.

This is a port of NVIDIA's [Synthetic Manipulation Motion Generation
blueprint](https://github.com/NVIDIA-Omniverse-blueprints/synthetic-manipulation-motion-generation)
that:

- Replaces the NGC-shipped Isaac Sim 4.5 / IsaacLab 2.0.2 / torch 2.5.1+cu118
  base (incompatible with Blackwell `sm_120`) with **Isaac Sim 5.1 + Isaac Lab
  v2.2.1 + torch 2.7.0+cu128**
- Replaces the upstream Cosmos Transfer 1 client with a **native Cosmos
  Transfer 2.5** client + Flask API
- Produces a self-contained T2.5 image (`vyshnavsreeshan05/cosmos-transfer2.5:standalone`)
  that runs anywhere with one `docker run` (Hopper, Blackwell, Ada — any
  cu130-compatible NVIDIA GPU)

---

## Architecture

```
┌─────────────────────────────────┐    HTTP    ┌──────────────────────────────┐
│  smmg-lab container             │            │  cosmos-api container        │
│  ─────────────────────          │  POST      │  ─────────────────────       │
│  Isaac Sim 5.1                  │  /v1/...   │  Cosmos Transfer 2.5         │
│  Isaac Lab v2.2.1               │ ──────────►│  Flask wrapper around        │
│  Jupyter Lab + patched notebook │            │  examples/inference.py       │
│  Generates Mimic trajectories,  │            │  Returns photoreal MP4       │
│  encodes shaded-segmentation    │            │                              │
│  MP4s as Cosmos input           │ ◄──────────│  Heavy GPU + RAM for         │
│                                 │  result    │  model loading (4 models,    │
│                                 │  MP4       │  ~22 GB peak)                │
└─────────────────────────────────┘            └──────────────────────────────┘
        Blackwell host                              Same host or remote
                                                    (recommended: Hopper H100/H200)
```

The two services are decoupled: `smmg-lab` is dataset generation, `cosmos-api`
is photoreal augmentation. They communicate over HTTP. You can run them on
the same machine (24 GB RAM is borderline OOM during Cosmos load) or on
separate machines (recommended for any serious workload).

## Quick start

### 0. Prerequisites

- **GPU(s)** — any cu128-compatible NVIDIA (Ampere / Ada / Hopper / Blackwell), 24+ GB VRAM
- **RAM** — 32+ GB if running both services on one host, 24 GB minimum if cosmos lives elsewhere
- **Disk** — ~150 GB free (datasets + checkpoints + Docker images)
- **Docker** with `nvidia-container-toolkit`
- **HuggingFace account** with access to
  [`nvidia/Cosmos-Transfer2.5-2B`](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B)
  (gated — accept terms of access first)

### 1. Clone + grab upstream sources

```bash
git clone https://github.com/vyshnavsreeshan/physical-ai-augment.git
cd physical-ai-augment

# Upstream code we mount/COPY at build time
mkdir -p third_party
git -C third_party clone --depth 1 --branch v2.2.1 \
    https://github.com/isaac-sim/IsaacLab.git
git -C third_party clone --depth 1 \
    https://github.com/nvidia-cosmos/cosmos-transfer2.5.git
git -C third_party clone --depth 1 \
    https://github.com/NVIDIA-Omniverse-blueprints/synthetic-manipulation-motion-generation.git

# Some assets need git-lfs in the cosmos-transfer2.5 repo
(cd third_party/cosmos-transfer2.5 && git lfs install && git lfs pull)
```

### 2. Pull the prebuilt Cosmos image (or build from source)

```bash
# Option A — fastest, prebuilt image
docker pull vyshnavsreeshan05/cosmos-transfer2.5:standalone

# Option B — rebuild from source (~10 min)
docker build -f third_party/cosmos-transfer2.5/docker/nightly.Dockerfile \
    -t cosmos-transfer2.5:blackwell-nightly third_party/cosmos-transfer2.5
docker build -f docker/cosmos-api.Dockerfile \
    -t cosmos-transfer2.5:blackwell-api .
docker build -f docker/cosmos-api-standalone.Dockerfile \
    -t cosmos-transfer2.5:standalone .
```

### 3. Build the SMMG image

```bash
docker build -f docker/smmg-blackwell.Dockerfile \
    -t physical-ai-lab/smmg-blackwell:1.0 .
# ~30 min the first time (Isaac Sim wheel download is the bulk)
```

### 4. (Optional) Pre-fetch HF model weights

Avoids ~10 min of first-run Cosmos downloads. Requires a HuggingFace token
that has accepted Cosmos-Transfer2.5-2B's terms.

```bash
export HF_TOKEN=hf_...   # https://huggingface.co/settings/tokens
huggingface-cli download nvidia/Cosmos-Transfer2.5-2B
huggingface-cli download nvidia/Cosmos-Reason1-7B
huggingface-cli download nvidia/Cosmos-Guardrail1
```

### 5. Bring up the stack

```bash
docker compose -f docker/docker-compose.yml up -d
```

- Jupyter Lab → http://localhost:8888/lab (notebook auto-opens)
- Cosmos API → http://localhost:5001/healthz

### 6. Run the notebook

In Jupyter:

1. Cell 1: pick `num_trials = 1` for a fast first end-to-end test
2. Cell 2: launch Isaac Sim (Mimic env) — ~5 min first time
3. Cell 3: accept default randomization sliders
4. Cell 4: generate trajectory — watch for `1/N successful demos`
5. Cells 5–7: pick camera (`table_cam`), encode shaded-segmentation MP4
6. Cell 8: URL widget — auto-filled with `http://cosmos-api:5000`
7. Cells 9–10: prompt mixer + T2.5 params (defaults are fine)
8. Cell 11: submit — ~5–14 min, returns photoreal MP4 inline

---

## Two-machine deployment (recommended)

For serious use, run Cosmos on a separate Hopper or Blackwell box. The
prebuilt standalone image makes this one command:

```bash
# on the Cosmos host (e.g. an H100/H200 instance)
docker run -d --name cosmos-api \
    --restart unless-stopped \
    --runtime=nvidia --gpus all --ipc=host \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e HF_TOKEN=$HF_TOKEN \
    -p 5000:5000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vyshnavsreeshan05/cosmos-transfer2.5:standalone

# on the Isaac Sim host: edit docker/docker-compose.yml so the smmg-lab
# service has  COSMOS_API_URL: "http://<cosmos-host>:5000"
docker compose -f docker/docker-compose.yml up -d smmg-lab
```

---

## Repository layout

```
physical-ai-augment/
├── docker/                          # Dockerfiles + compose + Flask API
│   ├── smmg-blackwell.Dockerfile        # Isaac Sim 5.1 + Lab 2.2.1 + Jupyter
│   ├── cosmos-api.Dockerfile            # T2.5 + Flask, expects bind-mount
│   ├── cosmos-api-standalone.Dockerfile # T2.5 + Flask, fully self-contained
│   ├── cosmos_api.py                    # Flask API server
│   └── docker-compose.yml               # Both services networked
├── notebook_patch/                  # Native Cosmos T2.5 notebook surface
│   ├── cosmos_t25_client.py             # HTTP client (replaces SMMG's T1 client)
│   ├── notebook_widgets_t25.py          # ipywidgets for T2.5 params
│   ├── patch_notebook.py                # Patches upstream notebook with T2.5 cells
│   └── generate_dataset_t25.ipynb       # Pre-patched notebook (output of patcher)
├── smmg_patches/                    # Runtime patches for IsaacLab/notebook drift
│   ├── force_headless.py                # Loads correct offscreen experience file
│   ├── patch_env_loop_call.py           # Updates env_loop signature for IsaacLab 2.2.1
│   ├── skip_empty_image_save.py         # Avoids Pillow crash on init-time empty obs
│   └── imageio_encode.py                # Replaces NVENC encode with software h264
├── scripts/                         # Drivers for Isaac Lab side
│   ├── activate_env.sh                  # Source to activate local env (non-Docker)
│   ├── smoke_test_isaaclab.sh           # 4-stage smoke test for the SMMG image
│   ├── smmg_smoke.py                    # Same, run inside container
│   └── robust_eval_record.py            # Patched robust_eval that saves rollout MP4s
├── docs/
│   └── PROJECT_PLAN.md                  # The 9-phase plan
└── README.md
```
## References

- [NVIDIA Synthetic Manipulation Motion Generation
  blueprint](https://github.com/NVIDIA-Omniverse-blueprints/synthetic-manipulation-motion-generation)
- [Isaac Lab — Augmented Imitation
  Learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/augmented_imitation.html)
- [Cosmos Transfer 2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)
- [Hugging Face — `nvidia/PhysicalAI-Robotics-Manipulation-Augmented`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Augmented)

## License

Mirrors NVIDIA's source licenses (Apache 2.0 / BSD-3-Clause for the
underlying frameworks). See individual upstream repos for terms. Cube-
stacking dataset is CC-BY-4.0 (NVIDIA).
