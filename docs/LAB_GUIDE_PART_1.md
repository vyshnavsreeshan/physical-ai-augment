# Lab Guide — Part 1
## Synthetic Trajectory Generation + Photoreal Augmentation
### Isaac Sim · Isaac Lab Mimic · Cosmos Transfer 2.5 NIM

---

## Objective

By the end of Part 1 you will have, with one click in a Jupyter notebook:

1. **Synthesized** a brand-new robot manipulation trajectory in Isaac Sim from
   a small seed dataset (10 human demos) using Isaac Lab Mimic
2. **Encoded** that trajectory as a hybrid "shaded-segmentation" control video
   (semantic segmentation × surface-normal Lambertian shading)
3. **Augmented** the trajectory through the **NVIDIA Cosmos Transfer 2.5**
   diffusion model — re-rendering the same robot motion as a photoreal video
   under any text prompt you compose ("rustic copper cubes in a university
   laboratory", "marble cubes in a high-tech cleanroom", etc.)

This is the **data-generation half** of an augmented imitation-learning
pipeline. Part 2 (subsequent labs) trains visuomotor policies on these
augmented datasets and evaluates robustness across visual perturbations.

---

## Lab Description

Robot imitation learning is bottlenecked by the cost of human-collected
demonstrations. Two compounding problems:

- **Quantity:** A few hundred teleoperated demos isn't enough to train a
  robust visuomotor policy.
- **Visual diversity:** Whatever lighting, table material, and background
  was present at collection time is *all* the policy learns to handle.
  Deploy the same policy in a slightly different room and it fails.

NVIDIA addresses both with a two-stage synthetic data pipeline:

```
   10 human demos
        │
        ▼ ┌──────────────── Stage 1: Mimic ─────────────────┐
          │ Spatial transformation + replay in Isaac Sim    │
          │ → 1,000s of physically-valid trajectories       │
          └──────────────────┬──────────────────────────────┘
                             │
                             ▼ ┌────── Stage 2: Cosmos Transfer 2.5 ─────┐
                               │ Same robot motion, any photoreal scene  │
                               │ → unlimited visual diversity per traj   │
                               └─────────────────────────────────────────┘
```

This lab implements both stages on a Blackwell-clean stack — Isaac Sim 5.1 +
Isaac Lab v2.2.1 + torch 2.7.0+cu128 — replacing the original SMMG blueprint
which only worked on Hopper or older. The two stages run in **separate
Docker containers** that talk over HTTP, so each can scale independently and
the heavyweight Cosmos model can live on a different machine if you have
limited RAM on the Isaac Sim host.

---

## Key Concepts

- **Imitation learning** — A policy learns by mimicking expert action
  sequences instead of being trained via reward. Each demo is a (state,
  action) trajectory; the policy learns `π(action | state)`.

- **Visuomotor policy** — The policy takes camera pixels (and proprioception)
  as state and outputs robot actions. Susceptible to visual distribution
  shift: train on one lighting/texture, the policy's image features go off
  the rails when the test scene looks different.

- **Isaac Sim** — NVIDIA's GPU-accelerated robotics simulator. Built on
  Omniverse Kit. Provides physics (PhysX), photoreal rendering (RTX/Vulkan),
  and a USD scene graph.

- **Isaac Lab** — A robotics-learning framework built on top of Isaac Sim.
  Provides standardized envs (Gymnasium-compatible), managers (Observations,
  Actions, Events, Recorders, Terminations, Curriculum, Rewards), and
  utility scripts (Mimic, Robomimic integration, robust evaluation).

- **Isaac Lab Mimic** — A trajectory-multiplication tool. Given a seed
  dataset of demos *with subtask annotations*, it spatially transforms and
  re-stitches the segments into trajectories that match new randomized
  scenes (different cube positions, different start poses), replays them
  in the simulator, and keeps the ones that succeed.

- **Cosmos Transfer 2.5** — A controlnet-style diffusion video model. Given
  an input video (used as a control signal: edges / depth / segmentation /
  blur) plus a text prompt, it generates a new video that follows the
  geometry of the control but takes its appearance from the prompt. Same
  motion, different look.

- **Shaded segmentation** — The control input we feed to Cosmos. Instead
  of raw RGB (too constrained — Cosmos overfits to source colors) or flat
  semantic segmentation (too sparse — no depth cues), we multiply the
  segmentation colors by `0.5 + 0.5 · dot(normal, light_direction)`. This
  hybrid has crisp object edges (from segmentation) AND 3D shape cues (from
  normals), which is exactly what the diffusion model needs to preserve
  geometry while the prompt drives appearance.

- **NIM (NVIDIA Inference Microservice)** — NVIDIA's pre-packaged
  inference servers for foundation models. Persistent in-memory model
  loading + Triton-backed concurrent request handling + standard HTTP
  endpoints. We use the Cosmos Transfer 2.5 NIM
  (`nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:1.0.0`).

- **Two-container architecture** — SMMG (Mimic) and Cosmos run in separate
  containers that communicate over HTTP. Decouples generation from
  augmentation, lets you put Cosmos on a beefier host (Hopper / Blackwell
  H100/H200) while the Isaac Sim side stays on a workstation.

---

## Setup — SMMG Container

**Image:** `physical-ai-lab/smmg-blackwell:1.0` (built locally; ~54 GB)

### What's inside

| Component | Version / details |
|---|---|
| Base | `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` |
| Python | 3.11 (via deadsnakes) |
| Isaac Sim | 5.1.0 (`pip install isaacsim[all,extscache]==5.1.0.0`) |
| Isaac Lab | v2.2.1 cloned + `./isaaclab.sh --install` |
| torch | 2.7.0+cu128 (Blackwell sm_120 capable) |
| Jupyter Lab | latest |
| Warp (NVIDIA) | for the shading kernel |
| imageio + ffmpeg | software h264 encoding |
| Notebook + helpers | `/workspace/generate_dataset.ipynb`, `cosmos_t25_client.py`, `notebook_widgets_t25.py`, `notebook_utils.py`, `notebook_widgets.py`, `stacking_prompt.toml` |
| Seed dataset | `/workspace/datasets/annotated_dataset.hdf5` (10 human teleop demos with subtask boundaries) |
| Runtime patches | applied at build time (see below) |

### Runtime patches applied at build time

Five patches handle drift between SMMG's original Isaac Lab 2.0.2 / Isaac Sim
4.5 reference target and our newer Isaac Sim 5.1 + Isaac Lab 2.2.1 stack:

| Patch | What it fixes |
|---|---|
| `flatdict==4.0.1 --no-build-isolation` | flatdict's `setup.py` imports `pkg_resources` which setuptools 80+ removed; build under our setuptools 79 instead |
| `skip_empty_image_save.py` | `image()` obs function's `save_image` crashed on the 0-sized tensor used for shape probing during ObservationManager init (Pillow 11+ stricter on tile size) |
| `patch_env_loop_call.py` | Isaac Lab 2.2.1's `env_loop` adds a new `reset_queue` arg between `env` and `action_queue`; the upstream notebook's call site is updated automatically |
| `force_headless.py` | Without explicit `args_cli.headless = True`, AppLauncher loads the display-mode experience file then auto-degrades to `--no-window` — but with the wrong experience file, camera buffers don't refresh between sim steps and PNG dumps are stationary |
| `imageio_encode.py` | Replaces `omni.videoencoding`'s NVENC wrapper (fails on Blackwell with `NV_ENC_ERR_INVALID_PARAM`) with software libx264 via imageio. Warp shading still runs on GPU — only encoding moves to CPU. |

### How to build

```bash
# from the project root
docker build -f docker/smmg-blackwell.Dockerfile \
    -t physical-ai-lab/smmg-blackwell:1.0 .
# ~30 min the first time (Isaac Sim wheel download is the bulk)
```

### How to run

```bash
docker run -d --name smmg-lab \
    --runtime=nvidia --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e COSMOS_API_URL=http://<cosmos-host>:8000 \
    -p 8888:8888 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    physical-ai-lab/smmg-blackwell:1.0
```

Then open http://localhost:8888/lab — the patched notebook auto-opens.

---

## Setup — Cosmos NIM Container

**Image:** `nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:1.0.0` (pulled from NGC; ~30 GB)

### What's inside

| Component | Purpose |
|---|---|
| Cosmos-Transfer2.5-2B | The controlnet model — accepts edge / depth / segmentation / blur control video + prompt → photoreal output |
| Cosmos-Predict2.5-2B | Base video diffusion model (the world model the controlnet steers) |
| Cosmos-Reason1-7B | Text encoder + content-safety reasoner (analyzes your prompt for unsafe content) |
| Cosmos-Guardrail1 | Output-side video safety filter |
| Qwen3Guard, RetinaFace, Siglip, Grounding-DINO | Additional safety / control utilities |
| Wan2.1 VAE | Decodes the diffusion latents to RGB frames |
| Triton Inference Server (pyTriton) | Hosts each model as a Triton model; routes infer calls |
| FastAPI / uvicorn | Public HTTP surface (`/v1/infer`, `/v1/health/*`, `/v1/metrics`) |

### Why a NIM and not a custom server

We started Part 1 with a hand-rolled Flask server around the upstream
`examples/inference.py` CLI (kept in `legacy/` for reference). It worked
but spawned a fresh subprocess per request, reloading 22 GB of model
weights each time — ~14 min/job dominated by that load tax.

NVIDIA's NIM container does the right thing: loads models once at startup,
keeps them resident in VRAM, and serves concurrent requests through Triton.
Same per-job inference work but no per-job overhead, and the architecture
matches what NVIDIA recommends for production.

### How to deploy

#### 1. Get an NGC API key (free, one-time)

- Sign in at https://org.ngc.nvidia.com
- Generate API Key at https://org.ngc.nvidia.com/setup/api-key
- Copy the `nvapi-...` string

#### 2. Authenticate Docker to NGC

```bash
export NGC_API_KEY=nvapi-...
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

#### 3. Pull and run

```bash
docker pull nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:1.0.0  # ~30 GB pull

mkdir -p ~/.cache/nim && chmod -R 777 ~/.cache/nim 2>/dev/null || true

docker run -d --name cosmos-nim --restart unless-stopped \
    --runtime=nvidia --gpus all \
    --shm-size=32GB --ulimit nofile=65536:65536 \
    -e NGC_API_KEY=$NGC_API_KEY \
    -v ~/.cache/nim:/opt/nim/.cache \
    -p 8000:8000 \
    nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:1.0.0
```

#### 4. Wait for ready

First start downloads ~30 GB of model weights into `~/.cache/nim/`,
loads them to VRAM. ~5–10 min on a fast link.

```bash
docker logs -f cosmos-nim 2>&1 | grep -E "ready|Application is ready|error"
# look for: "Welcome! Application is ready to receive API requests."

curl http://localhost:8000/v1/health/ready
# expect: {"object":"Triton readiness check","message":"ready","status":"ready"}
```

### Two-machine deployment

For best performance, run the NIM on a separate Hopper/Blackwell host
(H100, H200, RTX 4090, etc.) so SMMG's Isaac Sim doesn't compete for RAM.
On the Isaac Sim host, set `COSMOS_API_URL=http://<nim-host>:8000` and the
notebook will use it transparently.

---

## Notebook Walkthrough — `generate_dataset.ipynb`

Open http://localhost:8888/lab and execute cells top-to-bottom. The notebook
has **two halves**: cells 1–4 generate a Mimic trajectory, cells 5–11 augment
it with Cosmos.

### Cell 1 — Configure number of trials

```python
from notebook_widgets import create_num_trials_input
num_envs = 1
num_trials = create_num_trials_input()
```

**What it does:** Renders an `IntText` widget where you set how many
successful trajectories Mimic should generate.

**Recommended value:** `1` for a first end-to-end sanity check (≈30–60 s of
generation). For a real fine-tune dataset, raise to 5–20.

**Outcome:** A "Number of trials" input box. The variable `num_trials.value`
will be read by Cell 2.

---

### Cell 2 — Spin up Isaac Sim and create the env

```python
parser = ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args([])
args_cli.enable_cameras = True
args_cli.headless = True       # ← critical patch: forces correct experience file
...
config = {
    "task": "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0",
    "num_envs": num_envs,
    "generation_num_trials": num_trials.value,
    "input_file": "datasets/annotated_dataset.hdf5",
    "output_file": "datasets/generated_dataset.hdf5",
    ...
}
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
...
env = gym.make(env_name, cfg=env_cfg).unwrapped
```

**What it does:**
- Boots Omniverse Kit (`isaaclab.python.headless.rendering.kit` experience
  file — selected because we set `headless=True`)
- Loads the cube-stacking scene with three cubes, a Franka arm, table, and
  lighting
- Constructs the Mimic-flavored env which knows how to spatially transform
  source demos to fit the randomized scene

**Expected outcome (in the cell output area):**
```
[INFO][AppLauncher]: Loading experience file: /opt/IsaacLab/apps/isaaclab.python.headless.rendering.kit
| Driver Version: ... | Graphics API: Vulkan
| GPU 0 | NVIDIA RTX PRO 6000 Blackwell (or H200, etc.)
[INFO]: Time taken for scene creation : 2-5 seconds
[INFO] Action Manager: <ActionManager> contains 2 active terms.
[INFO] Observation Manager: <ObservationManager> contains 3 groups.
[INFO]: Completed setting up the environment...
```

**Time:** 3–5 minutes the first time (Vulkan compiles shaders, USD scene
loads); subsequent reloads are faster but only reachable via kernel restart.

---

### Cell 3 — Adjust randomization

```python
randomizable_params = {
    "randomize_franka_joint_state": {"mean": (0.0, 0.5, 0.01), "std": (0.0, 0.1, 0.01)},
    "randomize_cube_positions": {"pose_range": {"x": (0.3, 0.9, 0.01), "y": (-0.3, 0.3, 0.01)},
                                 "min_separation": (0.0, 0.5, 0.01)},
}
for i in range(len(env.unwrapped.event_manager._mode_term_cfgs["reset"])):
    event_term = env.unwrapped.event_manager._mode_term_cfgs["reset"][i]
    name = env.unwrapped.event_manager.active_terms["reset"][i]
    if name not in randomizable_params:        # ← skip-unknown patch
        continue
    interactive_update_randomizable_params(event_term, name, randomizable_params[name], env=env)
```

**What it does:** Renders a slider for each randomization parameter — Franka
joint-pose noise (mean + std), cube spawn ranges (x, y), minimum separation
between cubes.

**Note:** Isaac Lab 2.2.1 added a new reset event (`init_franka_arm_pose`)
not in the original SMMG dict; our skip-unknown patch silently ignores it.

**Recommended:** Accept defaults for the first run.

**Outcome:** Five sliders display showing current values and allowed ranges.
Adjustments take effect on the *next* env reset.

---

### Cell 4 — Generate the trajectory (Mimic)

```python
async_components = setup_async_generation(
    env=env, num_envs=args_cli.num_envs,
    input_file=args_cli.input_file,
    success_term=success_term, pause_subtask=args_cli.pause_subtask,
)
future = asyncio.ensure_future(asyncio.gather(*async_components['tasks']))
env_loop(env, async_components['reset_queue'], async_components['action_queue'],   # ← signature patch
        async_components['info_pool'], async_components['event_loop'])
```

**What it does:** Two concurrent coroutines:
- A *generator* coroutine takes a source demo from the seed dataset,
  spatially transforms its subtask segments to land each cube at the new
  randomized pose, and pushes the resulting actions onto a queue.
- An *executor* (`env_loop`) drains actions from the queue, calls
  `env.step(actions)` to advance the physics + render the cameras, records
  observations and actions to disk, and signals success/failure.

**Expected outcome:**
```
Loaded 10 to datagen info pool
**************************************************
1/N (X%) successful demos generated by mimic
**************************************************
Reached 1 successes/attempts. Exiting.
Tasks were properly cancelled during cleanup.
```

Generation writes:
- `datasets/generated_dataset.hdf5` (low-dim states + actions)
- `_isaaclab_out/*.png` (per-frame `table_cam_normals_*`, `table_cam_semantic_segmentation_*`,
  `table_high_cam_*` — ~250 PNGs per trial × 4 modality streams)

**Time:** 30–60 s per trial after the env is warm.

---

### Cell 5 — Pick a camera

```python
from notebook_widgets import create_camera_input
VIDEO_LENGTH = 120
camera_selection = create_camera_input(ISAACLAB_OUTPUT_DIR)
```

**What it does:** Scans `_isaaclab_out/` for camera prefixes (e.g.
`table_cam`, `table_high_cam`) and renders a dropdown.

**Recommended:** `table_cam` — that's the camera the published Robomimic
policies were trained on, so any later fine-tune lines up.

**Outcome:** A dropdown widget. Pick a value, run the next cell.

---

### Cell 6 — Encode shaded-segmentation MP4 (input to Cosmos)

```python
env_trial_frames = get_env_trial_frames(ISAACLAB_OUTPUT_DIR, camera_selection.value, 10)
camera = camera_selection.value
for env_num, trial_nums in env_trial_frames.items():
    for trial_num, (start_frame, end_frame) in trial_nums.items():
        ...
        encode_video(ISAACLAB_OUTPUT_DIR, video_start, VIDEO_LENGTH,
                     camera, video_filepath, env_num, trial_num)
        display(Video(video_filepath, width=1000))
```

**What it does:**
- Picks the **last 120 frames** of the chosen trial (the most action-dense
  segment — typically the cube-placing motion)
- For each frame, loads the segmentation PNG + normals PNG into GPU
- Runs a Warp kernel: `shade = 0.5 + 0.5 * dot(normal, light_dir)`,
  multiplies the segmentation colors by `shade`
- Encodes the resulting frames as h264 MP4 via imageio/ffmpeg (note: our
  patch replaced the upstream NVENC path which fails on Blackwell)

**Outcome:**
- New file: `_isaaclab_out/shaded_segmentation_<camera>_trial_<N>_tile_<E>.mp4`
- Inline video preview shows a 5-second cel-shaded clip: solid colored
  table + cubes + arm with Lambertian shading from a top-down light. Crisp
  edges, clear 3D shape, no real-world textures.

**Time:** ~1–2 sec encoding (Warp shading on GPU is instant; libx264 is
the bottleneck).

---

### Cell 7 — Cosmos NIM URL widget

```python
default_url = os.environ.get("COSMOS_API_URL", "http://cosmos-nim:8000")
url_widget = widgets.Text(value=default_url, ..., description="Cosmos NIM URL:")
display(url_widget)
try:
    from cosmos_t25_client import healthz
    print("Cosmos NIM health:", healthz(url_widget.value))
except Exception as e:
    print(f"(Cosmos NIM unreachable yet — ...): {e}")
```

**What it does:** Renders a text input pre-filled with the NIM URL (defaults
to the docker-compose service name; override if NIM is on a different host)
and immediately probes `/v1/health/ready`.

**Outcome:**
```
Cosmos NIM health: {'object': 'Triton readiness check', 'message': 'ready', 'status': 'ready'}
```

If unreachable: check that NIM container is up (`docker ps`), port 8000 is
forwarded, and any network firewall lets the SMMG host reach the Cosmos host.

---

### Cell 8 — Compose prompt + Cosmos parameters

```python
from notebook_widgets import create_variable_dropdowns
from notebook_widgets_t25 import create_t25_params
prompt_manager = create_variable_dropdowns("stacking_prompt.toml")  # cube × table × location
cosmos_params = create_t25_params(ISAACLAB_OUTPUT_DIR)              # seed/guidance/num_steps/etc.
for w in cosmos_params.values():
    display(w)
```

**What it does:** Builds two widget groups:
- **Prompt mixer** (from `stacking_prompt.toml`): three dropdowns —
  cube material × table material × location — plus a live preview of the
  combined prompt
- **T2.5 params**: input video dropdown, seed, guidance (0–7), num_steps
  (4–80), sigma_max (0–80), resolution (256/480/512/720), and per-control
  weight sliders for edge / depth / seg / vis (blur)

**Recommended starting values:**
- Pick anything fun: `glass cubes` / `marble table` / `high-tech cleanroom`
- guidance=3, num_steps=35, sigma_max=70, resolution=720
- edge.control_weight=1.0, all others 0

**Outcome:** The prompt updates live as you change dropdowns. Param sliders
ready for submission.

---

### Cell 9 — Submit to Cosmos NIM

```python
call = widgets_to_t25_call_kwargs(cosmos_params)
input_video_name = call.pop("_input_video_filename")
video_filepath = os.path.join(ISAACLAB_OUTPUT_DIR, input_video_name)
output_path = os.path.join(COSMOS_OUTPUT_DIR, f"cosmos_t25_seed{call['seed']}.mp4")
result = transfer(api_url=url_widget.value, video_path=video_filepath,
                  output_path=output_path, prompt=prompt_manager.prompt, **call)
display(Video(result["output_path"], width=900))
```

**What it does:**
- Reads the shaded-segmentation MP4 into memory
- Base64-encodes the bytes, builds a JSON request body (prompt, video,
  controls, seed, guidance, num_steps, sigma_max, resolution,
  num_conditional_frames)
- POSTs to `http://<nim-host>:8000/v1/infer`
- **Blocks** waiting for the response (NIM is sync)
- Decodes the returned `b64_video` field and writes the MP4 to disk
- Inline-plays the result via `IPython.display.Video`

**Expected outcome:**
```
[cosmos] POST http://10.79.252.45:8000/v1/infer  body={'prompt': '...', 'video': '<b64 video, 570K bytes>',
       'resolution': '720', 'guidance': 3, 'num_steps': 35, 'num_conditional_frames': 1,
       'seed': 2025, 'edge': {'control_weight': 1.0}}
[cosmos] saved → _cosmos_out/cosmos_t25_seed2025.mp4 (~500 KB, ~600s, seed=2025)
```
Plus the inline video player showing a photoreal version of your scene.

**Time:** **~14 minutes** for a 35-step / 720p / 120-frame submission. Most
of that is diffusion (35 × ~6 sec/step × 2 chunks for >93 frames). The
client polls just once and waits — there's no submit/poll dance because
NIM is synchronous.

---

## Lab Summary

You now have a working two-stage synthetic data pipeline:

```
seed (10 demos)        Mimic                     Cosmos T2.5
        │           generates new           rerenders trajectory
        │           trajectory in           as photoreal video
        │           Isaac Sim               under text prompt
        │                                                ▼
   annotated_dataset    →  generated_dataset.hdf5  →  cosmos_t25_seedN.mp4
   .hdf5                +  shaded_segmentation_*    (and as many more
   (input)                 .mp4 (intermediate)       as you want)
```

Each notebook run produces:
- 1 trajectory in HDF5 with low-dim observations and actions
- ~250 per-modality PNG frames in `_isaaclab_out/`
- 1 shaded-segmentation MP4 in `_isaaclab_out/`
- 1 photoreal MP4 in `_cosmos_out/`

To scale into a fine-tunable dataset, repeat the loop with different
randomization seeds and different prompts. Part 2 of the lab covers the
batch driver, the fine-tune itself, and robust evaluation.

### Performance numbers (H200 — your mileage will vary)

| Step | Time | Notes |
|---|---|---|
| First-time NIM startup | ~5–10 min | downloads + loads ~30 GB of weights |
| First-time Isaac Sim boot | ~3–5 min | Vulkan shader compile + USD scene load |
| Mimic trajectory synthesis (per trial) | 30–60 s | physics + replay |
| Shaded-segmentation encode | 1–2 s | Warp kernel + libx264 |
| Cosmos inference (35 steps × 720p × 120 frames) | ~14 min | dominated by 70 diffusion steps across 2 chunks |
| Cosmos inference (8 steps × 480p × 120 frames) | ~90 s | useful for fast iteration |

### Speed/quality knobs

| Knob | Effect |
|---|---|
| `num_steps` 35 → 16 | ~2 min off diffusion, minor quality loss |
| `resolution` 720 → 480 | ~3× faster diffusion, lower spatial detail |
| `sigma_max` 70 → 50 | tighter adherence to control video, less prompt expression |
| `edge.control_weight` 1.0 → 0.6 | less geometry preservation, more prompt freedom |

---

## Troubleshooting Guide

These are the actual issues we hit during development, in roughly the order
you'd encounter them.

### Container won't start

| Symptom | Cause | Fix |
|---|---|---|
| `pull access denied` for `physical-ai-lab/smmg-blackwell:1.0` | Image not built locally | `docker build -f docker/smmg-blackwell.Dockerfile -t physical-ai-lab/smmg-blackwell:1.0 .` |
| `pull access denied` for `nvcr.io/nim/nvidia/cosmos-transfer2.5-2b` | Not logged in to NGC | `echo $NGC_API_KEY \| docker login nvcr.io --username '$oauthtoken' --password-stdin` |
| NIM container crashloops with no logs | Forgot to set `NGC_API_KEY` env | `docker run -e NGC_API_KEY=$NGC_API_KEY ...` |

### Isaac Sim crash inside SMMG container

| Symptom | Cause | Fix (already baked in our image) |
|---|---|---|
| `sm_120 is not compatible` | Blackwell needs cu128 torch | Dockerfile uses torch 2.7.0+cu128 |
| `VkResult: ERROR_INCOMPATIBLE_DRIVER` | Vulkan capability not exposed | Dockerfile sets `NVIDIA_DRIVER_CAPABILITIES=all` |
| `libXt.so.6: cannot open shared object` | X11 dev libs missing | Dockerfile installs `libxt6` etc. |
| `tile cannot extend outside image` (PIL) | Empty obs tensor at init | `skip_empty_image_save.py` patch |
| `env_loop() missing 1 required positional argument` | IsaacLab 2.2.1 API drift | `patch_env_loop_call.py` patch |

### Cosmos / NIM issues

| Symptom | Cause | Fix |
|---|---|---|
| `404 Not Found` on `/v1/transfer` | Old client URL | Use the NIM client (rewritten) — endpoint is `/v1/infer` |
| `"Extra inputs are not permitted"` for `guidance_scale`/`steps` | Wrong field names | NIM uses `guidance` and `num_steps` (the ones in our current client) |
| `Access denied. This repository requires approval.` (HF) | Cosmos T2.5 weights are gated | Set `HF_TOKEN` on NIM container (or skip — NIM downloads from NGC by default, not HF) |
| `NV_ENC_ERR_INVALID_PARAM` during encode | `omni.videoencoding`'s NVENC fails on Blackwell | `imageio_encode.py` patch (libx264 software) |
| Inline video player shows nothing | Cell still running — NIM takes ~14 min for 35-step / 720p | Wait. If completed but still blank: codec issue. Check `_cosmos_out/` directly. |

### Mimic generation issues

| Symptom | Cause | Fix |
|---|---|---|
| `0/N successful demos generated by mimic` | Cube placement out of reach for source demos | Tighten `randomize_cube_positions.pose_range.x/y` in cell 3 |
| Stationary frames in encoded video (all PNGs identical) | AppLauncher loaded wrong experience file (display-mode + auto `--no-window`) | `force_headless.py` patch — explicitly sets `args_cli.headless = True` |
| `KeyError: 'init_franka_arm_pose'` | New IsaacLab 2.2.1 reset event | Notebook patched to skip unknown event names |

### Out-of-memory

| Symptom | Cause | Fix |
|---|---|---|
| `oom_killed=true` on Cosmos container | Both Isaac Sim + Cosmos peak load > host RAM | Two-machine deployment (Cosmos on a separate Hopper/Blackwell host) |
| Process killed during model load | Swap exhausted on small-RAM host | Add 32 GB swap, or move Cosmos to a bigger box |

### Stale Jupyter kernel state

| Symptom | Cause | Fix |
|---|---|---|
| `from cosmos_t25_client import transfer` returns the old function after editing the .py file | Python's import cache | `importlib.reload(cosmos_t25_client); from cosmos_t25_client import transfer` (or kernel restart) |
| `KeyError: 'vis_weight'` on submit | Widget dict built by older `create_t25_params` | Re-run the params widget cell to rebuild |

---

## Additional References

### NVIDIA documentation
- **NIM for Cosmos quickstart** — https://docs.nvidia.com/nim/cosmos/latest/quickstart-guide.html
- **Synthetic Manipulation Motion Generation blueprint** — https://github.com/NVIDIA-Omniverse-blueprints/synthetic-manipulation-motion-generation
- **Isaac Lab — Augmented Imitation Learning** — https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/augmented_imitation.html
- **Cosmos Transfer 2.5 source** — https://github.com/nvidia-cosmos/cosmos-transfer2.5
- **Cosmos Cookbook** — https://github.com/nvidia-cosmos/cosmos-cookbook

### Datasets & checkpoints
- **NVIDIA's pre-augmented Mimic dataset (CC-BY-4.0)** — https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Augmented
- **Cosmos-Transfer2.5-2B model card** — https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B
- **Cosmos-Reason1-7B (text encoder + safety reasoner)** — https://huggingface.co/nvidia/Cosmos-Reason1-7B

### Tooling
- **Robomimic (BC training framework used in Part 2)** — https://robomimic.github.io
- **NVIDIA Warp (Python-callable CUDA kernels)** — https://github.com/NVIDIA/warp
- **Isaac Sim docs** — https://docs.isaacsim.omniverse.nvidia.com
- **Isaac Lab docs** — https://isaac-sim.github.io/IsaacLab/

### This project
- **Repository** — https://github.com/vyshnavsreeshan/physical-ai-augment
- **Standalone Cosmos image (legacy / no-NGC fallback)** — `vyshnavsreeshan05/cosmos-transfer2.5:standalone`

---

*Lab Guide Part 1 — synthetic trajectory generation + photoreal augmentation.
Part 2 covers batch augmentation, fine-tuning Policy B, and robust evaluation.*
