# physical-ai-lab/smmg-blackwell:1.0
#
# Replacement for nvcr.io/nvidia/gr00t-smmg-bp:1.0 that works on Blackwell (sm_120).
# Brings up Isaac Sim 5.1 + Isaac Lab 2.2.1 + the SMMG notebook + Jupyter Lab,
# inside a CUDA 12.8 / Ubuntu 22.04 base.
#
# Why this image exists: NGC's SMMG image ships torch 2.5.1+cu118 baked into
# Isaac Sim 4.5's prebundle, which has no sm_120 kernels and cannot be cleanly
# upgraded (no 2.5.1+cu128 wheel exists). We rebuild on a newer stack.

ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    OMNI_KIT_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    ISAACLAB_PATH=/opt/IsaacLab \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# ── system deps ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl wget git git-lfs \
        ffmpeg build-essential cmake \
        libgl1 libglu1-mesa libxext6 libsm6 libxrender1 libxi6 libfontconfig1 \
        libxt6 libxmu6 libxinerama1 libxrandr2 libxcursor1 libxshmfence1 \
        libvulkan1 vulkan-tools libegl1 libgles2 \
        libglib2.0-0 libxkbcommon0 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# ── Isaac Sim 5.1 (matches our local env) ───────────────────────────────────
RUN python3.11 -m pip install --upgrade pip setuptools==79.0.1 wheel && \
    python3.11 -m pip install 'isaacsim[all,extscache]==5.1.0.0' \
        --extra-index-url https://pypi.nvidia.com

# ── Pre-fix flatdict: 4.0.1 setup.py needs pkg_resources, fails under
#    setuptools >= 80 build isolation. Installing without isolation pins
#    against our setuptools 79 in the venv. (Same fix we used locally.)
RUN python3.11 -m pip install flatdict==4.0.1 --no-build-isolation

# ── Isaac Lab v2.2.1 ───────────────────────────────────────────────────────
# Pulled from our snapshot fork (vyshnavsreeshan/IsaacLab) rather than upstream
# isaac-sim/IsaacLab so the build is reproducible — upstream main / tags can
# move; the fork is frozen at the v2.2.1 SHA we tested against.
RUN git -c advice.detachedHead=false clone --depth 1 --branch v2.2.1 \
        https://github.com/vyshnavsreeshan/IsaacLab.git ${ISAACLAB_PATH}

# isaaclab.sh --install pulls torch 2.7.0+cu128 (Blackwell-capable) and all
# extension dependencies. The installer detects the pip-installed Isaac Sim
# via the `isaacsim-rl` package check.
RUN TERM=xterm cd ${ISAACLAB_PATH} && \
    TERM=xterm ./isaaclab.sh --install

# Above sometimes leaves the core `isaaclab` package out due to flatdict
# already-installed quirk; run an explicit editable install to be safe.
RUN python3.11 -m pip install -e ${ISAACLAB_PATH}/source/isaaclab

# Patch IsaacLab's Blueprint env_cfg to skip the PNG-save block when the
# camera buffer is empty (which it always is during ObservationManager init —
# the framework calls each obs function once for shape determination before
# the first sim step, and Pillow crashes on a 0-sized tensor).
COPY smmg_patches/skip_empty_image_save.py /opt/skip_empty_image_save.py
RUN python3.11 /opt/skip_empty_image_save.py

# Helper script that adapts SMMG notebook calls to IsaacLab 2.2.1 API drift
# (env_loop now takes reset_queue between env and action_queue).
COPY smmg_patches/patch_env_loop_call.py /opt/patch_env_loop_call.py

# Replace SMMG's encode_video (uses NVENC via omni.videoencoding, fails on
# Blackwell with NV_ENC_ERR_INVALID_PARAM) with a software-h264 imageio
# version. Warp shading still runs on GPU; only encode moves to CPU.
COPY smmg_patches/imageio_encode.py /opt/imageio_encode.py

# Force headless=True in the AppLauncher cell so AppLauncher loads the
# correct offscreen-rendering experience file. Without this, the notebook's
# args_cli leaves headless at False, AppLauncher loads the display-mode
# experience file, then auto-degrades to --no-window when DISPLAY is unset
# — but the camera buffer never refreshes between sim steps in that mismatched
# state, producing stationary PNG dumps despite the env actually moving.
COPY smmg_patches/force_headless.py /opt/force_headless.py

# ── Jupyter + notebook deps ─────────────────────────────────────────────────
RUN python3.11 -m pip install \
        jupyterlab \
        ipywidgets \
        nest_asyncio \
        warp-lang \
        toml \
        imageio[ffmpeg]

# ── SMMG notebook + helpers + annotated dataset ────────────────────────────
# We bake in:
#   - SMMG's upstream notebook helpers (notebook_utils.py, notebook_widgets.py,
#     stacking_prompt.toml) — used unchanged
#   - the upstream annotated_dataset.hdf5
#   - OUR additions for Cosmos Transfer 2.5: cosmos_t25_client.py,
#     notebook_widgets_t25.py
#   - the patched notebook generate_dataset_t25.ipynb that hits our T2.5 API
WORKDIR /workspace
COPY third_party/synthetic-manipulation-motion-generation/notebook/notebook_utils.py /workspace/
COPY third_party/synthetic-manipulation-motion-generation/notebook/notebook_widgets.py /workspace/
COPY third_party/synthetic-manipulation-motion-generation/notebook/stacking_prompt.toml /workspace/
COPY third_party/synthetic-manipulation-motion-generation/samples/annotated_dataset.hdf5 /workspace/datasets/annotated_dataset.hdf5
COPY notebook_patch/cosmos_t25_client.py /workspace/
COPY notebook_patch/notebook_widgets_t25.py /workspace/
COPY notebook_patch/generate_dataset_t25.ipynb /workspace/generate_dataset.ipynb

# Apply notebook + utility patches (env_loop signature for IsaacLab 2.2.1,
# imageio-based encode_video for Blackwell). The COPY above brings in our
# already-patched generate_dataset_t25.ipynb so we only need to re-patch
# notebook_utils.py here.
RUN python3.11 /opt/imageio_encode.py

# Output directories the notebook expects in CWD
RUN mkdir -p /workspace/_isaaclab_out /workspace/_cosmos_out

EXPOSE 8888
ENTRYPOINT ["python3.11", "-m", "jupyter", "lab", \
    "/workspace/generate_dataset.ipynb", \
    "--allow-root", "--ip=0.0.0.0", "--no-browser", \
    "--NotebookApp.token=", "--NotebookApp.password=", \
    "--NotebookApp.default_url=/tree/generate_dataset.ipynb"]
