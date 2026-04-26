#!/usr/bin/env bash
# Activate the Isaac Sim / Isaac Lab venv for this project.
# Usage: source scripts/activate_env.sh
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/env/bin/activate"
export ISAACLAB_PATH="${PROJECT_ROOT}/third_party/IsaacLab"
export PATH="${ISAACLAB_PATH}:${PATH}"
# Non-interactive EULA acceptance for Omniverse Kit (Isaac Sim 5.1).
# Equivalent to answering "Yes" to the first-run Omniverse EULA prompt.
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
# Reuse the existing HuggingFace cache shared with Cosmos Transfer.
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
echo "[env] Python:    $(which python)"
echo "[env] ISAACLAB:  ${ISAACLAB_PATH}"
