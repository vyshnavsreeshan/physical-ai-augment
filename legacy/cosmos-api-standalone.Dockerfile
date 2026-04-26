# cosmos-transfer2.5:standalone — self-contained Cosmos T2.5 API server.
#
# Inherits from cosmos-transfer2.5:blackwell-api (which already has Flask +
# cosmos_api.py) and bakes the cosmos-transfer2.5 source code INTO /workspace,
# so no runtime bind-mount is required.
#
# Result: pull this image, `docker run`, done. Only thing the host still
# provides is the HuggingFace cache mount (~30 GB of model weights).
#
# Works on any cu130-compatible NVIDIA GPU (Ampere / Ada / Hopper / Blackwell)
# despite the "blackwell" name in the parent tag — the NGC PyTorch base image
# ships multi-arch kernels.

ARG BASE_IMAGE=cosmos-transfer2.5:blackwell-api
FROM ${BASE_IMAGE}

# Bake source. The .dockerignore.cosmos-standalone excludes .venv / .git /
# docs / tests so we only carry runtime-needed files (~600 MB instead of 12 GB).
COPY third_party/cosmos-transfer2.5/ /workspace/

# Install cosmos_transfer2 at BUILD time instead of via the inherited
# nightly-entrypoint.sh.  This makes the image self-contained.
RUN pip install --no-deps -e /workspace

# Override the inherited entrypoint (which would re-run pip install -e on
# every container start, expecting a bind-mounted /workspace).
ENTRYPOINT []
CMD ["python3", "/opt/cosmos_api.py"]

EXPOSE 5000
