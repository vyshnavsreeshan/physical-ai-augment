# cosmos-transfer2.5:blackwell-api
#
# Extends the cosmos-transfer2.5:blackwell-nightly image we already built.
# Adds a Flask wrapper around `examples/inference.py` so other containers
# can submit T2.5 inference jobs over HTTP.
#
# Endpoint surface: see docker/cosmos_api.py docstring.

ARG BASE_IMAGE=cosmos-transfer2.5:blackwell-nightly
FROM ${BASE_IMAGE}

# Flask + multipart parsing
RUN pip install --no-cache-dir flask gunicorn

WORKDIR /workspace

COPY docker/cosmos_api.py /opt/cosmos_api.py

ENV COSMOS_REPO=/workspace \
    COSMOS_API_PORT=5000 \
    COSMOS_JOB_ROOT=/tmp/cosmos_jobs \
    COSMOS_PYTHON=python3

# IMPORTANT: keep the parent image's ENTRYPOINT (nightly-entrypoint.sh) which
# does `pip install --no-deps -e .` from the bind-mounted /workspace before
# exec'ing the CMD. Without this, `cosmos_transfer2` is not importable.
EXPOSE 5000
CMD ["python3", "/opt/cosmos_api.py"]
