# legacy/

These files are from the project's first iteration when no public Cosmos
Transfer 2.5 inference service existed yet. We hand-rolled a Flask API
around the upstream `examples/inference.py` CLI so the SMMG-style notebook
could submit jobs over HTTP.

When NVIDIA published the **Cosmos Transfer 2.5 NIM**
(`nvcr.io/nim/nvidia/cosmos-transfer2.5-2b:1.0.0`) — a production-grade
inference microservice with persistent models and concurrent request
handling — the hand-rolled wrapper was no longer needed and the project
switched to NIM as the primary path.

## What's here

| File | Original purpose |
|---|---|
| `cosmos_api.py` | Flask server: spawned a fresh `python inference.py` subprocess per request. Single-worker queue with `submit → poll → fetch` semantics. |
| `cosmos-api.Dockerfile` | Wrapped `cosmos-transfer2.5:blackwell-nightly`, expected the cosmos-transfer2.5 source bind-mounted at `/workspace`. |
| `cosmos-api-standalone.Dockerfile` | Bakes the cosmos-transfer2.5 source into the image so it ships as one self-contained unit. Pushed to Docker Hub as `vyshnavsreeshan05/cosmos-transfer2.5:standalone`. |
| `.dockerignore.cosmos-standalone` | Excluded `.venv` / `.git` / `docs` / `tests` from the standalone image build context. |

## When you might still want this

- **No NGC access.** NIM requires an NGC API key (free, but gated).
  The legacy wrapper is auth-free.
- **Custom Cosmos build.** If you've forked cosmos-transfer2.5 and want to
  serve your fork, NIM doesn't help; the legacy wrapper does.
- **Air-gapped deployment.** Pull the standalone image from Docker Hub once,
  then run with no further internet access (NIM expects to download model
  weights into `/opt/nim/.cache` from NGC on first start).

For the active path (NIM) see the project root README.
