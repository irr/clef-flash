# clef-flash

Local [System One](https://huggingface.co/Cloudflare/clef-flash) server for
[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), a 9B multimodal
decision model that turns a state and a schema of typed questions (`noul`, `choice`,
`score`) into decisions — one forward pass, no text generation, no output parsing.

`run.sh` sets up a `uv` virtualenv, downloads the ~19 GB weights, and serves the
Jev-compatible `POST /v1/systemone` API.

## Quick start

```sh
make run          # or: ./run.sh
```

Requires [uv](https://docs.astral.sh/uv/) (installed automatically if missing) and
Python 3.12.

### Targets

| Target | Description |
|---|---|
| `make run` | Start the server (override with `make run PORT=9000`) |
| `make smoke` | Load the model and run one in-process request, no server |
| `make debug` | Start the server with request/prompt/latency logging |
| `make test` | Send one request per question type (server must be running) |

### Environment

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `CLEF_DEVICE` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `MODEL_DIR` | `./models/clef-flash` | Local weights directory |
| `DEBUG` | `0` | `1` to log requests, rendered prompts, and latency |

## API

- `GET /health`, `GET /v1/models` — server status
- `POST /v1/systemone` (also `/v1/decisions`) — Jev-compatible decision endpoint

```sh
curl -s http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "clef-flash",
    "state": "Checkout is failing for every customer.",
    "questions": {
      "urgent": {"type": "noul", "instructions": "Is this urgent?"},
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments, invoices, or refunds",
          "technical": "Bugs, outages, or blocked orders"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "How soon does this need a response?",
        "criteria": ["Can wait", "This week", "Today"]
      }
    }
  }'
```

Responses contain `answers` keyed by question ID with probabilities for every
allowed option, plus `usage`. Optional `images` and `videos` may be added to requests.

## ROCm (Strix Halo / gfx1151)

On Radeon 8060S-class GPUs (Ryzen AI Max, GMKtec EVO-X2), the pytorch.org ROCm
wheel segfaults. `run.sh` installs TheRock gfx1151 wheels from
`https://rocm.nightlies.amd.com/v2/gfx1151/` and leaves `HSA_OVERRIDE_GFX_VERSION`
unset. Verified: torch 2.10.0+rocm7.13.0a20260513, device "Radeon 8060S Graphics".

## Files

| File | Purpose |
|---|---|
| `run.sh` | Environment setup, model download, server start |
| `serve_clef_flash.py` | HTTP server wrapping `systemone` |
| `test.sh` | One request per question type against a running server |
