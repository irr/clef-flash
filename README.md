# clef-flash

Local [System One](https://huggingface.co/Cloudflare/clef-flash) server for
[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), a 9B multimodal
decision model that turns a state and a schema of typed questions (`noul`, `choice`,
`score`) into decisions — one forward pass, no text generation, no output parsing.

`run.sh` sets up a `uv` virtualenv, downloads the ~19 GB weights, and serves the
Jev-compatible `POST /v1/decisions` API.

## Question types

A request's `questions` object maps question IDs to one of three typed questions.
All are answered in a single forward pass.

| Type | Purpose | `criteria` | Answer |
|---|---|---|---|
| `noul` | Yes/no question | not needed | `noul`: probability in \[0, 1\] that the answer is yes (1.0 = yes, 0.0 = no) |
| `choice` | Pick one option from a finite set | object mapping option ID to its description | `choice` (option ID), `confidence`, and `probabilities` for every option |
| `score` | Rate on an ordered 0..N-1 scale | array of labels, index 0 = lowest | `score` (continuous value on the scale), `confidence`, `legend` (index to label), and `probabilities` per index |

All types require `instructions` (the question in natural language). Multiple
questions of any mix of types can be combined in one request; each is answered
independently in the same forward pass:

```json
{
  "is_down": {
    "type": "noul",
    "instructions": "Is a service down?"
  },
  "department": {
    "type": "choice",
    "instructions": "Which team should handle the message?",
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
```

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
- `POST /v1/decisions` (also `/v1/systemone`) — Jev-compatible decision endpoint

```sh
curl -s http://127.0.0.1:8000/v1/decisions \
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

## Example output

Output of `make test` against `http://192.168.1.198:8000` (state: "Checkout has
been failing for every customer for the last hour. Orders are blocked and support
is getting refund demands."):

| Question type | Request | Answer | Latency |
|---|---|---|---|
| `noul` — "Is a service down?" | 163 tokens | `noul: 0.9425` | 0.274s |
| `choice` — "Which team should handle the message?" | 180 tokens | `technical` (confidence 0.9804; billing 0.0134, sales 0.0062) | 0.264s |
| `score` — "How soon does this need a response?" (Can wait / This week / Today) | 167 tokens | `1.9583` ≈ Today (confidence 0.9752; 0.0170 / 0.0078 / 0.9752) | 0.273s |

Raw responses:

<details>
<summary><code>noul</code></summary>

```json
{
  "model": "clef-flash",
  "answers": {
    "outage": { "type": "noul", "noul": 0.9425 }
  },
  "usage": { "input_tokens": 163, "output_tokens": 0 }
}
```

</details>

<details>
<summary><code>choice</code></summary>

```json
{
  "model": "clef-flash",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "technical",
      "confidence": 0.9804,
      "probabilities": { "billing": 0.0134, "technical": 0.9804, "sales": 0.0062 }
    }
  },
  "usage": { "input_tokens": 180, "output_tokens": 0 }
}
```

</details>

<details>
<summary><code>score</code></summary>

```json
{
  "model": "clef-flash",
  "answers": {
    "urgency": {
      "type": "score",
      "score": 1.9583,
      "confidence": 0.9752,
      "legend": { "0": "Can wait", "1": "This week", "2": "Today" },
      "probabilities": { "0": 0.017, "1": 0.0078, "2": 0.9752 }
    }
  },
  "usage": { "input_tokens": 167, "output_tokens": 0 }
}
```

</details>

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
