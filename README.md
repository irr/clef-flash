# clef-flash

A local, self-contained server for [**Cloudflare/clef-flash**](https://huggingface.co/Cloudflare/clef-flash)
— a 9B multimodal **decision** model in the [System One](https://huggingface.co/Cloudflare/clef-flash)
family. You give it a `state` (the situation, as JSON or text) and a schema of typed
questions; it returns a probability distribution over each question's allowed options.

It is **not a chat model**. There is no text generation, no sampling, and no output
parsing: the model runs **one forward pass** and emits logits, and the server formats
those logits into the Jev/SystemOne JSON envelope. That is why answers come back in
~0.3 s on a Radeon 8060S even though the model is 9B parameters.

`run.sh` sets up a `uv` virtualenv, installs a matching torch build, downloads the
~19 GB weights, and serves `POST /v1/decisions`.

## Contents

- [Quick start](#quick-start)
- [Question types](#question-types)
- [How it works](#how-it-works)
- [Step-by-step execution](#step-by-step-execution)
- [API](#api)
- [Worked example](#worked-example)
- [Configuration](#configuration)
- [ROCm (Strix Halo / gfx1151)](#rocm-strix-halo--gfx1151)
- [Troubleshooting](#troubleshooting)
- [Files](#files)

## Quick start

```sh
make run                                    # start the server (downloads weights on first run)
curl -s http://127.0.0.1:8000/health        # wait for {"status":"ok", ...}
make test                                   # one request per question type
```

Requirements:

- [uv](https://docs.astral.sh/uv/) (installed automatically if missing)
- Python 3.12 (uv fetches it)
- ~19 GB of disk for the weights, plus a few GB for torch
- Optional: a CUDA / ROCm / MPS GPU. CPU works but is slow

## Question types

A request's `questions` object maps a **question ID** (any string you choose, echoed
back as the answer key) to one of three typed questions. Every question requires
`instructions` — the question in natural language. All questions in a request are
answered **independently, in the same single forward pass**.

| Type | Purpose | `criteria` | Answer fields |
|---|---|---|---|
| `noul` | Yes/no question | optional, usually omitted | `noul`: probability your proposition is true, in `[0, 1]` |
| `choice` | Pick exactly one option | **required**: object mapping option ID → description | `choice` (winning ID), `confidence`, `probabilities` per option |
| `score` | Rate on an ordered scale | **required**: array of labels, index `0` = lowest | `score` (fractional, `0..N-1`), `confidence`, `legend` (index → label), `probabilities` per index |

All three types in one `questions` object:

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

Notes on the types:

- **`noul`** — the name means "no/yes" (as in null-or-not). The model is always given
  exactly two options, `true` and `false`, with descriptions injected automatically;
  you get back only `p(true)`, so `p(false) = 1 − noul`. It is the only type with no
  `probabilities` block and no `confidence`. You cannot change the option IDs, but you
  can pass a `criteria` object with `true`/`false` keys to override their descriptions.
- **`choice`** — criteria keys are the option IDs and are **sorted** before rendering,
  so option order in the prompt is alphabetical, not insertion order. The returned
  `choice` is the argmax.
- **`score`** — your array is an ordered scale, not a set. `score` is the
  **expected value** of the distribution, so it can be fractional (`1.96` on a
  `0..2` scale) even though the model only ever "voted" on integers. Use `confidence`
  when you need a hard class; use `score` when an ordinal position is meaningful.

## How it works

There is no autoregressive decoding and no parsing of model text. The model emits one
**logit vector per allowed option**; everything after that is arithmetic.

```text
request (state + questions)
   │
   ▼
encode_record()      state + schema → one token sequence,
   │                 recording the token span of every question and option
   ▼
Qwen backbone        ONE forward pass → hidden states (no sampling, no KV cache)
   │
   ▼
JointSchemaHead      span pooling + evidence routing → one logit per option
   │
   ▼
softmax(-1)          one probability distribution per question
   │
   ▼
systemone_answer()   argmax / expectation / pass-through → answer fields
   │
   ▼
response             {"model", "answers", "usage"}
```

### 1. The prompt is a schema, not a conversation

`encode_record()` renders the whole request into a single token sequence:

1. A system message: *"Read the complete state and schema. Decide every field jointly.
   Each answer must be exactly one of that field's allowed options."*
2. `STATE:` — followed by the state, rendered as compact key-sorted JSON if it is not
   already a string. Images/videos become `<|vision_start|><|image_pad|><|vision_end|>`
   and video placeholders.
3. `SCHEMA FIELDS:` — then one block per question:

   ```text
   FIELD 1
   ID: department
   TYPE: choice
   INSTRUCTION: Which team should handle the message?
   ALLOWED OPTIONS:
   OPTION 1: {"option_id":"billing","description":"Payments, invoices, or refunds"}
   OPTION 2: {"option_id":"technical","description":"Bugs, outages, or blocked orders"}
   END FIELD
   ```

4. An assistant prefill:
   `<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:` — a turn the
   backbone attends to, not a place the model writes into.

While building this, `encode_record()` records the **token span** (start/end offsets)
of each question's instructions and of each option's rendered text. The head reads
hidden states at those spans later — this is how a text prompt is turned into a
classification task without any parsing.

The state is truncated to fit `max_length` (16384). If the schema alone does not fit,
the request is rejected rather than silently truncated, because truncating the schema
would remove options.

### 2. One forward pass over the backbone

`ClefModel.forward()` runs the Qwen3.5 multimodal backbone once with
`use_cache=False`, keeping only the last hidden states. No sampling, no temperature, no
stop tokens, no decoding loop. (If no media is present, the vision tower is skipped.)

### 3. The joint schema head turns spans into per-option logits

`JointSchemaHead.forward()`:

- **Pools** hidden states over each question span → a *question vector*; over each
  option span → an *option context vector*; and over the final token → a *global
  vector*.
- **Adds lexical embeddings** of each option's own text, so an option's wording matters
  independently of where it appears in the state.
- **Routes evidence** — option queries attend over the *whole* sequence (the
  "memory"), so an option can pull evidence from anywhere in the state. This is the
  "joint" part: every field is decided together in one pass with shared evidence.
- **Scores** each option against its field vector with a learned cosine term plus a
  small residual MLP, added to a lexical prior (option text vs. question/global anchor).
- **Emits one logit per option.**

### 4. Logits → probabilities → answers

`systemone()` applies `softmax(-1)` per question (independently per field) and hands the
distribution to `systemone_answer()`, which builds each type's fields:

| Type | Model emits | Derivation in Python |
|---|---|---|
| `noul` | 2 logits: `true`, `false` | pass through `p(true)`; `false` discarded |
| `choice` | one logit per criterion key | `choice` = argmax, `confidence` = max prob, full `probabilities` |
| `score` | one logit per index `0..N-1` | `score` = `sum(i · pᵢ)`, `confidence` = max prob, `legend` built from criteria |

### 5. What the model computes vs. what the server computes

Only the **probability distributions** come from the model. Everything else is Python
post-processing in the release's `joint_schema_model.py`:

| Field | Origin |
|---|---|
| `probabilities` | model output (softmax over logits) |
| `noul` | model output, passed through and rounded to 4 decimals |
| `choice` | derived: `max()` over probabilities |
| `score` | derived: expected value of the distribution |
| `confidence` | derived: the maximum probability |
| `legend`, `type`, `model` | echoed from the request |
| `usage.input_tokens` | length of the encoded token sequence |
| `usage.output_tokens` | **hardcoded `0`** — a constant, not a measurement |

`output_tokens: 0` is not a bug or a missing metric. The model never produces tokens,
so the Jev/OpenAI-style `usage` object reports zero. A more meaningful "output" size
would be the number of questions answered or options scored.

## Step-by-step execution

This is exactly what happens between `make run` and a JSON answer.

### Step 0 — prerequisites

```sh
python3 --version       # 3.12 preferred; uv can fetch it
df -h .                 # need ~25 GB free
ls /dev/kfd 2>/dev/null # present on ROCm hosts
```

### Step 1 — `make run` → `run.sh`

`run.sh` is a bash script that prepares the environment and then hands off to Python:

| Phase | What happens |
|---|---|
| arg parsing | `--smoke`, `--debug`, `--help` |
| uv check | installs uv to `~/.local/bin` if missing |
| ROCm detect | if `/dev/kfd`, `/opt/rocm`, or `ROCM_PATH` exists → native gfx1151 mode; unsets `HSA_OVERRIDE_GFX_VERSION`, sets `HSA_USE_SVM=0`, `HSA_ENABLE_SDMA=0`, `HIP_VISIBLE_DEVICES=0` |
| venv | creates `./.venv` with Python 3.12 if absent |
| torch | installs torch — from the TheRock gfx1151 index on ROCm hosts, otherwise the default wheel |
| deps | `transformers>=5.10.2`, `huggingface_hub`, `safetensors`, `pillow`, `accelerate` |
| weights | if `models/clef-flash/joint_schema_model.py` is missing, `hf download Cloudflare/clef-flash` (~19 GB) |
| GPU probe | on ROCm, runs a tiny CUDA op before loading 19 GB, so a broken GPU fails fast |
| env | exports `CLEF_MODEL_DIR`, `CLEF_DEVICE`, `CLEF_HOST`, `CLEF_PORT`, `CLEF_SMOKE`, `CLEF_DEBUG` |
| exec | `uv run python serve_clef_flash.py` |

### Step 2 — server startup

`serve_clef_flash.py` then:

1. Picks a device (`cuda` → `mps` → `cpu` when `CLEF_DEVICE=auto`).
2. Adds the model directory to `sys.path` and imports `joint_schema_model`.
3. `load_release_model()` loads:
   - the merged Qwen3.5 backbone via `transformers` (`Qwen3_5ForConditionalGeneration`, bfloat16),
   - `joint_head_config.json` + `joint_head.safetensors` → `JointSchemaHead`,
   - `AutoProcessor` (tokenizer + image/video processor),
   and calls `.eval()`.
4. Prints `model ready` and serves `ThreadingHTTPServer` on `HOST:PORT`.

Watch for:

```text
torch 2.10.0+rocm7.13.0a20260513
device Radeon 8060S Graphics
loading ./models/clef-flash on cuda
model ready
serving http://0.0.0.0:8000/v1/systemone
```

### Step 3 — verify it is up

```sh
curl -s http://127.0.0.1:8000/health
```

```json
{"status":"ok","model":"clef-flash","device":"cuda","torch":"...","object":"list","data":[{"id":"clef-flash","object":"model"}]}
```

### Step 4 — send a request

Inside the server, per request:

1. **Route check** — only `POST /v1/decisions` and `/v1/systemone` are accepted;
   `GET /health` and `GET /v1/models` are handled separately. Anything else → 404.
2. **Body read** — `Content-Length` is read and parsed as JSON.
3. **`decide()`** — validates that `state` and `questions` exist, defaults `model`.
   With `DEBUG=1` it also prints the request, the *fully rendered prompt text*, and the
   model latency.
4. **`systemone()`** — validates each question (`type` must be one of the three;
   `criteria` must be non-empty for `choice`/`score`), encodes the record, runs the one
   forward pass, softmaxes per question, and builds answers.
5. **`_send()`** — `json.dumps` the dict and write it with
   `Content-Type: application/json`.

Any exception becomes a `400` with `{"error": "..."}` and a traceback in the server
log. A full run right after startup is dominated by loading, not by this path.

### Step 5 — read the answer

The response is always:

```json
{
  "model": "clef-flash",
  "answers": { "<question id>": { "type": "...", ... } },
  "usage": { "input_tokens": N, "output_tokens": 0 }
}
```

### Step 6 — run all three types

```sh
make test                                   # local, http://127.0.0.1:8000
BASE_URL=http://192.168.1.198:8000 ./test.sh  # or any remote server
```

`test.sh` posts one request per type with the same `STATE`, pretty-prints each
request/response with `jq` (falling back to raw text if `jq` is absent), and prints
`total` and `ttfb` timings. The `Makefile` target only points at `127.0.0.1`; for a
remote host, call `test.sh` directly and pass `BASE_URL`, as above.

Note: `ttfb` (time to first byte) is within a fraction of a millisecond of `total` in
every response — there is no streaming and no decode phase, so the first byte *is* the
whole answer.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health`, `/v1/models` | status, device, torch version, model list |
| `POST` | `/v1/decisions`, `/v1/systemone` | Jev-compatible decision endpoint |

Request body:

| Field | Required | Description |
|---|---|---|
| `model` | no | echoed back; defaults to `"clef-flash"` |
| `state` | **yes** | the situation — a string, or any JSON value (rendered compact, key-sorted) |
| `questions` | **yes** | non-empty object of question ID → question |
| `images` | no | image inputs (Qwen vision placeholders are inserted) |
| `videos` | no | video inputs |

Example:

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

### Response fields

`answers[<id>]` depends on the type:

```jsonc
// noul
{ "type": "noul", "noul": 0.9425 }

// choice
{
  "type": "choice",
  "choice": "technical",
  "confidence": 0.9804,
  "probabilities": { "billing": 0.0134, "technical": 0.9804, "sales": 0.0062 }
}

// score
{
  "type": "score",
  "score": 1.9583,
  "confidence": 0.9752,
  "legend": { "0": "Can wait", "1": "This week", "2": "Today" },
  "probabilities": { "0": 0.017, "1": 0.0078, "2": 0.9752 }
}
```

Probabilities are rounded to 4 decimals and each question's distribution sums to 1
(before rounding). `legend` lets you read a `score` answer without keeping the original
request. `usage.input_tokens` is the encoded prompt length; `output_tokens` is always
`0` (see [How it works](#how-it-works)).

## Worked example

Output of `make test` against `http://192.168.1.198:8000`. The state is identical for
all three requests:

```text
Checkout has been failing for every customer for the last hour. Orders are blocked and
support is getting refund demands.
```

Only the schema differs, which is why the prompt lengths (163 / 180 / 167 tokens)
differ slightly: they track the number of options and the length of their descriptions.

| Type | Prompt | Answer | Latency |
|---|---|---|---|
| `noul` — "Is a service down?" | 163 tok | `0.9425` | 0.270 s |
| `choice` — "Which team should handle the message?" | 180 tok | `technical` @ 0.9804 | 0.263 s |
| `score` — "How soon does this need a response?" | 167 tok | `1.9583` ≈ Today @ 0.9752 | 0.270 s |

Latency varies by a few tens of milliseconds between runs; the numbers below are from
the same run as the verbatim output at the end of this section.

<details>
<summary><b><code>noul</code></b> — 163 tokens, <code>0.9425</code></summary>

**What executes.** `encode_record()` renders one field with two options. For `noul`,
`question_options()` injects the descriptions automatically:

```text
FIELD 1
ID: outage
TYPE: noul
INSTRUCTION: Is a service down?
ALLOWED OPTIONS:
OPTION 1: {"option_id":"true","description":"The proposition is true or the answer is yes."}
OPTION 2: {"option_id":"false","description":"The proposition is false or the answer is no."}
END FIELD
```

With the system message, `STATE:`, and the assistant prefill, that is 163 tokens. One
forward pass emits two option logits; `softmax` yields `{true: 0.9425, false: 0.0575}`.
`systemone_answer()` passes `p(true)` through, so `p(false)` is implied but never
returned. There is no argmax and no expected value here — the model's number is the
answer.

Read it as: ~94% likely a service is down.

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
<summary><b><code>choice</code></b> — 180 tokens, <code>technical</code></summary>

**What executes.** `question_options()` **sorts** criteria keys, so options are rendered
`billing`, `sales`, `technical` (not the order in the request). Prompt and state total
180 tokens. Three logits → softmax → `{billing: 0.0134, sales: 0.0062, technical:
0.9804}`. Then, in Python:

```python
choice = max(options, key=probabilities.__getitem__)   # "technical"
confidence = probabilities[choice]                     # 0.9804
```

`confidence` is simply the winning probability — not a separate model output. The
displayed probabilities sum to 1.0 (`0.0134 + 0.0062 + 0.9804`).

Read it as: route to `technical`, ~98% sure; `billing` is a ~1% long shot.

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
<summary><b><code>score</code></b> — 167 tokens, <code>1.9583</code></summary>

**What executes.** The three criteria become indexed levels `0`, `1`, `2`, rendered in
array order (167 tokens). Three logits → softmax → `{0: 0.017, 1: 0.0078, 2: 0.9752}`.
Then:

```python
score = sum(index * probabilities[level] for index, level in enumerate(levels))
#      0*0.0170 + 1*0.0078 + 2*0.9752 = 1.9582 from the rounded values shown;
#      the server computes this at full precision and reports 1.9583
confidence = max(probabilities[level] for level in levels)   # 0.9752
legend = dict(zip(levels, question["criteria"]))             # {"0": "Can wait", ...}
```

This is the clearest demonstration that the headline number is post-processed: `1.9583`
is the distribution's mean, not something the model chose. It is below `2.0` because
the small mass on `0` and `1` pulls it down. `legend` is copied from the request, so
the response is readable on its own.

Read it as: essentially "Today" (~98% of the mass on the last level). Prefer
`confidence` for hard decisions: a flat distribution over `0` and `2` would give a
score near `1.0` with low confidence, which is very different from a confident `1.0`.

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

### Full `test.sh` output

Verbatim output of `BASE_URL=http://192.168.1.198:8000 ./test.sh`, in order. Each
block is the echoed request, the response, and one `[latency]` line where `total` and
`ttfb` are within a fraction of a millisecond of each other — the whole answer arrives
in the first byte because there is no decode phase.

<details open>
<summary><b>All three cases</b></summary>

```text
===== noul =====
--- request ---
{
  "model": "clef-flash",
  "state": "Checkout has been failing for every customer for the last hour. Orders are blocked and support is getting refund demands.",
  "questions": {
    "outage": {
      "type": "noul",
      "instructions": "Is a service down?"
    }
  }
}
--- response ---
{
  "model": "clef-flash",
  "answers": {
    "outage": {
      "type": "noul",
      "noul": 0.9425
    }
  },
  "usage": {
    "input_tokens": 163,
    "output_tokens": 0
  }
}
[latency] total=0.270046s ttfb=0.269843s
===== choice =====
--- request ---
{
  "model": "clef-flash",
  "state": "Checkout has been failing for every customer for the last hour. Orders are blocked and support is getting refund demands.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle the message?",
      "criteria": {
        "billing": "Payments, invoices, or refunds",
        "technical": "Bugs, outages, or blocked orders",
        "sales": "New purchases or upgrades"
      }
    }
  }
}
--- response ---
{
  "model": "clef-flash",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "technical",
      "confidence": 0.9804,
      "probabilities": {
        "billing": 0.0134,
        "technical": 0.9804,
        "sales": 0.0062
      }
    }
  },
  "usage": {
    "input_tokens": 180,
    "output_tokens": 0
  }
}
[latency] total=0.262985s ttfb=0.262697s
===== score =====
--- request ---
{
  "model": "clef-flash",
  "state": "Checkout has been failing for every customer for the last hour. Orders are blocked and support is getting refund demands.",
  "questions": {
    "urgency": {
      "type": "score",
      "instructions": "How soon does this need a response?",
      "criteria": [
        "Can wait",
        "This week",
        "Today"
      ]
    }
  }
}
--- response ---
{
  "model": "clef-flash",
  "answers": {
    "urgency": {
      "type": "score",
      "score": 1.9583,
      "confidence": 0.9752,
      "legend": {
        "0": "Can wait",
        "1": "This week",
        "2": "Today"
      },
      "probabilities": {
        "0": 0.017,
        "1": 0.0078,
        "2": 0.9752
      }
    }
  },
  "usage": {
    "input_tokens": 167,
    "output_tokens": 0
  }
}
[latency] total=0.269899s ttfb=0.269709s
```

</details>

## Configuration

### Make targets

| Target | Description |
|---|---|
| `make run` | Start the server (override with `make run PORT=9000`) |
| `make smoke` | Load the model and run one in-process request, no server |
| `make debug` | Start the server with request / rendered-prompt / latency logging |
| `make test` | One request per question type against `127.0.0.1:$(PORT)` |
| `make help` | List targets |

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `CLEF_DEVICE` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `MODEL_DIR` | `./models/clef-flash` | Local weights directory |
| `DEBUG` | `0` | `1` to log requests, rendered prompts, and latency |
| `BASE_URL` | `http://127.0.0.1:8000` | Target for `test.sh` |
| `STATE` | built-in incident text | State used by `test.sh` |
| `MODEL_ID` | `Cloudflare/clef-flash` | HF repo to download |
| `PYTHON_VERSION` | `3.12` | venv Python |
| `TORCH_INDEX_URL` | TheRock gfx1151 index | torch wheel source on ROCm |

The server reads the `CLEF_*` variables, so it can also be started directly:

```sh
CLEF_MODEL_DIR=./models/clef-flash CLEF_PORT=9000 .venv/bin/python serve_clef_flash.py
```

...but `run.sh` is the supported path, since it also prepares the venv and weights.

### Useful invocations

```sh
make run PORT=9000                          # different port
CLEF_DEVICE=cpu make run                    # force CPU
make smoke                                  # load + one request, no HTTP
make debug                                  # show the rendered prompt and latency
BASE_URL=http://192.168.1.198:8000 ./test.sh  # test a remote server
STATE="Database replica lag is growing." BASE_URL=... ./test.sh
```

## ROCm (Strix Halo / gfx1151)

On Radeon 8060S-class GPUs (Ryzen AI Max, GMKtec EVO-X2), the pytorch.org ROCm wheel
segfaults or raises `hipErrorNoBinaryForGpu`. `run.sh` installs TheRock gfx1151 wheels
from `https://rocm.nightlies.amd.com/v2/gfx1151/` and leaves
`HSA_OVERRIDE_GFX_VERSION` unset.

Verified on: torch `2.10.0+rocm7.13.0a20260513`, device `"Radeon 8060S Graphics"`.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `connection refused` from `test.sh` | server not up yet — wait for `model ready`, or check `/health` |
| `make test` fails but `./test.sh` works | `make test` targets `127.0.0.1`; use `./test.sh` with `BASE_URL` for remote hosts |
| request/response not pretty-printed | `jq` not installed (falls back to raw output) |
| first request very slow | weights are still loading; wait for `serving http://...` |
| `hipErrorNoBinaryForGpu` / segfault at startup | wrong torch wheel — use `run.sh` on a ROCm host so the gfx1151 index is used |
| `schema requires N tokens ... maximum is 16384` | the questions/schema alone exceed the context; shorten descriptions or ask fewer questions |
| `criteria must not be empty` | `choice`/`score` need non-empty `criteria` |
| answers look wrong / want to inspect the prompt | run with `DEBUG=1` (`make debug`) to print the rendered prompt |

## Files

| File | Purpose |
|---|---|
| `run.sh` | Environment setup, weights download, server start |
| `serve_clef_flash.py` | HTTP server wrapping the model's `systemone()` entry point |
| `test.sh` | One request per question type against a running server |
| `Makefile` | Convenience targets |
| `models/clef-flash/` | Downloaded snapshot (backbone, joint head, processor) — created on first run |
| `.venv/` | uv virtualenv — created on first run |

The model itself (`joint_schema_model.py`: `encode_record`, `JointSchemaHead`,
`systemone`) ships with the [HF release](https://huggingface.co/Cloudflare/clef-flash)
and is imported from `MODEL_DIR` at startup — this repo only serves it.
