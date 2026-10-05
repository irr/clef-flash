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

## How it works

Despite the OpenAI-style envelope, nothing is generated. There is no autoregressive
decoding and no parsing of model text: the model emits one **logit vector per allowed
option**, and the server formats those numbers into JSON. End-to-end latency is one
prefill — that is where the speed comes from.

```text
request (state + questions)
   │
   ▼
encode_record()      state + schema → one token sequence,
   │                 recording the token span of every question and option
   ▼
Qwen backbone        ONE forward pass → hidden states (no sampling)
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

`encode_record()` renders the request into a single token sequence:

1. A system message: “Read the complete state and schema. Decide every field jointly.
   Each answer must be exactly one of that field's allowed options.”
2. `STATE:` followed by the state rendered as compact, key-sorted JSON (media become
   `<|vision_start|><|image_pad|><|vision_end|>` / video placeholders).
3. `SCHEMA FIELDS:`, then one block per question:

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

4. An assistant prefill: `<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:`
   — a turn the backbone attends to, not a place to write.

While building it, `encode_record()` records the token span of each question's
instructions and of each option's rendered text. Those spans are what the head reads
later. The state is truncated to fit `max_length` (16384); if the schema alone does not
fit, the request is rejected.

### 2. One forward pass over the backbone

`ClefModel.forward()` runs the Qwen multimodal backbone once and keeps only the hidden
states. The KV cache, temperature, stop tokens, and decoding loops never enter the
picture.

### 3. The joint schema head turns spans into per-option logits

`JointSchemaHead.forward()`:

- pools hidden states over each **question span** → question vector, over each **option
  span** → option context vector, and over the final token → a global vector;
- adds a lexical embedding of the option's own text, so the option's wording matters
  independently of context;
- routes option queries through attention layers over the *whole* sequence (the
  "memory"), letting an option collect evidence from anywhere in the state — this is
  the **joint** part: every field is decided together in one pass, sharing evidence;
- scores each option against its field vector with a cosine term plus a small residual
  MLP, added to a lexical prior;
- emits one logit per option.

### 4. Logits to probabilities to answers

`systemone()` applies `softmax(-1)` per question and hands the distribution to
`systemone_answer()`, which derives each type's fields:

| Type | Model emits | Derivation |
|---|---|---|
| `noul` | logits for `true` / `false` (criteria injected automatically) | passes through `p(true)`; `false` is dropped |
| `choice` | one logit per criterion key | `choice` = argmax, `confidence` = max prob, full `probabilities` |
| `score` | one logit per index `0..N-1` | `score` = expected value `sum(i·pᵢ)`, `confidence` = max prob, `legend` built from `criteria` |

### 5. What the model computes vs. what the server computes

Only the probability distributions come from the model's forward pass. Everything else
is Python post-processing in the model release's `joint_schema_model.py`:

- `choice` is a `max()` over the probabilities; `score` is an expected value (a
  fractional number the model never "chose");
- `confidence` is the maximum probability; `legend`, `type`, and `model` are echoed
  from the request;
- `usage.input_tokens` is the length of the encoded sequence, while
  `usage.output_tokens` is a hardcoded `0` — a constant, not a measurement, because no
  tokens are ever produced.

Every question in a request — any mix of types — is answered in the same single forward
pass.

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

| Question type | Prompt | Answer | Latency |
|---|---|---|---|
| `noul` — "Is a service down?" | 163 tokens | `noul: 0.9425` | 0.274s |
| `choice` — "Which team should handle the message?" | 180 tokens | `technical` (confidence 0.9804; billing 0.0134, sales 0.0062) | 0.264s |
| `score` — "How soon does this need a response?" (Can wait / This week / Today) | 167 tokens | `1.9583` ≈ Today (confidence 0.9752; 0.0170 / 0.0078 / 0.9752) | 0.273s |

Each request is a separate process-level round trip, but inside the server every one of
them is a single prefill: `ttfb` is within ~0.2 ms of `total` in all three cases, so
there is no decoding phase to wait for. The differences in latency are just prompt
length (163 / 180 / 167 tokens) and GPU scheduling noise.

In all three cases the state is identical, so the difference in prompt length is only
the rendered schema: the number of options and the length of their descriptions.

Raw responses:

<details>
<summary><code>noul</code></summary>

**What executes.** `encode_record()` renders the schema as one field with two options
(`question_options()` injects `true`/`false` descriptions automatically for `noul`):

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

Prefixed with `STATE:`, the system message, and the assistant prefill, this is 163
tokens. One forward pass gives two option logits, `softmax` turns them into
`{true: 0.9425, false: 0.0575}`, and `systemone_answer()` passes `p(true)` through —
this is the only type with no argmax and no expectation, and the only one that discards
half the distribution: `false` is implied as `1 − 0.9425 = 0.0575` but never returned.

Read it as: "with ~94% probability, a service is down".

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

**What executes.** `question_options()` sorts the criteria keys, so the three options
are rendered `billing`, `sales`, `technical`. Prefix, state, and schema come to 180
tokens. The head emits three logits; `softmax` gives
`{billing: 0.0134, sales: 0.0062, technical: 0.9804}`. Then, in Python:

```python
choice = max(options, key=probabilities.__getitem__)   # "technical"
confidence = probabilities[choice]                     # 0.9804
```

So `choice` and `confidence` are both derived from the same distribution — `confidence`
is just the winning probability, not a separate model output. The probabilities sum to
1.0 (`0.0134 + 0.0062 + 0.9804`) and are rounded to 4 decimals for display.

Read it as: "route this to the technical team, ~98% sure; billing is a ~1% long shot".

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

**What executes.** The three criteria become indexed levels `0`, `1`, `2`; the schema
renders them in order, giving 167 tokens. The head emits three logits and `softmax`
gives `{0: 0.017, 1: 0.0078, 2: 0.9752}`. The answer is then computed in Python:

```python
score = sum(index * probabilities[level] for index, level in enumerate(levels))
#      0*0.0170 + 1*0.0078 + 2*0.9752 = 1.9582 from the rounded values above;
#      the server computes it at full precision and reports 1.9583
confidence = max(probabilities[level] for level in levels)   # 0.9752
legend = dict(zip(levels, question["criteria"]))  # {"0": "Can wait", ...}
```

This is the clearest illustration that the headline number is post-processed: `1.9583`
is a weighted average, not something the model chose. It is not exactly `2.0` because
the small mass on `0` and `1` pulls the mean down. The `legend` is copied verbatim from
the request's `criteria`, which is why the response can be read without the original
request.

Read it as: "today, essentially — ~98% of the mass on the last level" — but prefer
`confidence` over `score` when you need a hard cut, since a flat distribution across
`0` and `2` yields `1.0` with low confidence.

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
