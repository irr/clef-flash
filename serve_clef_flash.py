"""Local System One server for Cloudflare/clef-flash.

Started by run.sh via `uv run`. Loads the snapshot in CLEF_MODEL_DIR and
exposes POST /v1/systemone, the Jev-compatible body from the model card.

On Strix Halo the process must be started with HSA_OVERRIDE_GFX_VERSION unset
and a gfx1151 torch wheel (torch 2.10.0+rocm7.13 verified on a Radeon 8060S).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

MODEL_DIR = os.environ["CLEF_MODEL_DIR"]
HOST = os.environ.get("CLEF_HOST", "127.0.0.1")
PORT = int(os.environ.get("CLEF_PORT", "8091"))
SMOKE = os.environ.get("CLEF_SMOKE", "0") == "1"
DEBUG = os.environ.get("CLEF_DEBUG", "0") == "1"


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device(os.environ.get("CLEF_DEVICE", "auto"))

sys.path.insert(0, MODEL_DIR)
from joint_schema_model import encode_record, load_release_model, systemone  # noqa: E402

print(f"torch {torch.__version__}", flush=True)
if DEVICE == "cuda":
    print(f"device {torch.cuda.get_device_name(0)}", flush=True)
print(f"loading {MODEL_DIR} on {DEVICE}", flush=True)
MODEL, PROCESSOR = load_release_model(MODEL_DIR, device=DEVICE)
MODEL.eval()
print("model ready", flush=True)


def decide(body: dict) -> dict:
    if "questions" not in body or "state" not in body:
        raise ValueError("request needs state and questions")
    body.setdefault("model", "clef-flash")
    if not DEBUG:
        return systemone(MODEL, PROCESSOR, body)

    # The model is not generative: the "prompt" is the token sequence built by
    # encode_record, and the "response" is the per-option probabilities.
    print("----- DEBUG request -----", flush=True)
    print(json.dumps(body, indent=2, ensure_ascii=False), flush=True)
    try:
        encoded = encode_record(PROCESSOR.tokenizer, body, processor=PROCESSOR)
        print(f"----- DEBUG prompt ({len(encoded.input_ids)} tokens) -----", flush=True)
        print(PROCESSOR.tokenizer.decode(list(encoded.input_ids)), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"(could not render prompt: {exc})", flush=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        response = systemone(MODEL, PROCESSOR, body)
    finally:
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000
        print(f"----- DEBUG latency: {latency_ms:.1f} ms -----", flush=True)
    print("----- DEBUG response -----", flush=True)
    print(json.dumps(response, indent=2, ensure_ascii=False), flush=True)
    return response


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def _send(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/health", "/v1/models"):
            self._send(
                200,
                {
                    "status": "ok",
                    "model": "clef-flash",
                    "device": DEVICE,
                    "torch": torch.__version__,
                    "object": "list",
                    "data": [{"id": "clef-flash", "object": "model"}],
                },
            )
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path not in ("/v1/systemone", "/v1/decisions"):
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            self._send(200, decide(body))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send(400, {"error": str(exc)})


def smoke() -> None:
    response = decide(
        {
            "model": "clef-flash",
            "state": "Checkout has been failing for every customer for the last hour.",
            "questions": {
                "urgent": {"type": "noul", "instructions": "Is this urgent?"},
                "team": {
                    "type": "choice",
                    "instructions": "Which team should handle this?",
                    "criteria": {
                        "billing": "Payments or invoices",
                        "technical": "Outages and errors",
                    },
                },
            },
        }
    )
    print(json.dumps(response, indent=2), flush=True)


if __name__ == "__main__":
    if SMOKE:
        smoke()
        raise SystemExit(0)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"serving http://{HOST}:{PORT}/v1/systemone", flush=True)
    server.serve_forever()
