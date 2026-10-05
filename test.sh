#!/usr/bin/env bash
# One System One request per Clef-flash question type: noul, choice, score.
# Server must already be running (./run.sh).
#
#   ./test.sh
#   BASE_URL=http://127.0.0.1:8000 ./test.sh
#   STATE="Database replica lag is growing." BASE_URL=... ./test.sh
#
# Pretty-prints every request and response, coloured on a terminal. JSON is rendered
# with jq when available, otherwise python3, otherwise raw. Set NO_COLOR=1 to disable
# colour and FORCE_COLOR=1 to keep it when piping (useful for a screenshot or a pager).

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
STATE="${STATE:-Checkout has been failing for every customer for the last hour. Orders are blocked and support is getting refund demands.}"
TIMEOUT="${TIMEOUT:-120}"
WIDTH="${WIDTH:-70}"

# ---------------------------------------------------------------- colours ----
# Colour when attached to a terminal (or when FORCE_COLOR is set); NO_COLOR wins.
COLOR=0
if [[ -n "${FORCE_COLOR:-}" ]] || { [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; }; then
  COLOR=1
fi

if (( COLOR )); then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
  BLUE=$'\033[34m'; GREEN=$'\033[32m'
  YELLOW=$'\033[33m'; RED=$'\033[31m'
  JQ_MODE=(-C)
else
  BOLD=""; DIM=""; RESET=""; BLUE=""; GREEN=""; YELLOW=""; RED=""
  JQ_MODE=(-M)
fi

rule() { printf '%s\n' "${DIM}$(printf '─%.0s' $(seq 1 "$WIDTH"))${RESET}"; }
title() {
  printf '\n%s\n' "${BOLD}${BLUE}▎ ${1}${RESET}"
  rule
}
label() { printf '%s\n' "${YELLOW}${BOLD}· ${1}${RESET}"; }
indent() { sed 's/^/    /'; }

# ------------------------------------------------------------ json pretty ----
# Colourised JSON rendering, used when jq is unavailable. No single quotes in here
# (the whole script is passed via single-quoted python3 -c).
PY_PRETTY='
import json, sys

COLOR = sys.argv[1] == "1"
KEY = "\033[34;1m"
STR = "\033[32m"
NUM = "\033[36m"
LIT = "\033[35m"
PUNCT = "\033[2m"
RESET = "\033[0m"


def paint(code, text):
    return code + text + RESET if COLOR else text


def dump(obj, level=0):
    pad = "  " * level
    inner = "  " * (level + 1)
    if isinstance(obj, dict):
        if not obj:
            return paint(PUNCT, "{}")
        parts = [inner + paint(KEY, json.dumps(k)) + paint(PUNCT, ": ") + dump(v, level + 1) for k, v in obj.items()]
        return paint(PUNCT, "{") + "\n" + paint(PUNCT, ",\n").join(parts) + "\n" + pad + paint(PUNCT, "}")
    if isinstance(obj, list):
        if not obj:
            return paint(PUNCT, "[]")
        parts = [inner + dump(v, level + 1) for v in obj]
        return paint(PUNCT, "[") + "\n" + paint(PUNCT, ",\n").join(parts) + "\n" + pad + paint(PUNCT, "]")
    if isinstance(obj, bool):
        return paint(LIT, "true" if obj else "false")
    if obj is None:
        return paint(LIT, "null")
    if isinstance(obj, str):
        return paint(STR, json.dumps(obj))
    return paint(NUM, json.dumps(obj))


print(dump(json.load(sys.stdin)))
'

# Prints JSON indented and (when enabled) coloured, or the input unchanged when
# nothing can parse it. jq is forced to -C/-M because its stdout is a pipe here.
pretty() {
  local input="$1"
  if command -v jq >/dev/null 2>&1 && printf '%s\n' "$input" | jq "${JQ_MODE[@]}" . 2>/dev/null; then
    return 0
  fi
  if command -v python3 >/dev/null 2>&1 && printf '%s' "$input" | python3 -c "$PY_PRETTY" "$COLOR" 2>/dev/null; then
    return 0
  fi
  printf '%s\n' "$input"
}

# --------------------------------------------------------- summary fields ----
# Prints "<tokens>\t<answer>" for the first answer in a response, or "?\t?".
describe() {
  local response="$1"
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$response" | jq -r '
      (.answers | to_entries[0].value) as $a
      | (.usage.input_tokens // "?")                       as $t
      | (if $a.type == "noul"   then ($a.noul    | tostring)
         elif $a.type == "choice" then "\($a.choice) (\($a.confidence))"
         else "\($a.score) (\($a.confidence))"
         end)                                               as $ans
      | "\($t)\t\($ans)"' 2>/dev/null && return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    printf '%s' "$response" | python3 -c '
import json, sys

try:
    d = json.load(sys.stdin)
    a = next(iter(d["answers"].values()))
    t = d.get("usage", {}).get("input_tokens", "?")
    if a["type"] == "noul":
        ans = str(a["noul"])
    elif a["type"] == "choice":
        ans = "%s (%s)" % (a["choice"], a["confidence"])
    else:
        ans = "%s (%s)" % (a["score"], a["confidence"])
    print("%s\t%s" % (t, ans))
except Exception:
    print("?\t?")
' 2>/dev/null && return 0
  fi
  printf '?\t?\n'
}

# ------------------------------------------------------------- one request ----
ROWS=()
post() {
  local name="$1" question="$2" body="$3"
  title "${name} — ${question}"

  label "request"
  pretty "$body" | indent

  local out timing rc=0
  out="$(mktemp)"
  timing="$(curl -sS -m "${TIMEOUT}" "${BASE_URL}/v1/decisions" \
      -H 'Content-Type: application/json' \
      -d "$body" \
      -o "$out" \
      -w '%{time_total} %{time_starttransfer} %{http_code}')" || rc=$?

  if (( rc != 0 )); then
    printf '%s\n' "${RED}request failed (curl exit ${rc})${RESET}" | indent
    rm -f "$out"
    ROWS+=("${name}|-|-|-")
    return 1
  fi

  local response
  response="$(cat "$out")"
  rm -f "$out"

  label "response"
  pretty "$response" | indent

  local total ttfb code
  read -r total ttfb code <<<"$timing"
  printf '%s\n' "${GREEN}latency${RESET}  total $(printf '%.3f' "$total")s  ttfb $(printf '%.3f' "$ttfb")s  http ${code}" | indent

  local fields tokens answer
  fields="$(describe "$response")"
  IFS=$'\t' read -r tokens answer <<<"$fields"
  ROWS+=("${name}|${tokens}|${answer}|$(printf '%.3f' "$total")s")
}

# ------------------------------------------------------------------ header ----
printf '%s\n' "${BOLD}clef-flash test${RESET}  ${DIM}${BASE_URL}${RESET}"
printf '%s\n' "${DIM}state: ${STATE}${RESET}"

post noul "Is a service down?" "$(cat <<EOF
{
  "model": "clef-flash",
  "state": "${STATE}",
  "questions": {
    "outage": {
      "type": "noul",
      "instructions": "Is a service down?"
    }
  }
}
EOF
)"

post choice "Which team should handle the message?" "$(cat <<EOF
{
  "model": "clef-flash",
  "state": "${STATE}",
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
EOF
)"

post score "How soon does this need a response?" "$(cat <<EOF
{
  "model": "clef-flash",
  "state": "${STATE}",
  "questions": {
    "urgency": {
      "type": "score",
      "instructions": "How soon does this need a response?",
      "criteria": ["Can wait", "This week", "Today"]
    }
  }
}
EOF
)"

# ----------------------------------------------------------------- summary ----
title "summary"
printf '  %-8s %-10s %-26s %s\n' "TYPE" "PROMPT" "ANSWER" "LATENCY"
for row in "${ROWS[@]}"; do
  IFS='|' read -r name tokens answer latency <<<"$row"
  printf '  %-8s %-10s %-26s %s\n' "$name" "${tokens} tok" "$answer" "$latency"
done
echo
