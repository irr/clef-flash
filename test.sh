#!/usr/bin/env bash
# One System One request per Clef-flash question type: noul, choice, score.
# Server must already be running (./run.sh).
#
#   ./test.sh
#   BASE_URL=http://127.0.0.1:8000 ./test.sh
#   STATE="Database replica lag is growing." BASE_URL=... ./test.sh
#
# Pretty-prints every request and response. Uses jq when available, falls back to
# python3 -m json.tool, and finally to the raw text. Colours are emitted only when
# stdout is a terminal, so redirected output stays clean.

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
STATE="${STATE:-Checkout has been failing for every customer for the last hour. Orders are blocked and support is getting refund demands.}"
TIMEOUT="${TIMEOUT:-120}"
WIDTH="${WIDTH:-70}"

# ---------------------------------------------------------------- colours ----
if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
  BLUE=$'\033[34m'; GREEN=$'\033[32m'
  YELLOW=$'\033[33m'; RED=$'\033[31m'
else
  BOLD=""; DIM=""; RESET=""; BLUE=""; GREEN=""; YELLOW=""; RED=""
fi

rule() { printf '%s\n' "${DIM}$(printf '─%.0s' $(seq 1 "$WIDTH"))${RESET}"; }
title() {
  printf '\n%s\n' "${BOLD}${BLUE}▎ ${1}${RESET}"
  rule
}
label() { printf '%s\n' "${YELLOW}${BOLD}· ${1}${RESET}"; }
indent() { sed 's/^/    /'; }

# ------------------------------------------------------------ json pretty ----
# Prints JSON indented, or the input unchanged when nothing can parse it.
pretty() {
  local input="$1"
  if command -v jq >/dev/null 2>&1 && printf '%s\n' "$input" | jq . 2>/dev/null; then
    return 0
  fi
  if command -v python3 >/dev/null 2>&1 && printf '%s\n' "$input" | python3 -m json.tool 2>/dev/null; then
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
