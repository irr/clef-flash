#!/usr/bin/env bash
# One System One request per Clef-flash question type: noul, choice, score.
# Server must already be running (./run.sh).
#
#   ./test.sh
#   BASE_URL=http://127.0.0.1:8000 ./test.sh

set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
STATE="${STATE:-Checkout has been failing for every customer for the last hour. Orders are blocked and support is getting refund demands.}"

post() {
  local name="$1"
  local body="$2"
  echo "===== ${name} ====="
  curl -sS -m 120 "${BASE_URL}/v1/systemone" \
    -H 'Content-Type: application/json' \
    -d "$body" \
    -w '\n[latency] total=%{time_total}s ttfb=%{time_starttransfer}s\n'
}

post noul "$(cat <<EOF
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

post choice "$(cat <<EOF
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

post score "$(cat <<EOF
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
