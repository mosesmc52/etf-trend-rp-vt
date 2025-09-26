#!/bin/bash
echo "[$(date)] Running ETF Trend Algo..."

set -euo pipefail

set -a
[ -f /app/.env ] && . /app/.env
set +a

# sanity checks (don’t print secrets)
: "${ALPACA_KEY_ID:?ALPACA_KEY_ID not set}"
: "${ALPACA_SECRET_KEY:?ALPACA_SECRET_KEY not set}"

cd /app
poetry run python algo.py
