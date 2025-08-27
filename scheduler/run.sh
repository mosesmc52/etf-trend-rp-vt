#!/bin/bash
echo "[$(date)] Running ETF Trend Algo..."
cd /app/
poetry run python algo.py
