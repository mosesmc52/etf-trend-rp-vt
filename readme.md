# ETF Trend-RP-VT Strategy

This project implements a **trend-following portfolio rotation strategy** with:

- **Trend filter:** invest only in sleeves (e.g., QQQ, BND, GLD) that are above their moving averages.  
- **Risk parity sizing:** allocate among active sleeves using inverse volatility weights.  
- **EMA smoothing:** smooth sleeve weights to avoid whipsaw reallocations.  
- **Volatility targeting:** scale the sleeves block to target a desired annualized portfolio volatility.  
- **Cash sleeve:** allocate the unused portion (or full allocation when no sleeves are active) into a defensive cash ETF (e.g., BIL).  
- **Scheduled rebalances:** run on **month-end** or **week-end**, with optional `force_rebalance` to sync anytime.  

Data comes from the [Alpaca API](https://alpaca.markets/) (IEX feed). Orders can be simulated or submitted live using Alpaca’s trading endpoint.

---

## How it works

1. **Fetch data**  
   Daily OHLCV bars for `SLEEVES + CASH` are pulled from Alpaca via `price_history()`.

2. **Generate signals**  
For each sleeve, a binary signal is 1 if  
(window defined in `MA_FIXED` per sleeve).

3. **Select sleeves**  
Only sleeves with signal = 1 are eligible.  
- If none are active → 100% in `CASH`.

4. **Inverse-volatility weights**  
Among active sleeves, assign weights inversely proportional to trailing volatility (default 60-day lookback).

5. **EMA smoothing**  
Blend today’s raw weights with previous weights using an exponential moving average (`EMA_ALPHA`).

6. **Volatility targeting**  
Estimate sleeves’ covariance over a lookback (default 60 days), then scale weights so the portfolio annualized volatility ≈ target (e.g., 10%).  
- Gross exposure is capped at `GROSS_CAP`.  
- If margin is not allowed, capped at 1.0.  

7. **Residual → cash**  
Any leftover allocation goes into `CASH`.

8. **Rebalance schedule**  
- `REB="M"` → last trading day of each month.  
- `REB="W"` → last trading day of each ISO calendar week.  
- `force_rebalance=True` overrides schedule.  

9. **Orders**  
Convert target weights → target quantities using latest prices, then call `process_position()` to generate buy/sell/hold actions.  
- In backtest mode, actions are just returned.  
- In live mode (`is_live_trade=True`), Alpaca orders are submitted.

---

## Example Run

```python
from alpaca_trade_api.rest import REST
from helpers import run_single_iteration, print_orders_table

# Alpaca client
api = REST(KEY_ID, SECRET, BASE_URL, api_version="v2")

# Strategy config
SLEEVES = ["QQQ", "BND", "GLD"]
CASH    = "BIL"
MA_FIXED = {"QQQ": 150, "BND": 150, "GLD": 150}

result = run_single_iteration(
api,
sleeves=SLEEVES,
cash=CASH,
ma_fixed=MA_FIXED,
force_rebalance=False,   # True to rebalance immediately
is_live_trade=False,     # True to place Alpaca market orders
)

print_orders_table(result)
Rebalance date: 2025-08-29 | sleeves_on=['QQQ','GLD'] | gross_sleeves=0.75 | cash=0.250
symbol   action   target_qty   diff     price     alloc_w
QQQ      buy             12      12    420.50    0.5000
GLD      buy             30      30    180.20    0.2500
BND      hold             0       0    100.15    0.0000
BIL      hold             0       0     91.70    0.2500


## Quickstart with Docker

This repo comes with a `Makefile` and `docker-compose.yml` to make builds and runs reproducible.

### 1. Build the image
```bash
make build
make up        # run interactive (logs to console)
make upd       # run in daemon mode
make logs      # view container logs
make shell     # launch into container shell
make stop      # stop services
make restart   # restart services
make down      # stop + remove containers
make clean     # remove containers + image
```
