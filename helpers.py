# === single_iteration_trader.py ===
import json
import math
import os
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
from alpaca_trade_api.rest import APIError, TimeFrame
from log import log

# ---------- Config (match your backtest) ----------

VOL_LKBK = 60
EMA_ALPHA = 0.40  # None to disable
VT_TARGET = 0.10  # None to disable
VT_LKBK = 60
GROSS_CAP = 1.25
REB = "M"  # "M" = month-end rebalance


# Live-trading knobs
STATE_PATH = "trend_state.json"  # persists EMA prev weights + last_reb_date
ROUND_LOT = 1  # share granularity
ALLOW_MARGIN = False  # if True, allows gross > 1 up to GROSS_CAP

# Alpaca side constants used by your process_position()
BUY, SELL = "buy", "sell"


def str2bool(value):
    valid = {
        "true": True,
        "t": True,
        "1": True,
        "on": True,
        "false": False,
        "f": False,
        "0": False,
    }

    if isinstance(value, bool):
        return value

    lower_value = value.lower()
    if lower_value in valid:
        return valid[lower_value]
    else:
        raise ValueError('invalid literal for boolean: "%s"' % value)


def process_position(api, security, qty, is_live_trade=False):
    is_existing_position = False
    try:
        position = api.get_position(security)
        current_qty = int(position.qty)
        is_existing_position = True
    except tradeapi.rest.APIError:
        current_qty = 0

    diff = qty - current_qty

    if is_live_trade:
        if is_existing_position:
            if diff > 0:
                api.submit_order(
                    symbol=security,
                    time_in_force="day",
                    side=BUY,
                    type="market",
                    qty=diff,
                )
            elif diff < 0:
                api.submit_order(
                    symbol=security,
                    time_in_force="day",
                    side=SELL,
                    type="market",
                    qty=abs(diff),
                )
        else:
            if qty > 0:
                api.submit_order(
                    symbol=security,
                    time_in_force="day",
                    side=BUY,
                    type="market",
                    qty=qty,
                )

    if is_existing_position:
        action = BUY if diff > 0 else SELL
    else:
        action = BUY

    return action, qty, diff


# ---------- Small utils ----------


def price_history(
    api,
    ticker: str,
    start_date: datetime.date,
    end_date: datetime.date,
    feed: str = "iex",
):
    """
    Fetch daily bars using Alpaca Data API.
    Tries IEX (free) by default; auto-retries with IEX if SIP is denied.
    """
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    try:
        return api.get_bars(
            ticker,
            TimeFrame.Day,
            start_str,
            end_str,
            adjustment="all",
            feed=feed,
        )
    except APIError as e:
        msg = str(e).lower()
        if "sip" in msg and "subscription" in msg:
            return api.get_bars(
                ticker,
                TimeFrame.Day,
                start_str,
                end_str,
                adjustment="all",
                feed="iex",
            )
        raise


# =================================
# Small utils (state & scheduling)
# =================================
def _load_state(path: str = STATE_PATH) -> Dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"ema_prev": {}, "last_reb_date": None}


def _save_state(
    ema_prev: Dict[str, float], last_reb_date, path: str = STATE_PATH
) -> None:
    with open(path, "w") as f:
        json.dump({"ema_prev": ema_prev, "last_reb_date": str(last_reb_date)}, f)


def _month_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # last trading day each month
    df = pd.Series(1, index=idx)
    return df.groupby([idx.year, idx.month]).tail(1).index


def _is_rebalance_day(
    idx: pd.DatetimeIndex, today: pd.Timestamp, reb_rule: str
) -> bool:
    if reb_rule.upper().startswith("M"):
        return today.normalize() in _month_end_index(idx)
    return True  # default: always rebalance


# ==========================================
# Bars → pandas Series (robust converter)
# ==========================================
def _bars_to_series_close(bars) -> pd.Series:
    """
    Convert Alpaca bars (DataFrame with timestamp + close) to a daily Series of closes.
    - Drops intraday time, keeps DATE only
    - If multiple rows per day exist, keeps the LAST close
    """

    # If the object has a .df property (common with Alpaca SDK)
    if hasattr(bars, "df"):
        df = bars.df.copy()
    else:
        # If it's already a DataFrame
        try:
            df = pd.DataFrame(bars)
        except Exception:
            return pd.Series(dtype=float)

    if df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df.index = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        elif "t" in df.columns:
            df.index = pd.to_datetime(df["t"], utc=True, errors="coerce")
        else:
            return pd.Series(dtype=float)

    # Normalize to date (strip time zone and intraday time)
    dates = df.index.tz_convert(None).normalize()

    # Build series of closes
    ser = pd.Series(df["close"].astype(float).values, index=dates)

    # If multiple bars per day, keep the last close
    ser = ser.groupby(level=0).last().dropna().sort_index()
    return ser


def _download_history_alpaca(api, tickers, ma_fixed, min_days=400) -> pd.DataFrame:
    need_days = max(min_days, max(ma_fixed.values()) + max(VOL_LKBK, VT_LKBK) + 40)
    end_date = datetime.utcnow().date() + timedelta(days=1)  # inclusive through "today"
    start_date = end_date - timedelta(days=need_days)

    series = {}
    missing = []
    for t in sorted(set(tickers)):
        try:
            bars = price_history(api, t, start_date, end_date, feed="iex")  # <-- IEX
        except APIError as e:
            print(f"[WARN] APIError for {t}: {e}")
            bars = None

        ser = _bars_to_series_close(bars)

        if ser.empty:
            missing.append(t)
        else:
            series[t] = ser

    if not series:
        raise RuntimeError(
            f"No bars returned for any symbols. Check data feed/permissions and dates. "
            f"Tried: {tickers} | start={start_date} end={end_date}"
        )

    df = pd.concat(series, axis=1).dropna(how="all")
    df.index.name = "Date"

    if missing:
        print(
            f"[INFO] No data for: {missing}. Proceeding with available symbols: {list(series.keys())}"
        )

    return df


def _get_last_prices_from_history(px: pd.DataFrame) -> Dict[str, float]:
    return px.ffill().iloc[-1].to_dict()


# ===========================
# Core portfolio math helpers
# ===========================
def _ma_signal(series: pd.Series, win: int) -> int:
    if len(series) < win + 5:
        return 0
    return int(series.iloc[-1] > series.rolling(win).mean().iloc[-1])


def _invvol_weights(rets: pd.DataFrame) -> pd.Series:
    vol = rets.std().replace(0, np.nan)
    inv = (1.0 / vol).fillna(0.0)
    if inv.sum() == 0:
        return pd.Series(1.0 / len(rets.columns), index=rets.columns)
    return inv / inv.sum()


def _ema_blend(
    current: pd.Series, prev: Dict[str, float] | None, alpha: float, sleeves: List[str]
) -> pd.Series:
    if prev is None or not (0 < alpha < 1):
        return current
    out = current.copy()
    for t in current.index:
        p = float(prev.get(t, 0.0))
        out[t] = (1 - alpha) * p + alpha * float(current[t])
    ssum = out[sleeves].sum()
    if ssum > 0:
        out[sleeves] = out[sleeves] / ssum
    return out


def _target_gross_from_cov(rets_window: pd.DataFrame, weights: pd.Series) -> float:
    if VT_TARGET is None or rets_window.empty:
        return 1.0
    cov = rets_window.cov() * 252.0
    w = weights.reindex(rets_window.columns).fillna(0.0).values
    port_var = float(w @ cov.values @ w.T)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    if port_vol <= 0:
        return 1.0
    gross = VT_TARGET / port_vol
    gross = min(gross, GROSS_CAP)
    if not ALLOW_MARGIN:
        gross = min(gross, 1.0)
    return float(gross)


# ==========================================
# Compute today's target weights (single run)
# ==========================================
def compute_today_target_weights(
    api,
    sleeves: List[str],
    cash: str,
    ma_fixed: Dict[str, int],
    *,
    force_rebalance: bool = False,
):
    tickers = sleeves + [cash]
    px = _download_history_alpaca(api, tickers, ma_fixed)
    if px.empty:
        raise RuntimeError(
            "Downloaded price frame is empty after aggregation. Check feed, symbols, date range."
        )
    if not set(tickers).issubset(px.columns):
        missing = sorted(set(tickers) - set(px.columns))
        raise RuntimeError(f"Missing prices for: {missing}")

    rets = px.pct_change().dropna()
    today = px.index[-1]

    # Respect month-end schedule unless forced
    if not force_rebalance and not _is_rebalance_day(px.index, today, REB):
        return None, {"reason": "not a rebalance day", "date": str(today.date())}, px

    # MA signals + min history for vol
    on_list = []
    for t in sleeves:
        sig = _ma_signal(px[t], ma_fixed.get(t, 150))
        enough = len(rets[t]) >= VOL_LKBK + 5
        if sig == 1 and enough:
            on_list.append(t)

    # Base weights: inverse-vol over "on" sleeves, else all cash
    w = pd.Series(0.0, index=tickers)
    if len(on_list) == 0:
        w[cash] = 1.0
    else:
        w_rp = _invvol_weights(rets[on_list].tail(VOL_LKBK))
        w.loc[on_list] = w_rp.values

    # EMA smoothing on sleeves
    state = _load_state()
    ema_prev = state.get("ema_prev", {})
    if EMA_ALPHA is not None:
        w = _ema_blend(w, ema_prev, EMA_ALPHA, sleeves)

    # Normalize sleeves block; leftover goes to cash
    sleeves_sum = float(w[sleeves].sum())
    if sleeves_sum > 0:
        w[sleeves] = w[sleeves] / sleeves_sum
        w[cash] = 0.0
    else:
        w[cash] = 1.0

    # Vol targeting → scale sleeves gross
    if sleeves_sum > 0 and VT_TARGET is not None:
        gross = _target_gross_from_cov(rets[sleeves].tail(VT_LKBK), w[sleeves])
        w[sleeves] = w[sleeves] * gross
        w[cash] = 1.0 - float(w[sleeves].sum())

    # Clean tiny dust / exact renorm
    w[w.abs() < 1e-8] = 0.0
    w = w / w.sum()

    meta = {
        "date": str(today.date()),
        "on_sleeves": on_list,
        "gross_sleeves": float(w[sleeves].sum()),
        "cash_weight": float(w[cash]),
        "rebalance": True,
    }

    # Persist EMA state
    _save_state(ema_prev={k: float(w[k]) for k in sleeves}, last_reb_date=today.date())
    return w, meta, px


# ==========================================
# Orders: translate weights → quantities
# ==========================================
def _round_shares(x: float, lot: int = ROUND_LOT) -> int:
    if lot <= 1:
        return int(math.floor(x))
    return int(math.floor(x / lot) * lot)


def place_orders_for_weights(
    api,
    target_w: pd.Series,
    equity: float | None,
    last_px: Dict[str, float],
    is_live_trade: bool = False,
):
    """
    Uses your provided `process_position(api, security, qty, is_live_trade)` to submit diffs.
    Returns a list of {symbol, action, target_qty, diff, price, alloc_w}.
    """
    if equity is None:
        try:
            acct = api.get_account()
            equity = float(getattr(acct, "equity", getattr(acct, "cash", "0")))
        except Exception:
            raise RuntimeError("Equity not provided and api.get_account() failed.")

    results = []
    for sym in target_w.index:
        w = float(target_w.get(sym, 0.0))
        px = float(last_px.get(sym, np.nan))
        if not np.isfinite(px) or px <= 0:
            continue

        target_dollars = equity * w
        target_qty = _round_shares(target_dollars / px, lot=ROUND_LOT)

        # Your function should be imported / defined elsewhere
        action, qty, diff = process_position(
            api, sym, target_qty, is_live_trade=is_live_trade
        )

        action_readable = "hold" if diff == 0 else ("buy" if diff > 0 else "sell")
        results.append(
            {
                "symbol": sym,
                "action": action_readable,
                "target_qty": int(qty),
                "diff": int(diff),
                "price": px,
                "alloc_w": round(w, 6),
            }
        )

    return results


# ==========================================
# Entry point: single iteration
# ==========================================
def run_single_iteration(
    api,
    sleeves: List[str],
    cash: str,
    ma_fixed: Dict[str, int],
    *,
    force_rebalance: bool = False,
    is_live_trade: bool = False,
    equity_override: float | None = None,
):
    """
    - Computes today's target weights (or exits with 'not a rebalance day').
    - Generates buy/sell/hold instructions for the provided sleeves + cash.
    - If is_live_trade=True, submits market orders via process_position().
    """
    res = compute_today_target_weights(
        api, sleeves, cash, ma_fixed, force_rebalance=force_rebalance
    )
    w, meta, px = res
    if w is None:
        return {"meta": meta, "orders": []}

    last_px = _get_last_prices_from_history(px[w.index])
    orders = place_orders_for_weights(
        api,
        w,
        equity=equality_or_none(equity_override),
        last_px=last_px,
        is_live_trade=is_live_trade,
    )
    return {"meta": meta, "weights": w.to_dict(), "orders": orders}


def equality_or_none(x):
    return x if x is not None else None


# ==========================================
# Optional: simple pretty printer
# ==========================================
def print_orders_table(result: Dict):
    meta = result.get("meta", {})
    print(
        f"Rebalance date: {meta.get('date')} | sleeves_on={meta.get('on_sleeves')} "
        f"| gross_sleeves={meta.get('gross_sleeves')} | cash={meta.get('cash_weight'):.3f}"
    )
    print("symbol   action   target_qty   diff   price    alloc_w")
    for o in result.get("orders", []):
        print(
            f"{o['symbol']:6}  {o['action']:6}  {o['target_qty']:11d}  {o['diff']:5d}  "
            f"{o['price']:8.2f}  {o['alloc_w']:.4f}"
        )
