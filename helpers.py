# === single_iteration_trader.py ===
import json
import math
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from alpaca.common.exceptions import APIError
from alpaca.data.timeframe import TimeFrame
from alpaca_adapter import AlpacaAPI

# ---------- Config (match your backtest) ----------

VOL_LKBK = 60
EMA_ALPHA = 0.30  # None to disable
VT_TARGET = 0.12  # None to disable
VT_LKBK = 40
GROSS_CAP = 1.5
REB = "W"  # "M" = week-end rebalance

MOM_LKBK = 84
MOM_SKIP = 21
REQ_POS_MOM = True  # require best equity momentum > 0


# Live-trading knobs
STATE_PATH = "trend_state.json"  # persists EMA prev weights + last_reb_date
ROUND_LOT = 1  # share granularity
ALLOW_MARGIN = False  # if True, allows gross > 1 up to GROSS_CAP

# Alpaca side constants used by your process_position()
BUY, SELL = "buy", "sell"

DEFAULT_MA = 200


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


def getenv_float(name: str, default: float) -> float:
    """
    Read an environment variable as a float.

    - Returns `default` if the variable is missing
    - Returns `default` if conversion fails
    """
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_current_positions(api) -> Dict[str, int]:
    """
    Returns {symbol: signed_qty} for all open positions.
    Alpaca qty is typically positive for long positions.
    """
    out = {}
    try:
        positions = api.list_positions()
        for p in positions:
            sym = getattr(p, "symbol", None)
            qty = int(float(getattr(p, "qty", 0)))
            if sym and qty != 0:
                out[sym] = qty
    except Exception as e:
        print(f"[WARN] Unable to list positions: {e}")
    return out


def _liquidate_positions_not_in_universe(
    api,
    keep_symbols: set,
    *,
    is_live_trade: bool,
    last_px: Dict[str, float] | None = None,
    ignore_symbols: set[str] | None = None,
) -> List[Dict]:
    """
    For any currently-held symbol not in keep_symbols, create an order to sell to 0,
    EXCEPT those in ignore_symbols.

    Uses your process_position(api, security, qty, is_live_trade) to submit diffs.

    Returns orders in the same schema as place_orders_for_weights().
    """
    last_px = last_px or {}
    ignore = set(ignore_symbols or [])
    orders = []
    pos = _get_current_positions(api)

    for sym, current_qty in sorted(pos.items()):
        # Skip anything explicitly ignored
        if sym in ignore:
            continue

        # Keep anything in the universe
        if sym in keep_symbols:
            continue

        # Target is 0 (sell all)
        action, qty, diff = process_position(api, sym, 0, is_live_trade=is_live_trade)

        # process_position() will compute diff = 0 - current_qty (negative)
        if diff == 0:
            continue

        px = float(last_px.get(sym, np.nan))
        orders.append(
            {
                "symbol": sym,
                "action": "sell",
                "target_qty": 0,
                "diff": int(diff),
                "price": px if np.isfinite(px) else np.nan,
                "alloc_w": 0.0,
                "reason": "out_of_universe",
            }
        )

    return orders


def _get_ma_window(ticker: str, ma_fixed: Dict[str, int]) -> int:
    return int(ma_fixed.get(ticker, DEFAULT_MA))


def _pick_inverse_or_cash(
    px: pd.DataFrame,
    *,
    asof: pd.Timestamp,
    cash: str,
    inverse_set: set[str],
    ma_fixed: Dict[str, int],
    ma_fixed_inverse: Dict[str, int],
    mom_lkbk: int,
    mom_skip: int,
) -> str:
    """
    Match notebook behavior:
      - Choose inverse ETF if it passes MA gate AND has positive momentum.
      - If multiple qualify, pick best momentum.
      - Else return cash.
    """
    scores: Dict[str, float] = {}

    for t in sorted(inverse_set):
        if t not in px.columns:
            continue

        win = _get_ma_window(
            t,
            ma_fixed=ma_fixed,
            inverse_set=inverse_set,
            ma_fixed_inverse=ma_fixed_inverse,
        )
        if int(_ma_signal(px[t].loc[:asof], win)) != 1:
            continue

        sc = _momentum_score(px[t], asof=asof, lookback=mom_lkbk, skip=mom_skip)
        if np.isfinite(sc) and sc > 0:
            scores[t] = float(sc)

    if not scores:
        return cash

    return max(scores, key=scores.get)


def _compute_drawdown(px_ser: pd.Series) -> float:
    s = px_ser.dropna()
    if len(s) < 10:
        return np.nan
    return float((s / s.cummax() - 1.0).min())


def _select_vt_pair_from_regime(
    *,
    px_proxy: pd.Series,
    rets_all: pd.DataFrame,
    asof: pd.Timestamp,
    vt_regime_params: Dict[str, Tuple[float, int]],
) -> Tuple[float, int, Dict[str, float]]:
    """
    Decide (VT_TARGET, VT_LKBK) using only data up to `asof`.

    vt_regime_params must include keys:
      - "stress", "benign", "elevated", "default"
    Each value is (vt_target, vt_lkbk).

    Returns: (vt_target, vt_lkbk, diagnostics)
    """
    # proxy returns
    proxy_px = px_proxy.loc[:asof].dropna()
    proxy_rets = proxy_px.pct_change().dropna()

    vol_60 = (
        (proxy_rets.tail(60).std() * np.sqrt(252)) if len(proxy_rets) >= 60 else np.nan
    )
    dd_6m = _compute_drawdown(proxy_px.tail(126)) if len(proxy_px) >= 126 else np.nan

    # correlation across sleeves
    sub = rets_all.loc[:asof].tail(60).dropna(axis=1, how="any")
    if sub.shape[1] >= 2:
        c = sub.corr().values
        avg_corr_60 = float(c[np.triu_indices_from(c, 1)].mean())
    else:
        avg_corr_60 = np.nan

    # --- default thresholds (same as your notebook logic) ---
    stress = (np.isfinite(vol_60) and vol_60 > 0.22) or (
        np.isfinite(dd_6m) and dd_6m < -0.10
    )
    elevated = (
        (np.isfinite(vol_60) and vol_60 > 0.17)
        or (np.isfinite(dd_6m) and dd_6m < -0.06)
        or (np.isfinite(avg_corr_60) and avg_corr_60 > 0.60)
    )
    benign = (np.isfinite(vol_60) and vol_60 < 0.12) and (
        np.isfinite(dd_6m) and dd_6m > -0.03
    )

    if stress:
        key = "stress"
    elif benign:
        key = "benign"
    elif elevated:
        key = "elevated"
    else:
        key = "default"

    vt_target, vt_lkbk = vt_regime_params[key]

    diag = {
        "regime": key,
        "vol_60": float(vol_60) if np.isfinite(vol_60) else np.nan,
        "dd_6m": float(dd_6m) if np.isfinite(dd_6m) else np.nan,
        "avg_corr_60": float(avg_corr_60) if np.isfinite(avg_corr_60) else np.nan,
    }
    return float(vt_target), int(vt_lkbk), diag


def process_position(api, security, qty, is_live_trade=False):
    try:
        position = api.get_position(security)  # adapter -> trading.get_open_position
        current_qty = int(float(getattr(position, "qty", 0)))

    except APIError:
        current_qty = 0

    diff = qty - current_qty

    if is_live_trade:
        if diff > 0:
            api.submit_order(
                symbol=security,
                time_in_force="day",
                side=BUY,
                type="market",
                qty=abs(diff),
            )
        elif diff < 0:
            api.submit_order(
                symbol=security,
                time_in_force="day",
                side=SELL,
                type="market",
                qty=abs(diff),
            )

    action = "NOOP" if diff == 0 else (BUY if diff > 0 else SELL)
    return action, qty, diff


# ---------- Small utils ----------
def _momentum_score(
    series: pd.Series, asof: pd.Timestamp, lookback: int, skip: int
) -> float:
    """
    Dual momentum score:
      score = price(asof - skip) / price(asof - lookback - skip) - 1
    """
    s = series.loc[:asof].dropna()
    if len(s) < (lookback + skip + 5):
        return np.nan
    end = s.iloc[-1 - skip] if skip > 0 else s.iloc[-1]
    start = s.iloc[-1 - skip - lookback]
    if start <= 0 or end <= 0:
        return np.nan
    return float(end / start - 1.0)


def price_history(
    api,
    ticker: str,
    start_date,
    end_date,
    feed: str = "iex",
):
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
        # Preserve your SIP subscription fallback
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


# --- add this helper ---
def _last_trading_day_of_week(today: pd.Timestamp) -> pd.Timestamp:
    """
    Return the last trading day (as a date-like Timestamp, tz-naive, normalized)
    for the ISO week containing `today`.

    Tries pandas-market-calendars (XNYS). Falls back to calendar Friday if not available.
    """
    # Normalize 'today' to a date-like ts
    today = pd.Timestamp(today).tz_localize(None).normalize()

    # Week bounds (Mon..Sun)
    week_start = today - pd.Timedelta(days=today.weekday())  # Monday
    week_end = week_start + pd.Timedelta(days=6)  # Sunday

    # Try robust exchange calendar if installed

    nyse = mcal.get_calendar("XNYS")
    sched = nyse.schedule(start_date=week_start.date(), end_date=week_end.date())
    if not sched.empty:
        # Index are session dates with tz; take last, drop tz and time
        last = pd.Timestamp(sched.index[-1]).tz_localize(None).normalize()
        return last

    # Fallback: calendar Friday of this week (W-FRI anchored period end)
    end_of_week = today.to_period("W-FRI").end_time.normalize()
    return end_of_week


# --- optionally keep this for other callers (completed weeks index) ---
def _week_end_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Last trading day of each *completed* week from `idx`.
    (Excludes the current, possibly incomplete, week.)
    """
    if len(idx) == 0:
        return idx

    idx = pd.DatetimeIndex(idx).tz_localize(None).normalize()
    # Group by ISO year/week and take last in each group
    df = pd.Series(1, index=idx)
    week_last = (
        df.groupby([idx.isocalendar().year, idx.isocalendar().week]).tail(1).index
    )

    # Drop the current week’s last day (to avoid midweek rebalance)
    today_week = idx[-1].isocalendar().week
    today_year = idx[-1].isocalendar().year
    mask = ~(
        (week_last.isocalendar().year == today_year)
        & (week_last.isocalendar().week == today_week)
    )
    return week_last[mask]


# --- replace your _is_rebalance_day with this ---
def _is_rebalance_day(
    idx: pd.DatetimeIndex, today: pd.Timestamp, reb_rule: str
) -> bool:
    """
    - 'M': last trading day of month (from index)
    - 'W': last trading day of week (via exchange calendar if available; else Friday)
    - otherwise: always True
    """
    rule = (reb_rule or "").upper()
    today = pd.Timestamp(today).tz_localize(None).normalize()

    if rule.startswith("M"):

        # Month-end based on available trading days in idx
        me = _month_end_index(pd.DatetimeIndex(idx).tz_localize(None))
        return today in me

    if rule.startswith("W"):

        ldw = _last_trading_day_of_week(today)
        return today == ldw

    return True


# ==========================================
# Bars → pandas Series (robust converter)
# ==========================================
def _bars_to_series_close(bars, symbol: str | None = None) -> pd.Series:
    """
    Convert Alpaca bars to a daily close Series.

    alpaca-py BarSet.df is typically MultiIndex: (symbol, timestamp).
    """
    if bars is None:
        return pd.Series(dtype=float)

    if hasattr(bars, "df"):
        df = bars.df.copy()
    else:
        try:
            df = pd.DataFrame(bars)
        except Exception:
            return pd.Series(dtype=float)

    if df.empty:
        return pd.Series(dtype=float)

    # If MultiIndex (symbol, timestamp), slice to one symbol
    if isinstance(df.index, pd.MultiIndex):
        if symbol is None:
            # if caller didn't pass symbol, try to infer the first
            try:
                symbol = df.index.get_level_values(0)[0]
            except Exception:
                return pd.Series(dtype=float)
        try:
            df = df.xs(symbol, level=0)
        except Exception:
            return pd.Series(dtype=float)

    if "close" not in df.columns:
        return pd.Series(dtype=float)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df.index = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        elif "t" in df.columns:
            df.index = pd.to_datetime(df["t"], utc=True, errors="coerce")
        else:
            return pd.Series(dtype=float)

    dates = df.index.tz_convert(None).normalize()
    ser = pd.Series(df["close"].astype(float).values, index=dates)
    return ser.groupby(level=0).last().dropna().sort_index()


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


def _invvol_weights(
    rets: pd.DataFrame,
    *,
    gamma: float = 0.6,  # <1 dampens sensitivity (recommended 0.4–0.7)
    vol_floor: float = 0.06,  # annualized floor; prevents bonds dominating
    max_w: float | None = 0.60,  # optional cap per sleeve (set None to disable)
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Improved inverse-vol weights with:
      - vol floor (annualized)
      - power-law inverse vol (gamma)
      - optional max weight cap with renormalization
    Expects daily returns in `rets`.
    """
    if rets is None or rets.empty:
        raise ValueError("rets is empty")

    # Daily vol -> annualized vol (consistent with vol_floor)
    vol_d = rets.std().replace(0, np.nan)
    vol_ann = (vol_d * np.sqrt(periods_per_year)).fillna(np.nan)

    # Floor + sanitize
    vol_ann = vol_ann.clip(lower=vol_floor)
    vol_ann = vol_ann.replace([np.inf, -np.inf], np.nan)

    # Power-law inverse vol
    inv = 1.0 / (vol_ann ** float(gamma))
    inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Fallback if all zeros
    if inv.sum() <= 0:
        return pd.Series(1.0 / len(rets.columns), index=rets.columns)

    w = inv / inv.sum()

    # Optional max cap (iterative renormalization for stability)
    if max_w is not None:
        max_w = float(max_w)
        w = w.clip(lower=0.0)
        for _ in range(10):
            over = w > max_w
            if not over.any():
                break
            excess = (w[over] - max_w).sum()
            w[over] = max_w
            under = ~over
            if under.any() and w[under].sum() > 0 and excess > 0:
                w[under] += excess * (w[under] / w[under].sum())
            else:
                break

        # Final normalize
        if w.sum() > 0:
            w = w / w.sum()
        else:
            w = pd.Series(1.0 / len(rets.columns), index=rets.columns)

    return w


def _ema_blend(
    current: pd.Series,
    prev: Dict[str, float] | None,
    alpha: float,
    risky_sleeves: List[str],
    allowed_risky: List[str],
    cash: str,
) -> pd.Series:
    """
    EMA smoothing that prevents "smear" into OFF assets:

    - Smoothes only within the risky block.
    - Forces risky weights outside allowed_risky to 0 (post-blend).
    - Renormalizes allowed_risky to sum to 1 if any are non-zero;
      otherwise goes 100% cash.

    This keeps your live logic aligned to the notebook behavior.
    """
    if prev is None or not (0 < alpha < 1):
        out = current.copy()
    else:
        out = current.copy()
        for t in risky_sleeves + [cash]:
            p = float(prev.get(t, 0.0))
            out[t] = (1 - alpha) * p + alpha * float(current.get(t, 0.0))

    # Force OFF risky sleeves to 0
    allowed_set = set(allowed_risky)
    for t in risky_sleeves:
        if t not in allowed_set:
            out[t] = 0.0

    # Renormalize within allowed risky; remainder -> cash
    ssum = float(out[allowed_risky].sum()) if allowed_risky else 0.0
    if ssum > 0:
        out[allowed_risky] = out[allowed_risky] / ssum
        for t in risky_sleeves:
            if t not in allowed_set:
                out[t] = 0.0
        out[cash] = 0.0
    else:
        for t in risky_sleeves:
            out[t] = 0.0
        out[cash] = 1.0

    # Final sanitize
    out[out.abs() < 1e-10] = 0.0
    tot = float(out.sum())
    if tot > 0:
        out = out / tot
    else:
        out[:] = 0.0
        out[cash] = 1.0
    return out


def _target_gross_from_cov(
    rets_window: pd.DataFrame,
    weights: pd.Series,
    *,
    vt_target: float | None,
    gross_cap: float,
    allow_margin: bool,
) -> float:
    """
    Compute a portfolio gross scaler based on realized vol of the weighted basket.

    - vt_target: annualized volatility target (e.g., 0.12). If None, returns 1.0.
    - gross_cap: cap on gross exposure.
    - allow_margin: if False, cap gross at 1.0 even if gross_cap > 1.
    """
    if vt_target is None or rets_window.empty:
        return 1.0

    cov = rets_window.cov() * 252.0
    w = weights.reindex(rets_window.columns).fillna(0.0).values
    port_var = float(w @ cov.values @ w.T)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    if port_vol <= 0:
        return 1.0

    gross = float(vt_target) / port_vol
    gross = min(gross, float(gross_cap))
    if not allow_margin:
        gross = min(gross, 1.0)

    return float(gross)


# ==========================================
# Compute today's target weights (single run)
# ==========================================
def compute_today_target_weights_dual_mom_equity(
    api,
    equity_cands: List[str],
    other_sleeves: List[str],
    cash: str,
    ma_fixed: Dict[str, int],
    *,
    force_rebalance: bool = False,
    mom_lkbk: int = MOM_LKBK,
    mom_skip: int = MOM_SKIP,
    req_pos_mom: bool = REQ_POS_MOM,
    use_dynamic_vt: bool = False,
    vt_regime_params: Dict[str, Tuple[float, int]] | None = None,
    proxy_ticker: str | None = None,
):
    """
    Notebook-aligned live weights (no inverse sleeve):
      - Rebalance schedule (month-end/week-end unless forced)
      - Signals computed as-of prior trading day (no lookahead)
      - Dual Momentum on equities (select ONE equity candidate)
      - MA gating for other sleeves
      - inv-vol RP across active risky sleeves
      - EMA smoothing restricted to allowed risky sleeves
      - Vol targeting from trailing covariance
      - Defensive fallback: CASH only
    """

    # --- Universe ---
    risky_sleeves = list(dict.fromkeys(equity_cands + other_sleeves))
    tickers = list(dict.fromkeys(risky_sleeves + [cash]))

    # --- Download history ---
    px = _download_history_alpaca(api, tickers, ma_fixed)
    if px.empty:
        raise RuntimeError(
            "Downloaded price frame is empty after aggregation. Check feed/symbols/date range."
        )
    if not set(tickers).issubset(px.columns):
        missing = sorted(set(tickers) - set(px.columns))
        raise RuntimeError(f"Missing prices for: {missing}")

    px = px.dropna(how="all")
    rets = px.pct_change().dropna()
    today = px.index[-1]

    # Respect schedule unless forced
    if not force_rebalance and not _is_rebalance_day(px.index, today, REB):
        return None, {"reason": "not a rebalance day", "date": str(today.date())}, px

    # No-lookahead: use prior trading day for MA/momentum/cov
    if len(px.index) < 3:
        return None, {"reason": "insufficient history", "date": str(today.date())}, px
    asof = px.index[-2]

    # -----------------------------
    # Dynamic VT regime selection (as-of prior day)
    # -----------------------------
    vt_target_use = VT_TARGET
    vt_lkbk_use = VT_LKBK
    vt_diag = {}

    if use_dynamic_vt:
        # Default regime map if not provided
        if vt_regime_params is None:
            vt_regime_params = {
                "stress": (0.06, 20),
                "benign": (0.14, 20),
                "elevated": (0.10, 20),
                "default": (0.12, 20),
            }

        # choose proxy series for regime detection
        if proxy_ticker and proxy_ticker in px.columns:
            px_proxy = px[proxy_ticker]
        elif len(equity_cands) > 0 and equity_cands[0] in px.columns:
            px_proxy = px[equity_cands[0]]
        else:
            # fallback: average of risky sleeves that exist
            exist = [t for t in (equity_cands + other_sleeves) if t in px.columns]
            px_proxy = px[exist].mean(axis=1) if exist else px[cash]

        vt_target_use, vt_lkbk_use, vt_diag = _select_vt_pair_from_regime(
            px_proxy=px_proxy,
            rets_all=rets,
            asof=asof,
            vt_regime_params=vt_regime_params,
        )

    # MA gate for a ticker using data up to asof
    def is_on(t: str) -> bool:
        win = _get_ma_window(ticker=t, ma_fixed=ma_fixed)
        return int(_ma_signal(px[t].loc[:asof], win)) == 1

    # --- 1) Dual Momentum equity selection (ONE equity candidate, among MA-on) ---
    eq_on = [t for t in equity_cands if (t in px.columns) and is_on(t)]
    eq_scores: Dict[str, float] = {}
    eq_pick: Optional[str] = None

    for t in eq_on:
        sc = _momentum_score(px[t], asof=asof, lookback=mom_lkbk, skip=mom_skip)
        if np.isfinite(sc):
            eq_scores[t] = float(sc)

    if eq_scores:
        eq_pick = max(eq_scores, key=eq_scores.get)
        if req_pos_mom and eq_scores.get(eq_pick, -np.inf) <= 0:
            eq_pick = None

    # --- 2) Other sleeves MA-on + enough history for VOL_LKBK ---
    other_on = [
        t
        for t in other_sleeves
        if (t in px.columns) and is_on(t) and (len(rets[t].loc[:asof]) >= VOL_LKBK + 5)
    ]

    allowed_risky = ([eq_pick] if eq_pick is not None else []) + other_on

    # --- 3) Base weights: inv-vol across allowed risky; else CASH ---
    w = pd.Series(0.0, index=tickers)

    if not allowed_risky:
        w[cash] = 1.0
    else:
        sub = rets[allowed_risky].loc[:asof].tail(VOL_LKBK)
        w_rp = _invvol_weights(
            sub,
            gamma=0.6,
            vol_floor=0.06,
            max_w=0.60,
            periods_per_year=252,
        )
        w.loc[w_rp.index] = w_rp.values

        # Ensure only allowed risky have weight; cash off in risk mode
        allowed_set = set(allowed_risky)
        for t in risky_sleeves:
            if t not in allowed_set:
                w[t] = 0.0
        w[cash] = 0.0

    # --- 4) EMA smoothing restricted to allowed risky sleeves ---
    state = _load_state()
    ema_prev = state.get("ema_prev", {})

    if EMA_ALPHA is not None:
        w = _ema_blend(
            current=w,
            prev=ema_prev,
            alpha=float(EMA_ALPHA),
            risky_sleeves=risky_sleeves,
            allowed_risky=allowed_risky,
            cash=cash,
        )

        # If defensive (no risky), force cash=1
        if not allowed_risky:
            w[:] = 0.0
            w[cash] = 1.0

    # --- 5) Normalize risky block; if none, keep cash ---
    risky_sum = float(w[risky_sleeves].sum())
    if risky_sum > 0:
        w[risky_sleeves] = w[risky_sleeves] / risky_sum
        w[cash] = 0.0
    else:
        for t in risky_sleeves:
            w[t] = 0.0
        w[cash] = 1.0

    # --- 6) Vol targeting (only for risky mode) ---
    if float(w[risky_sleeves].sum()) > 0 and vt_target_use is not None:
        cov_win = rets[risky_sleeves].loc[:asof].tail(int(vt_lkbk_use))
        gross = _target_gross_from_cov(
            cov_win,
            w[risky_sleeves],
            vt_target=float(vt_target_use),
            gross_cap=float(GROSS_CAP),
            allow_margin=bool(ALLOW_MARGIN),
        )

    # Clean dust / renormalize
    w[w.abs() < 1e-8] = 0.0
    tot = float(w.sum())
    if tot <= 0:
        w[:] = 0.0
        w[cash] = 1.0
    else:
        w = w / tot

    meta = {
        "date": str(today.date()),
        "asof": str(asof.date()),
        "rebalance": True,
        "equity_candidates_on": eq_on,
        "equity_pick": eq_pick,
        "equity_scores": eq_scores,
        "on_sleeves": allowed_risky,
        "gross_risky": float(w[risky_sleeves].sum()),
        "cash_weight": float(w.get(cash, 0.0)),
        "vt_mode": "dynamic" if use_dynamic_vt else "static",
        "vt_target_used": float(vt_target_use) if vt_target_use is not None else None,
        "vt_lkbk_used": int(vt_lkbk_use) if vt_target_use is not None else None,
        "vt_diag": vt_diag,
    }

    # Persist EMA state for risky sleeves only
    _save_state(
        ema_prev={k: float(w.get(k, 0.0)) for k in risky_sleeves},
        last_reb_date=today.date(),
    )
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
    Filters out pure holds (diff==0) to keep the output clean.
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

        # If target weight is exactly 0, we still want to SELL to zero if currently held.
        px = float(last_px.get(sym, np.nan))
        if not np.isfinite(px) or px <= 0:
            # Still attempt liquidation if weight=0; process_position doesn't require px
            target_qty = 0
        else:
            target_dollars = equity * w
            target_qty = _round_shares(target_dollars / px, lot=ROUND_LOT)

        action, qty, diff = process_position(
            api, sym, int(target_qty), is_live_trade=is_live_trade
        )

        # Filter out holds (no change)
        if int(diff) == 0:
            continue

        action_readable = "buy" if diff > 0 else "sell"
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
    equity_cands: List[str],
    other_sleeves: List[str],
    cash: str,
    ma_fixed: Dict[str, int],
    *,
    force_rebalance: bool = False,
    is_live_trade: bool = False,
    equity_override: float | None = None,
    equity_fraction: float = 1.0,
    liquidate_out_of_universe: bool = True,
    ignore_liquidation_symbols: set[str] | None = None,
    vt_regime_params: Dict[str, Tuple[float, int]] | None = None,
    use_dynamic_vt: bool = False,
    proxy_ticker: str | None = None,
):
    """
    - Computes today's target weights (or exits with 'not a rebalance day').
    - Submits orders to reach those weights.
    - OPTIONAL: sells to zero any *currently held* symbols that are not in the
      current universe (equity_cands + other_sleeves + cash).
    """
    w, meta, px = compute_today_target_weights_dual_mom_equity(
        api,
        equity_cands=equity_cands,
        other_sleeves=other_sleeves,
        cash=cash,
        ma_fixed=ma_fixed,
        force_rebalance=force_rebalance,
        use_dynamic_vt=use_dynamic_vt,
        vt_regime_params=vt_regime_params,
        proxy_ticker=proxy_ticker,
    )

    if w is None:
        return {"meta": meta, "orders": []}

    universe = set(equity_cands) | set(other_sleeves) | {cash}

    last_px = _get_last_prices_from_history(px[w.index])

    if not (0.0 < float(equity_fraction) <= 1.0):
        raise ValueError(f"equity_fraction must be in (0, 1], got {equity_fraction}")

    if equity_override is not None:
        acct_equity = float(equity_override)
    else:
        acct = api.get_account()
        acct_equity = float(getattr(acct, "equity", getattr(acct, "cash", "0")))

    trade_equity = acct_equity * float(equity_fraction)

    orders = place_orders_for_weights(
        api,
        w,
        equity=trade_equity,
        last_px=last_px,
        is_live_trade=is_live_trade,
    )

    if liquidate_out_of_universe and float(equity_fraction) >= 0.999:
        exit_orders = _liquidate_positions_not_in_universe(
            api,
            keep_symbols=universe,
            is_live_trade=is_live_trade,
            last_px=last_px,
            ignore_symbols=ignore_liquidation_symbols,
        )
        orders.extend(exit_orders)

    return {"meta": meta, "weights": w.to_dict(), "orders": orders}


# ==========================================
# Optional: simple pretty printer
# ==========================================
def print_orders_table(result: dict):

    def fmt_num(x, prec=3):
        return (
            f"{x:.{prec}f}"
            if isinstance(x, (int, float, np.floating)) and pd.notna(x)
            else "N/A"
        )

    meta = result.get("meta", {}) or {}
    date = meta.get("date", "N/A")
    reason = meta.get("reason")  # present when it's not a rebalance day
    on = meta.get("on_sleeves", [])
    gross = fmt_num(meta.get("gross_risky"))
    cash = fmt_num(meta.get("cash_weight"))

    # --- NEW: dynamic VT regime diagnostics (stress level) ---
    vt_mode = meta.get("vt_mode", "static")
    vt_target_used = meta.get("vt_target_used", None)
    vt_lkbk_used = meta.get("vt_lkbk_used", None)

    vt_diag = meta.get("vt_diag", {}) or {}
    stress_level = vt_diag.get("regime", "N/A")
    vol_60 = vt_diag.get("vol_60", np.nan)
    dd_6m = vt_diag.get("dd_6m", np.nan)
    avg_corr_60 = vt_diag.get("avg_corr_60", np.nan)

    if reason:
        print(f"Rebalance date: {date} | reason={reason}")
        if not result.get("orders"):
            print("(No orders)")
        return

    header = (
        f"Rebalance date: {date} | sleeves_on={on} | gross_sleeves={gross} | cash={cash}\n"
        f"VT mode={vt_mode} | stress_level={stress_level} | "
        f"vt_target_used={fmt_num(vt_target_used, 3)} | vt_lkbk_used={vt_lkbk_used}\n"
        f"diag: vol_60={fmt_num(vol_60, 3)} | dd_6m={fmt_num(dd_6m, 3)} | avg_corr_60={fmt_num(avg_corr_60, 3)}"
    )
    print(header)

    orders = result.get("orders", [])
    if not orders:
        print("(No orders)")
        return

    print("symbol   action   target_qty   diff     price     alloc_w")
    for o in orders:
        px = o.get("price", np.nan)
        alloc_w = o.get("alloc_w", np.nan)
        print(
            f"{o['symbol']:6}  {o['action']:6}  {int(o['target_qty']):11d}  {int(o['diff']):6d}  "
            f"{(float(px) if pd.notna(px) else np.nan):9.4f}  {float(alloc_w):.4f}"
        )
