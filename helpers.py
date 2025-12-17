# === single_iteration_trader.py ===
import json
import math
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from alpaca_trade_api.rest import APIError, TimeFrame

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
) -> Tuple[Optional[pd.Series], Dict, pd.DataFrame]:
    """
    Notebook-aligned live weights:
      - Rebalance schedule (month-end unless forced)
      - Signals computed as-of prior trading day (no lookahead)
      - Dual Momentum on equities (select ONE equity candidate)
      - MA gating for other sleeves
      - inv-vol RP across active risky sleeves
      - EMA smoothing restricted to allowed risky sleeves
      - Vol targeting from trailing covariance
    """
    risky_sleeves = list(dict.fromkeys(equity_cands + other_sleeves))
    tickers = risky_sleeves + [cash]

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

    # Respect month-end schedule unless forced
    if not force_rebalance and not _is_rebalance_day(px.index, today, REB):
        return None, {"reason": "not a rebalance day", "date": str(today.date())}, px

    # No-lookahead: use prior trading day for MA/momentum/cov
    if len(px.index) < 3:
        return None, {"reason": "insufficient history", "date": str(today.date())}, px
    asof = px.index[-2]

    # MA gate for a ticker using data up to asof
    def is_on(t: str) -> bool:
        win = ma_fixed.get(t, 150)
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

    # --- 3) Base weights: inv-vol across allowed risky, else 100% cash ---
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
        )  # returns pd.Series in your implementation
        w.loc[w_rp.index] = w_rp.values
        # Ensure only allowed risky have weight
        for t in risky_sleeves:
            if t not in set(allowed_risky):
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

    # --- 5) Ensure risky block normalized; leftover to cash (same convention as notebook) ---
    risky_sum = float(w[risky_sleeves].sum())
    if risky_sum > 0:
        w[risky_sleeves] = w[risky_sleeves] / risky_sum
        w[cash] = 0.0
    else:
        w[risky_sleeves] = 0.0
        w[cash] = 1.0

    # --- 6) Vol targeting from trailing covariance up to asof ---
    if float(w[risky_sleeves].sum()) > 0 and VT_TARGET is not None:
        cov_win = rets[risky_sleeves].loc[:asof].tail(VT_LKBK)
        gross = _target_gross_from_cov(
            cov_win, w[risky_sleeves]
        )  # uses your VT_TARGET/GROSS_CAP/ALLOW_MARGIN
        w[risky_sleeves] = w[risky_sleeves] * gross
        w[cash] = 1.0 - float(w[risky_sleeves].sum())

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
        "cash_weight": float(w[cash]),
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
    equity_cands: List[str],
    other_sleeves: List[str],
    cash: str,
    ma_fixed: Dict[str, int],
    *,
    force_rebalance: bool = False,
    is_live_trade: bool = False,
    equity_override: float | None = None,
):
    """
    - Computes today's target weights (or exits with 'not a rebalance day').
    - Dual Momentum selects ONE equity among equity_cands (if eligible).
    - Other sleeves are included if MA-on.
    - If is_live_trade=True, submits orders.
    """
    w, meta, px = compute_today_target_weights_dual_mom_equity(
        api,
        equity_cands=equity_cands,
        other_sleeves=other_sleeves,
        cash=cash,
        ma_fixed=ma_fixed,
        force_rebalance=force_rebalance,
    )
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
def print_orders_table(result: dict):

    def fmt_num(x, prec=3):
        return f"{x:.{prec}f}" if isinstance(x, (int, float)) and pd.notna(x) else "N/A"

    meta = result.get("meta", {}) or {}
    date = meta.get("date", "N/A")
    reason = meta.get("reason")  # present when it's not a rebalance day
    on = meta.get("on_sleeves", [])
    gross = fmt_num(meta.get("gross_sleeves"))
    cash = fmt_num(meta.get("cash_weight"))

    if reason:
        print(f"Rebalance date: {date} | reason={reason}")
        if not result.get("orders"):
            print("(No orders)")
        return

    print(
        f"Rebalance date: {date} | sleeves_on={on} | gross_sleeves={gross} | cash={cash}"
    )

    orders = result.get("orders", [])
    if not orders:
        print("(No orders)")
        return

    print("symbol   action   target_qty   diff     price     alloc_w")
    for o in orders:
        print(
            f"{o['symbol']:6}  {o['action']:6}  {o['target_qty']:11d}  {o['diff']:6d}  "
            f"{o['price']:9.4f}  {o['alloc_w']:.4f}"
        )
