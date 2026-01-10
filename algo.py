import os

from alpaca_adapter import AlpacaAPI
from dotenv import find_dotenv, load_dotenv
from helpers import getenv_float, print_orders_table, run_single_iteration, str2bool
from log import log
from SES import AmazonSES

load_dotenv(find_dotenv())

# ---------- Config (match your backtest) ----------
EQUITY_CANDS = ["QLD", "QQQ"]
OTHER_SLEEVES = ["SMH", "HYMB", "GLDM"]
CASH = "BIL"

MA_FIXED = {
    "SMH": 200,
    "HYMB": 50,
    "GLDM": 110,
}


USE_DYNAMIC_VT = str2bool(os.getenv("USE_DYNAMIC_VT", False))
vt_map = {
    "stress": (0.06, 20),
    "benign": (0.14, 20),
    "elevated": (0.10, 20),
    "default": (0.12, 20),
}

EQUITY_FRACTION = getenv_float("EQUITY_FRACTION", 1)

LIQUIDATION_SYMBOLS_TO_IGNORE = None


FORCED_REBALANCE = str2bool(os.getenv("FORCED_REBALANCE", False))
LIVE_TRADE = str2bool(os.getenv("LIVE_TRADE", False))
log(f"Running in {'LIVE' if LIVE_TRADE else 'TEST'} mode", "info")

alpaca_key = os.getenv("ALPACA_KEY_ID")
alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
base_url = (os.getenv("ALPACA_BASE_URL") or "").lower()

# Simple heuristic: treat "paper" URLs as paper trading
is_paper = ("paper" in base_url) or str2bool(os.getenv("ALPACA_PAPER", True))

api = AlpacaAPI.from_env(
    api_key=alpaca_key,
    secret_key=alpaca_secret,
    paper=is_paper,
)

account = api.get_account()
portfolio_value = round(float(account.equity), 3)

portfolio = run_single_iteration(
    api,
    equity_cands=EQUITY_CANDS,
    other_sleeves=OTHER_SLEEVES,
    cash=CASH,
    ma_fixed=MA_FIXED,
    use_dynamic_vt=USE_DYNAMIC_VT,
    vt_regime_params=vt_map,
    equity_fraction=EQUITY_FRACTION,
    force_rebalance=FORCED_REBALANCE,
    is_live_trade=LIVE_TRADE,
    ignore_liquidation_symbols=LIQUIDATION_SYMBOLS_TO_IGNORE,
)
print_orders_table(portfolio)


# # Email Positions
EMAIL_POSITIONS = str2bool(os.getenv("EMAIL_POSITIONS", False))

message_body_html = ""
message_body_plain = ""
if USE_DYNAMIC_VT:
    # --- pull stress/regime info from meta (safe defaults) ---
    meta = (portfolio or {}).get("meta", {}) or {}
    vt_diag = meta.get("vt_diag", {}) or {}
    stress_level = vt_diag.get(
        "regime", "N/A"
    )  # 'stress'/'elevated'/'benign'/'default' or N/A
    vt_mode = meta.get("vt_mode", "static")
    vt_target_used = meta.get("vt_target_used", None)
    vt_lkbk_used = meta.get("vt_lkbk_used", None)

    # --- message body header ---
    message_body_html = (
        f"Portfolio Value: {portfolio_value}<br>"
        f"VT mode: {vt_mode}<br>"
        f"Stress level: {stress_level}<br>"
        f"VT used: target={vt_target_used} lookback={vt_lkbk_used}<br><br>"
    )

    message_body_plain = (
        f"Portfolio Value: {portfolio_value}\n"
        f"VT mode: {vt_mode}\n"
        f"Stress level: {stress_level}\n"
        f"VT used: target={vt_target_used} lookback={vt_lkbk_used}\n\n"
    )


# --- orders ---
for position in portfolio.get("orders", []):
    reason = position.get("reason", "")
    suffix = f" [{reason}]" if reason else ""

    message_body_html += (
        f'<a clicktracking=off href="https://finviz.com/quote.ashx?t={position["symbol"]}">'
        f'{position["symbol"]}</a>: {position["target_qty"]} '
        f'({position["action"]} {round(position["alloc_w"],3)}){suffix}<br>'
    )

    message_body_plain += (
        f'{position["symbol"]}: {position["target_qty"]} '
        f'({position["action"]} {round(position["alloc_w"], 3)}){suffix}\n'
    )

if EMAIL_POSITIONS:
    TO_ADDRESSES = [
        a.strip() for a in os.getenv("TO_ADDRESSES", "").split(",") if a.strip()
    ]
    FROM_ADDRESS = os.getenv("FROM_ADDRESS", "")

    ses = AmazonSES(
        region=os.environ.get("AWS_SES_REGION_NAME"),
        access_key=os.environ.get("AWS_SES_ACCESS_KEY_ID"),
        secret_key=os.environ.get("AWS_SES_SECRET_ACCESS_KEY"),
        from_address=FROM_ADDRESS,
    )

    status = "Live" if LIVE_TRADE else "Test"

    if USE_DYNAMIC_VT:
        subject = f"Monthly Trend Algo Report - {status} | regime={stress_level}"
    else:
        subject = f"Monthly Trend Algo Report - {status}"

    for to_address in TO_ADDRESSES:
        ses.send_html_email(
            to_address=to_address, subject=subject, content=message_body_html
        )

print("---------------------------------------------------\n")
print(message_body_plain)
