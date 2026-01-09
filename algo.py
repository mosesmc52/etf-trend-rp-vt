import os

import alpaca_trade_api as tradeapi
from dotenv import find_dotenv, load_dotenv
from helpers import getenv_float, print_orders_table, run_single_iteration, str2bool
from log import log
from SES import AmazonSES

# ---------- Config (match your backtest) ----------
EQUITY_CANDS = ["QLD", "QQQ"]
OTHER_SLEEVES = ["SMH", "HYMB", "GLDM"]
CASH = "BIL"

MA_FIXED = {
    "SMH": 200,
    "HYMB": 50,
    "GLDM": 110,
}

EQUITY_FRACTION = getenv_float("EQUITY_FRACTION", 1)

LIQUIDATION_SYMBOLS_TO_IGNORE = None

load_dotenv(find_dotenv())

FORCED_REBALANCE = str2bool(os.getenv("FORCED_REBALANCE", False))
LIVE_TRADE = str2bool(os.getenv("LIVE_TRADE", False))
log(f"Running in {'LIVE' if LIVE_TRADE else 'TEST'} mode", "info")

api = tradeapi.REST(
    os.getenv("ALPACA_KEY_ID"),
    os.getenv("ALPACA_SECRET_KEY"),
    base_url=os.getenv("ALPACA_BASE_URL"),
)

account = api.get_account()
portfolio_value = round(float(account.equity), 3)

portfolio = run_single_iteration(
    api,
    equity_cands=EQUITY_CANDS,
    other_sleeves=OTHER_SLEEVES,
    cash=CASH,
    ma_fixed=MA_FIXED,
    equity_fraction=EQUITY_FRACTION,
    force_rebalance=FORCED_REBALANCE,
    is_live_trade=LIVE_TRADE,
    ignore_liquidation_symbols=LIQUIDATION_SYMBOLS_TO_IGNORE,
)
print_orders_table(portfolio)


# # Email Positions
EMAIL_POSITIONS = str2bool(os.getenv("EMAIL_POSITIONS", False))
#
message_body_html = f"Portfolio Value: {portfolio_value}<br>"
message_body_plain = f"Portfolio Value: {portfolio_value}\n"
for position in portfolio["orders"]:
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
    TO_ADDRESSES = os.getenv("TO_ADDRESSES", "").split(",")
    FROM_ADDRESS = os.getenv("FROM_ADDRESS", "")
    ses = AmazonSES(
        region=os.environ.get("AWS_SES_REGION_NAME"),
        access_key=os.environ.get("AWS_SES_ACCESS_KEY_ID"),
        secret_key=os.environ.get("AWS_SES_SECRET_ACCESS_KEY"),
        from_address=os.environ.get("FROM_ADDRESS"),
    )
    if LIVE_TRADE:
        status = "Live"
    else:
        status = "Test"

    subject = "Montly Trend Algo Report - {}".format(status)

    for to_address in TO_ADDRESSES:
        ses.send_html_email(
            to_address=to_address, subject=subject, content=message_body_html
        )

print("---------------------------------------------------\n")
print(message_body_plain)
