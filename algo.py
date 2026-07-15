import os

from alpaca_adapter import AlpacaAPI
from dotenv import find_dotenv, load_dotenv
from helpers import (
    REB,
    export_strategy_json,
    getenv_float,
    next_rebalance_day,
    print_orders_table,
    print_weights_table,
    result_trade_today,
    run_single_iteration,
    str2bool,
    upload_file_to_digitalocean_spaces,
)
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

vt_map = {
    "stress": (0.06, 20),
    "benign": (0.14, 20),
    "elevated": (0.10, 20),
    "default": (0.12, 20),
}


def format_observe_allocations(weights: dict) -> list[str]:
    if not isinstance(weights, dict):
        return []

    non_zero = [
        (symbol, float(weight))
        for symbol, weight in weights.items()
        if abs(float(weight)) > 1e-12
    ]
    non_zero.sort(key=lambda item: item[1], reverse=True)
    return [f"{symbol}: {weight * 100:.2f}%" for symbol, weight in non_zero]


def get_app_state() -> str:
    app_state = (os.getenv("APP_STATE") or os.getenv("RUN_MODE") or "").strip().upper()
    valid_states = {"LIVE", "PAPER", "OBSERVE"}
    if app_state not in valid_states:
        raise ValueError(
            "APP_STATE must be set to one of "
            f"{sorted(valid_states)}; got {app_state!r}."
        )
    return app_state


def is_paper_account() -> bool:
    base_url = (os.getenv("ALPACA_BASE_URL") or "").lower()
    alpaca_paper = os.getenv("ALPACA_PAPER")
    return (
        str2bool(alpaca_paper) if alpaca_paper is not None else ("paper" in base_url)
    )


def main():
    load_dotenv(find_dotenv())

    use_dynamic_vt = str2bool(os.getenv("USE_DYNAMIC_VT", False))
    equity_fraction = getenv_float("EQUITY_FRACTION", 1)
    forced_rebalance = str2bool(os.getenv("FORCED_REBALANCE", False))
    sync_strategy_json_to_spaces = str2bool(
        os.getenv("SYNC_STRATEGY_JSON_TO_SPACES", False)
    )
    email_positions = str2bool(os.getenv("EMAIL_POSITIONS", False))

    app_state = get_app_state()
    is_observe = app_state == "OBSERVE"
    live_trade = app_state == "LIVE"

    log(f"Running in {app_state} mode", "info")

    api = AlpacaAPI.from_env(
        api_key=os.getenv("ALPACA_KEY_ID"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
        paper=is_paper_account(),
    )

    account = api.get_account()
    portfolio_value = round(float(account.equity), 3)

    portfolio = run_single_iteration(
        api,
        equity_cands=EQUITY_CANDS,
        other_sleeves=OTHER_SLEEVES,
        cash=CASH,
        ma_fixed=MA_FIXED,
        use_dynamic_vt=use_dynamic_vt,
        vt_regime_params=vt_map,
        equity_fraction=equity_fraction,
        # Observe mode should still respect the configured rebalance cadence.
        # Use FORCED_REBALANCE explicitly when an off-schedule sync is desired.
        force_rebalance=forced_rebalance,
        is_live_trade=live_trade,
        ignore_liquidation_symbols=None,
        persist_state=not is_observe,
    )
    out = print_weights_table(portfolio) if is_observe else print_orders_table(portfolio)
    scheduled_trade_today = result_trade_today(portfolio)
    meta = (portfolio or {}).get("meta", {}) or {}

    if sync_strategy_json_to_spaces:
        output_path = "etf-trend-rp-vt.json"
        log(f"Export Strategy Results: {output_path}", "info")
        export_strategy_json(
            result=portfolio,
            output_path=output_path,
            strategy_name="trend",
            equity_fraction=equity_fraction,
            trade_today=scheduled_trade_today if is_observe else None,
            liquidate_when_inactive=False if is_observe else None,
        )

        log(f"Save to Spaces: {os.environ.get('SPACES_BUCKET')}", "info")
        upload_file_to_digitalocean_spaces(
            file_path=output_path,
            region=os.environ.get("SPACES_REGION"),
            object_key=f"{os.environ.get('SPACES_OBJECT_KEY_PATH')}/{output_path}",
            bucket_name=os.environ.get("SPACES_BUCKET"),
            access_key=os.environ.get("SPACES_KEY"),
            secret_key=os.environ.get("SPACES_SECRET"),
        )

    portfolio_value_display = f"${portfolio_value:,.2f}"
    trade_today_display = "Yes" if scheduled_trade_today else "No"
    run_date = meta.get("date")
    next_rebalance_display = (
        str(next_rebalance_day(run_date, REB).date()) if run_date is not None else "N/A"
    )
    allocation_lines = (
        format_observe_allocations((portfolio or {}).get("weights", {}))
        if is_observe and scheduled_trade_today
        else []
    )
    message_body_html = (
        f"<strong>Portfolio Value:</strong> {portfolio_value_display}<br><br>"
    )
    message_body_plain = f"Portfolio Value: {portfolio_value_display}\n\n"
    if is_observe:
        message_body_html += (
            f"<strong>Trade Today:</strong> {trade_today_display}<br><br>"
        )
        message_body_html += (
            f"<strong>Next Rebalance Day:</strong> {next_rebalance_display}<br><br>"
        )
        message_body_plain += f"Trade Today: {trade_today_display}\n\n"
        message_body_plain += f"Next Rebalance Day: {next_rebalance_display}\n\n"
        if allocation_lines:
            message_body_html += "<strong>Portfolio Allocations:</strong><br><pre>"
            message_body_html += "<br>".join(allocation_lines)
            message_body_html += "</pre><br>"
            message_body_plain += (
                "Portfolio Allocations:\n" + "\n".join(allocation_lines) + "\n\n"
            )
    message_body_html += "<pre>" + out.replace("\n", "<br>") + "</pre>"
    message_body_plain += out

    if email_positions:
        to_addresses = [
            a.strip() for a in os.getenv("TO_ADDRESSES", "").split(",") if a.strip()
        ]
        from_address = os.getenv("FROM_ADDRESS", "")

        ses = AmazonSES(
            region=os.environ.get("AWS_SES_REGION_NAME"),
            access_key=os.environ.get("AWS_SES_ACCESS_KEY_ID"),
            secret_key=os.environ.get("AWS_SES_SECRET_ACCESS_KEY"),
            from_address=from_address,
        )

        status = app_state.title()

        if use_dynamic_vt:
            vt_diag = meta.get("vt_diag", {}) or {}
            stress_level = vt_diag.get("regime", "N/A")
            subject = f"Trend Algo Report - {status} | stress-level={stress_level}"
        else:
            subject = f"Trend Algo Report - {status}"

        for to_address in to_addresses:
            ses.send_html_email(
                to_address=to_address, subject=subject, content=message_body_html
            )

    print("---------------------------------------------------\n")
    print(message_body_plain)


if __name__ == "__main__":
    main()
