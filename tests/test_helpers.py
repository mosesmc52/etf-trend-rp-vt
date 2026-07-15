import importlib
import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd


def load_helpers_module():
    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *args, **kwargs: None

    fake_mcal = types.ModuleType("pandas_market_calendars")
    fake_mcal.get_calendar = lambda *args, **kwargs: None

    fake_alpaca_adapter = types.ModuleType("alpaca_adapter")
    fake_alpaca_adapter.AlpacaAPI = object

    fake_exceptions = types.ModuleType("alpaca.common.exceptions")

    class FakeAPIError(Exception):
        pass

    fake_exceptions.APIError = FakeAPIError

    fake_timeframe = types.ModuleType("alpaca.data.timeframe")

    class FakeTimeFrame:
        Day = "day"

    fake_timeframe.TimeFrame = FakeTimeFrame

    stub_modules = {
        "boto3": fake_boto3,
        "pandas_market_calendars": fake_mcal,
        "alpaca_adapter": fake_alpaca_adapter,
        "alpaca.common.exceptions": fake_exceptions,
        "alpaca.data.timeframe": fake_timeframe,
    }

    with patch.dict(sys.modules, stub_modules):
        sys.modules.pop("helpers", None)
        return importlib.import_module("helpers")


class HelpersScheduleTests(unittest.TestCase):
    def setUp(self):
        self.helpers = load_helpers_module()

    def test_off_schedule_run_does_not_reuse_stale_month_end_bar(self):
        px = pd.DataFrame(
            {
                "QLD": [100.0, 101.0, 102.0],
                "QQQ": [200.0, 201.0, 202.0],
                "SMH": [300.0, 301.0, 302.0],
                "HYMB": [400.0, 401.0, 402.0],
                "GLDM": [500.0, 501.0, 502.0],
                "BIL": [600.0, 601.0, 602.0],
            },
            index=pd.to_datetime(["2026-06-26", "2026-06-29", "2026-06-30"]),
        )

        with patch.object(self.helpers, "_download_history_alpaca", return_value=px):
            with patch.object(
                self.helpers, "_current_market_date", return_value=pd.Timestamp("2026-07-14")
            ):
                weights, meta, _ = self.helpers.compute_today_target_weights_dual_mom_equity(
                    api=object(),
                    equity_cands=["QLD", "QQQ"],
                    other_sleeves=["SMH", "HYMB", "GLDM"],
                    cash="BIL",
                    ma_fixed={"SMH": 200, "HYMB": 50, "GLDM": 110},
                    force_rebalance=False,
                    persist_state=False,
                )

        self.assertIsNone(weights)
        self.assertEqual(meta["reason"], "not a rebalance day")
        self.assertEqual(meta["date"], "2026-07-14")
        self.assertEqual(meta["market_data_date"], "2026-06-30")
        self.assertFalse(meta["scheduled_rebalance_day"])

    def test_result_trade_today_requires_actual_scheduled_rebalance(self):
        self.assertFalse(
            self.helpers.result_trade_today(
                {
                    "meta": {
                        "reason": "not a rebalance day",
                        "date": "2026-07-14",
                        "market_data_date": "2026-06-30",
                        "scheduled_rebalance_day": False,
                    }
                }
            )
        )

        self.assertFalse(
            self.helpers.result_trade_today(
                {
                    "meta": {
                        "rebalance": True,
                        "date": "2026-07-31",
                        "market_data_date": "2026-07-30",
                        "scheduled_rebalance_day": True,
                    }
                }
            )
        )

        self.assertTrue(
            self.helpers.result_trade_today(
                {
                    "meta": {
                        "rebalance": True,
                        "date": "2026-07-31",
                        "market_data_date": "2026-07-31",
                        "scheduled_rebalance_day": True,
                    }
                }
            )
        )

    def test_monthly_rebalance_uses_actual_month_end_not_latest_loaded_bar(self):
        idx = pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-14"])

        class FakeCalendar:
            def schedule(self, start_date, end_date):
                return pd.DataFrame(
                    index=pd.to_datetime(
                        ["2026-07-01", "2026-07-02", "2026-07-14", "2026-07-31"]
                    )
                )

        with patch.object(self.helpers, "mcal") as fake_mcal:
            fake_mcal.get_calendar.return_value = FakeCalendar()
            self.assertFalse(
                self.helpers._is_rebalance_day(idx, pd.Timestamp("2026-07-14"), "M")
            )
            self.assertTrue(
                self.helpers._is_rebalance_day(idx, pd.Timestamp("2026-07-31"), "M")
            )

    def test_next_rebalance_day_for_monthly_schedule(self):
        class FakeCalendar:
            def schedule(self, start_date, end_date):
                year_month = (str(start_date)[:7], str(end_date)[:7])
                if year_month[0] == "2026-07":
                    idx = pd.to_datetime(["2026-07-31"])
                else:
                    idx = pd.to_datetime(["2026-08-31"])
                return pd.DataFrame(index=idx)

        with patch.object(self.helpers, "mcal") as fake_mcal:
            fake_mcal.get_calendar.return_value = FakeCalendar()
            self.assertEqual(
                self.helpers.next_rebalance_day(pd.Timestamp("2026-07-14"), "M"),
                pd.Timestamp("2026-07-31"),
            )
            self.assertEqual(
                self.helpers.next_rebalance_day(pd.Timestamp("2026-08-01"), "M"),
                pd.Timestamp("2026-08-31"),
            )


if __name__ == "__main__":
    unittest.main()
