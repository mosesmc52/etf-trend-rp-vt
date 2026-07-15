import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch


def load_algo_module():
    fake_alpaca_adapter = types.ModuleType("alpaca_adapter")
    fake_alpaca_adapter.AlpacaAPI = types.SimpleNamespace(from_env=None)

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.find_dotenv = lambda: ""
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None

    fake_helpers = types.ModuleType("helpers")
    fake_helpers.export_strategy_json = lambda *args, **kwargs: None
    fake_helpers.getenv_float = lambda name, default: float(
        os.getenv(name, str(default))
    )
    fake_helpers.REB = "M"
    fake_helpers.next_rebalance_day = lambda today, reb_rule: types.SimpleNamespace(
        date=lambda: "2026-07-31"
    )
    fake_helpers.print_orders_table = lambda result: ""
    fake_helpers.print_weights_table = lambda result: ""
    fake_helpers.result_trade_today = lambda result: bool(
        ((result or {}).get("meta", {}) or {}).get("scheduled_rebalance_day", False)
    )
    fake_helpers.run_single_iteration = lambda *args, **kwargs: {}
    fake_helpers.str2bool = lambda value: str(value).strip().lower() in {
        "true",
        "t",
        "1",
        "on",
    }
    fake_helpers.upload_file_to_digitalocean_spaces = lambda *args, **kwargs: None

    fake_log = types.ModuleType("log")
    fake_log.log = lambda *args, **kwargs: None

    fake_ses = types.ModuleType("SES")
    fake_ses.AmazonSES = object

    stub_modules = {
        "alpaca_adapter": fake_alpaca_adapter,
        "dotenv": fake_dotenv,
        "helpers": fake_helpers,
        "log": fake_log,
        "SES": fake_ses,
    }

    with patch.dict(sys.modules, stub_modules):
        sys.modules.pop("algo", None)
        return importlib.import_module("algo")


class AlgoObserveModeTests(unittest.TestCase):
    def setUp(self):
        self.algo = load_algo_module()
        self.env = {
            "APP_STATE": "OBSERVE",
            "FORCED_REBALANCE": "false",
            "SYNC_STRATEGY_JSON_TO_SPACES": "false",
            "EMAIL_POSITIONS": "false",
            "USE_DYNAMIC_VT": "false",
            "EQUITY_FRACTION": "1",
            "ALPACA_PAPER": "true",
            "ALPACA_KEY_ID": "key",
            "ALPACA_SECRET_KEY": "secret",
        }

    def _run_main(self, forced_rebalance: str):
        env = dict(self.env)
        env["FORCED_REBALANCE"] = forced_rebalance

        with patch.dict(os.environ, env, clear=True):
            with patch.object(self.algo, "load_dotenv"), patch.object(
                self.algo, "find_dotenv", return_value=""
            ):
                with patch.object(self.algo.AlpacaAPI, "from_env") as from_env:
                    with patch.object(
                        self.algo, "run_single_iteration"
                    ) as run_single_iteration:
                        with patch.object(
                            self.algo,
                            "print_weights_table",
                            return_value="weights output",
                        ):
                            with patch("builtins.print"):
                                from_env.return_value.get_account.return_value = (
                                    types.SimpleNamespace(equity="1000")
                                )
                                run_single_iteration.return_value = {
                                    "meta": {"scheduled_rebalance_day": False},
                                    "weights": {},
                                    "orders": [],
                                }
                                self.algo.main()
                                return run_single_iteration.call_args.kwargs

    def test_observe_mode_respects_schedule_without_force_rebalance(self):
        kwargs = self._run_main("false")
        self.assertFalse(kwargs["force_rebalance"])
        self.assertFalse(kwargs["persist_state"])
        self.assertFalse(kwargs["is_live_trade"])

    def test_observe_mode_can_still_force_rebalance_explicitly(self):
        kwargs = self._run_main("true")
        self.assertTrue(kwargs["force_rebalance"])
        self.assertFalse(kwargs["persist_state"])
        self.assertFalse(kwargs["is_live_trade"])

    def test_app_state_is_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "APP_STATE must be set"):
                self.algo.get_app_state()

    def test_observe_email_reports_trade_today_status(self):
        env = dict(self.env)
        env["EMAIL_POSITIONS"] = "true"
        env["TO_ADDRESSES"] = "test@example.com"
        env["FROM_ADDRESS"] = "from@example.com"

        with patch.dict(os.environ, env, clear=True):
            with patch.object(self.algo, "load_dotenv"), patch.object(
                self.algo, "find_dotenv", return_value=""
            ):
                with patch.object(self.algo.AlpacaAPI, "from_env") as from_env:
                    with patch.object(
                        self.algo, "run_single_iteration"
                    ) as run_single_iteration:
                        with patch.object(
                            self.algo,
                            "print_weights_table",
                            return_value="weights output",
                        ):
                            with patch.object(self.algo, "AmazonSES") as amazon_ses:
                                with patch("builtins.print"):
                                    from_env.return_value.get_account.return_value = (
                                        types.SimpleNamespace(equity="1000")
                                    )
                                    run_single_iteration.return_value = {
                                        "meta": {
                                            "scheduled_rebalance_day": True,
                                            "rebalance": True,
                                            "date": "2026-07-31",
                                            "market_data_date": "2026-07-31",
                                        },
                                        "weights": {
                                            "QLD": 0.20624684560933468,
                                            "HYMB": 0.6000000000000001,
                                            "SMH": 0.1937531543906653,
                                        },
                                        "orders": [],
                                    }
                                    self.algo.main()

        kwargs = amazon_ses.return_value.send_html_email.call_args.kwargs
        self.assertIn("Trade Today:</strong> Yes", kwargs["content"])
        self.assertIn("Next Rebalance Day:</strong> 2026-07-31", kwargs["content"])
        self.assertIn("Portfolio Allocations:</strong>", kwargs["content"])
        self.assertIn("HYMB: 60.00%", kwargs["content"])
        self.assertNotIn("Diagnostics:", kwargs["content"])


if __name__ == "__main__":
    unittest.main()
