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
    fake_helpers.print_orders_table = lambda result: ""
    fake_helpers.print_weights_table = lambda result: ""
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


if __name__ == "__main__":
    unittest.main()
