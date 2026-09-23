#!/usr/bin/env python3
"""Offline smoke test for the PROMETHEUS trading bot.

Exercises the real computational core (technical indicators + strategy
signals) on synthetic OHLCV data. It performs NO network calls and NO
trading, so it is safe to run in CI / dev environments to verify that the
Python environment and dependencies (ccxt, pandas, numpy) are working.
"""
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_bot_module():
    """Load the extension-less `onepiece` script as a module."""
    path = REPO_ROOT / "onepiece"
    return SourceFileLoader("onepiece", str(path)).load_module()


def make_synthetic_ohlcv(rows: int = 300, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic OHLCV candles (a random-walk price)."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, rows)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.003, rows)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, rows)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.uniform(1_000, 10_000, rows)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def main() -> int:
    print("=" * 70)
    print("PROMETHEUS offline smoke test")
    print("=" * 70)
    print(f"numpy  {np.__version__}")
    print(f"pandas {pd.__version__}")
    import ccxt

    print(f"ccxt   {ccxt.__version__}")

    bot = load_bot_module()
    df = make_synthetic_ohlcv()
    print(f"\nGenerated {len(df)} synthetic candles "
          f"(last close={df['close'].iloc[-1]:.2f}).")

    indicators = bot.TechnicalIndicators.calculate_all(df)
    if not indicators:
        print("FAIL: no indicators computed")
        return 1
    print(f"Computed {len(indicators)} indicators. Sample:")
    for key in ("rsi_14", "macd", "ema_21", "bb_width", "atr_14", "adx"):
        if key in indicators:
            print(f"  {key:>10} = {indicators[key]:.6f}")

    cfg = bot.Config()
    logger = bot.Logger(str(REPO_ROOT / "smoke_test_log.txt"))
    strategy_classes = [
        bot.RSIOversoldOverbought,
        bot.MACDCrossover,
        bot.BollingerBandSqueeze,
        bot.EMACrossover,
        bot.ADXTrendStrength,
    ]
    print(f"\nRunning {len(strategy_classes)} strategies on the data:")
    ok = 0
    for cls in strategy_classes:
        strat = cls(cfg, logger)
        result = strat.get_signal(df, indicators, {})
        signal = result.get("signal")
        strength = result.get("strength", 0)
        assert signal in ("long", "short", "neutral"), result
        assert "error" not in result, result
        print(f"  {strat.name:>16}: {signal:<8} (strength={strength:.3f})")
        ok += 1

    print("\n" + "=" * 70)
    print(f"SUCCESS: environment works. {len(indicators)} indicators, "
          f"{ok}/{len(strategy_classes)} strategies produced valid signals.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
