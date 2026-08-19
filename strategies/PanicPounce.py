"""
PanicPounce — 恐慌抛售接刀（现货 / 只做多 / 1h / OKX）
====================================================================
设计依据（2026-02~08 OKX 1h 数据实证，见 analysis/edge_study2.py）：

  * 24h 跌幅 >10% 且收盘跌破布林下轨（20,2）时：
      前瞻 4h +1.02%，8h +1.01%，24h +1.48%（基线 ~0.00%），n=180。
  * 反直觉但数据明确：等"反弹确认"（阳线拐头）再买是负期望
    （-0.36% @4h）。刀要直接用限价单接住，不等确认。
  * 该形态在清算瀑布频发的 2026 熊尾市场中频率稳定（~1 次/天/28币）。

核心规则：
  入场：ret24h < -10% 且 close < 布林下轨。限价单直接挂当前买价。
  出场：时间衰减 ROI（4.5% -> 6h 后 2.5% -> 12h 后 1.2%），
        30h 强制离场（ROI=-1 技巧）；反弹冲上布林上轨直接止盈；
        断路器止损 -8%（平均 MAE 约 -4.5%，长尾更深）。
  风控：全局回撤保护 + 冷却期，防止在无底崩盘里连续接刀。
====================================================================
"""

import talib.abstract as ta
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import DecimalParameter, IStrategy


class PanicPounce(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False

    # 时间衰减止盈；1800 分钟（30h）后无条件离场
    minimal_roi = {
        "0": 0.045,
        "360": 0.025,
        "720": 0.012,
        "1800": -1,
    }

    stoploss = -0.08
    use_custom_stoploss = False
    trailing_stop = False

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count = 60

    # ---------------- 可调参数 ----------------
    dip_ret24 = DecimalParameter(-0.16, -0.07, default=-0.10, decimals=2, space="buy", optimize=True)

    @property
    def protections(self):
        return [
            # 同一币平仓后冷却 4h，防止同一波瀑布里反复进出
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            {
                # 组合 3 天内回撤超 10% -> 全部暂停 24h
                "method": "MaxDrawdown",
                "lookback_period_candles": 72,
                "trade_limit": 6,
                "stop_duration_candles": 24,
                "max_allowed_drawdown": 0.10,
            },
            {
                # 24h 内 3 次止损 -> 暂停 12h（崩盘无底时收手）
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 3,
                "stop_duration_candles": 12,
                "only_per_pair": False,
            },
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ret24"] = dataframe["close"].pct_change(24)
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lower"] = bb["lower"]
        dataframe["bb_upper"] = bb["upper"]
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["ret24"] < self.dip_ret24.value)
            & (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["volume"] > 0),
            ["enter_long", "enter_tag"],
        ] = (1, "panic_dip")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 直接反弹冲上布林上轨：拿钱走人
        dataframe.loc[
            (dataframe["close"] > dataframe["bb_upper"]) & (dataframe["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "bb_rip")
        return dataframe

    plot_config = {
        "main_plot": {
            "bb_lower": {"color": "grey"},
            "bb_upper": {"color": "grey"},
        },
        "subplots": {"ret24": {"ret24": {"color": "purple"}}},
    }
