"""
PanicPounce2 — 波动率闸门恐慌接刀 v2（现货 / 只做多 / 1h / OKX）
====================================================================
v1 教训与 v2 修正（见 analysis/dip_regime.py 的事件研究，保守成交假设）：

  v1 败因：止盈 4.5% / 止损 8% —— 不对称方向反了。事件的收益分布
  右偏严重（暴跌后偶发 +10~20% 反弹），截掉右尾、保留左尾必亏。

  v2 结构（2026 年 79 个闸门事件，网格邻域稳健）：
    * 止盈 10%（骑住右尾）  * 止损 5%（砍掉左尾）  * 36h 强制离场
    * 单笔期望 +2.68%、胜率 ~60%（扣 0.2% 费后）
  波动率闸门：BTC 7 日年化波动率 >= 40% 才允许接刀。
    2026 年内：无闸门 +1.72%/笔 -> 40 闸门 +2.21% -> 50 闸门 +4.15%。
    取 40 保留足够频率；低波动平静期自动休眠。
  已证伪并排除的选项：
    * 放量下跌过滤（反效果，放量的刀更钝）
    * 挂深限价单等更低价（逆向选择，只有最烂的刀才成交）
    * 等反弹确认再进（错过均值回归的主要部分）

  事件在崩盘期高度聚集（2026-06 单月 54 次），max_open_trades
  与仓位划分天然限制单波暴露。
====================================================================
"""

import numpy as np
from pandas import DataFrame

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import DecimalParameter, IStrategy, merge_informative_pair


class PanicPounce2(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False

    # 右偏分布：高目标骑右尾；2160 分钟(36h)后无条件离场
    minimal_roi = {
        "0": 0.10,
        "2160": -1,
    }

    stoploss = -0.05
    use_custom_stoploss = False
    trailing_stop = False

    process_only_new_candles = True
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # BTC 波动率需要 168h + 缓冲
    startup_candle_count = 220

    # ---------------- 可调参数 ----------------
    dip_ret24 = DecimalParameter(-0.16, -0.07, default=-0.10, decimals=2, space="buy", optimize=True)
    # 全史重放中 gate>=50 每年为正 (2023 +11%, 2024 +67%, 2025 +13%, 2026 +18%),
    # gate 40-50 段的边际已在 2025-2026 衰减为负, 只保留极端段
    btc_vol_gate = DecimalParameter(40.0, 60.0, default=50.0, decimals=0, space="buy", optimize=True)

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            {
                # 48h 内 5 次止损 -> 暂停 12h（无底崩盘断路器）
                "method": "StoplossGuard",
                "lookback_period_candles": 48,
                "trade_limit": 5,
                "stop_duration_candles": 12,
                "only_per_pair": False,
            },
        ]

    def informative_pairs(self):
        return [("BTC/USDT", "1h")]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ret24"] = dataframe["close"].pct_change(24)
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lower"] = bb["lower"]

        # BTC 7 日年化波动率（政权闸门）
        btc = self.dp.get_pair_dataframe("BTC/USDT", "1h")
        btc = btc[["date", "close"]].copy()
        btc["btc_vol7"] = (
            btc["close"].pct_change().rolling(168, min_periods=100).std()
            * np.sqrt(24 * 365) * 100
        )
        dataframe = merge_informative_pair(
            dataframe, btc[["date", "btc_vol7"]], self.timeframe, "1h", ffill=True
        )
        dataframe.rename(columns={"btc_vol7_1h": "btc_vol7"}, inplace=True)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["ret24"] < self.dip_ret24.value)
            & (dataframe["close"] < dataframe["bb_lower"])
            & (dataframe["btc_vol7"] >= self.btc_vol_gate.value)
            & (dataframe["volume"] > 0),
            ["enter_long", "enter_tag"],
        ] = (1, "panic_dip_v2")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    plot_config = {
        "main_plot": {"bb_lower": {"color": "grey"}},
        "subplots": {
            "ret24": {"ret24": {"color": "purple"}},
            "btc_vol7": {"btc_vol7": {"color": "orange"}},
        },
    }
