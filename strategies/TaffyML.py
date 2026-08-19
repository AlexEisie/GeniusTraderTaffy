"""
TaffyML — FreqAI/LightGBM 短线多头（现货 / 1h / OKX）
====================================================================
思路：2026 市场里所有单因子策略（趋势/突破/截面动量/恐慌接刀）
均已衰减为负；仅剩的可行路线之一是用步进式重训练的 ML 模型把
一批"弱信号"做非线性组合，随市场漂移每周自适应。

  * 标签：未来 12h 收益（回归）。
  * 特征：多周期收益率 / RSI / BB 位置 / ATR% / 量能，
    自动扩展到 1h+4h 两个时间框架与 BTC/ETH 关联对。
  * 训练：滚动 90 天窗口，每 7 天重训一次（walk-forward，无未来函数）。
  * 入场：预测 12h 收益 > 0.6%（≈3 倍往返成本）；
  * 出场：预测转负 / ROI 10% / 硬止损 -5% / 48h 强制离场。

风险声明：ML 策略的回测即 walk-forward 模拟，但特征/阈值的选择
仍然经过了本次研究的迭代，存在研究者自由度偏差；结论以
样本外 dry-run 为准。
====================================================================
"""

import numpy as np
import talib.abstract as ta
from pandas import DataFrame

from technical import qtpylib

from freqtrade.strategy import IStrategy


class TaffyML(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False

    minimal_roi = {"0": 0.10, "2880": -1}
    # v3a: v2 的出场分解显示 ml_flip 累计 +31%、stop_loss 累计 -65% ——
    # 模型有方向信号但 -5% 止损与 12h 预测尺度错配, 放宽到 -10% 验证
    stoploss = -0.10
    trailing_stop = False
    use_custom_stoploss = False

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    startup_candle_count = 200

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 2},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 48,
                "trade_limit": 5,
                "stop_duration_candles": 12,
                "only_per_pair": False,
            },
        ]

    # ---------------- FreqAI 特征工程 ----------------
    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-er-period"] = dataframe["close"].pct_change(period)
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=period, stds=2)
        bb_width = (bb["upper"] - bb["lower"]) / bb["mid"]
        dataframe["%-bb_width-period"] = bb_width
        dataframe["%-bb_pos-period"] = (dataframe["close"] - bb["lower"]) / (bb["upper"] - bb["lower"] + 1e-12)
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period) / dataframe["close"]
        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-vol_z"] = (
            (dataframe["volume"] - dataframe["volume"].rolling(96).mean())
            / (dataframe["volume"].rolling(96).std() + 1e-12)
        )
        for h in [4, 24, 72, 168]:
            dataframe[f"%-ret{h}"] = dataframe["close"].pct_change(h)
        dataframe["%-vol24"] = dataframe["close"].pct_change().rolling(24).std()
        dataframe["%-vol168"] = dataframe["close"].pct_change().rolling(168).std()
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour / 23
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek / 6
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        label_period = self.freqai_info["feature_parameters"]["label_period_candles"]
        dataframe["&-s_ret"] = (
            dataframe["close"].shift(-label_period) / dataframe["close"] - 1
        )
        return dataframe

    # ---------------- 指标 / 信号 ----------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        # v2: 非特征列的趋势闸门 —— v1 在 2026-06 崩盘月拿了整月市场β (-20.4%),
        # 训练窗外推失败时至少不逆势接多
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        enter = (
            (dataframe["do_predict"] == 1)
            & (dataframe["&-s_ret"] > 0.010)
            & (dataframe["close"] > dataframe["ema200"])
            & (dataframe["volume"] > 0)
        )
        dataframe.loc[enter, ["enter_long", "enter_tag"]] = (1, "ml_long")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_ = (
            (dataframe["do_predict"] == 1)
            & (dataframe["&-s_ret"] < -0.002)
        )
        dataframe.loc[exit_, ["exit_long", "exit_tag"]] = (1, "ml_flip")
        return dataframe
