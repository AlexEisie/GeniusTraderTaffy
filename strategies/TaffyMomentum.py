"""
TaffyMomentum — 截面相对动量轮动（现货 / 只做多 / 1h / OKX）
====================================================================
设计依据（2026-02~08 OKX 1h 数据实证，见 analysis/edge_study2.py）：

  * 7 日截面动量 top20% 币种的 48h 前瞻收益 +0.50%（top10% 达 +0.76%），
    显著高于 ~0.20% 的往返成本；中/尾部分组均为负。
  * 4 日尺度方差比 ~0.78：市场整体均值回归，时序突破无边际，
    但"相对强弱"横截面动量依然显著 —— 这是本策略唯一的 alpha 来源。
  * 2026-08 市场处于"选择性轮动"阶段（BTC 横盘、强势山寨轮涨），
    与该边际的结构吻合。

核心规则：
  入场：7d 动量截面排名 >= rank_enter(0.85) 且自身价格站上 EMA200，
        且非抛物线拉升（24h 涨幅 < 20%）、市场广度 >= 0.25。
  出场：排名跌破 rank_exit(0.55)（动量衰减）或 跌破 EMA200 且排名走弱；
        辅以 ATR 自适应移动止损让利润奔跑。
  风控：protections 处理连续止损与组合回撤；广度闸门挡住普跌崩盘。

排名的实现：populate_indicators 中通过 dataprovider 拉取整个白名单的
1h 收盘价，逐时间戳做横截面百分位排名。排名在 t 时刻只使用 <=t 的收盘，
无未来函数；类级缓存避免 O(n^2) 重复计算。
====================================================================
"""

from datetime import datetime

import talib.abstract as ta
from pandas import DataFrame, Series, isna
import pandas as pd

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    DecimalParameter,
    IStrategy,
    stoploss_from_open,
)


class TaffyMomentum(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False

    # 出场交给信号 + 移动止损，ROI 仅留极端兜底
    minimal_roi = {"0": 10}

    stoploss = -0.10
    use_custom_stoploss = True
    trailing_stop = False

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # 168h 动量 + EMA200，留缓冲
    startup_candle_count = 400

    # ---------------- 可调参数 ----------------
    rank_enter = DecimalParameter(0.75, 0.95, default=0.85, decimals=2, space="buy", optimize=True)
    rank_exit = DecimalParameter(0.40, 0.70, default=0.55, decimals=2, space="sell", optimize=True)
    breadth_min = DecimalParameter(0.10, 0.40, default=0.25, decimals=2, space="buy", optimize=True)
    max_ret24 = DecimalParameter(0.10, 0.30, default=0.20, decimals=2, space="buy", optimize=True)
    atr_stop_mult = DecimalParameter(1.5, 3.5, default=2.5, decimals=1, space="sell", optimize=True)
    atr_trail_mult = DecimalParameter(1.0, 3.0, default=2.0, decimals=1, space="sell", optimize=True)

    # ---------------- 保护 ----------------
    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 6},
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 168,
                "trade_limit": 8,
                "stop_duration_candles": 24,
                "max_allowed_drawdown": 0.12,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 48,
                "trade_limit": 4,
                "stop_duration_candles": 24,
                "only_per_pair": False,
            },
        ]

    # ---------------- 截面表缓存 ----------------
    # 值: (cache_key, DataFrame[date x pair] rank, Series[date] breadth)
    _xs_cache = None

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        return [(p, self.timeframe) for p in pairs]

    def _xs_tables(self):
        """跨币种 7d 动量百分位排名 + 市场广度。逐时间戳计算，无未来数据。"""
        pairs = tuple(sorted(self.dp.current_whitelist()))
        closes = {}
        last_dt = None
        for p in pairs:
            df = self.dp.get_pair_dataframe(p, self.timeframe)
            if df is None or df.empty:
                continue
            s = df.set_index("date")["close"]
            closes[p] = s
            if last_dt is None or s.index[-1] > last_dt:
                last_dt = s.index[-1]

        key = (pairs, last_dt)
        if self._xs_cache is not None and self._xs_cache[0] == key:
            return self._xs_cache[1], self._xs_cache[2]

        cm = pd.DataFrame(closes)
        mom = cm.pct_change(168, fill_method=None)
        rank = mom.rank(axis=1, pct=True)
        ema200 = cm.ewm(span=200, min_periods=100).mean()
        above = (cm > ema200) & cm.notna()
        breadth = above.sum(axis=1) / cm.notna().sum(axis=1).clip(lower=1)
        type(self)._xs_cache = (key, rank, breadth)
        return rank, breadth

    # ---------------- 指标 ----------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["ret24"] = dataframe["close"].pct_change(24)

        rank, breadth = self._xs_tables()
        pair = metadata["pair"]
        if pair in rank.columns:
            dataframe["xs_rank"] = dataframe["date"].map(rank[pair])
        else:
            dataframe["xs_rank"] = float("nan")
        dataframe["breadth"] = dataframe["date"].map(breadth)
        return dataframe

    # ---------------- 入场 ----------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["xs_rank"] >= self.rank_enter.value)
            & (dataframe["close"] > dataframe["ema200"])
            & (dataframe["ret24"] < self.max_ret24.value)
            & (dataframe["breadth"] >= self.breadth_min.value)
            & (dataframe["volume"] > 0),
            ["enter_long", "enter_tag"],
        ] = (1, "xs_mom")
        return dataframe

    # ---------------- 出场 ----------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["xs_rank"] < self.rank_exit.value) & (dataframe["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "mom_decay")

        dataframe.loc[
            (dataframe["close"] < dataframe["ema200"])
            & (dataframe["xs_rank"] < 0.75)
            & (dataframe["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "trend_break")
        return dataframe

    # ---------------- ATR 自适应止损 ----------------
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None
        atr_pct = dataframe["atr_pct"].iloc[-1]
        if isna(atr_pct) or atr_pct <= 0:
            return None

        if current_profit < 0.05:
            # 初始阶段：相对开仓价的 ATR 止损，夹在 4%~10%
            initial = min(max(atr_pct * self.atr_stop_mult.value, 0.04), 0.10)
            return stoploss_from_open(
                -initial, current_profit, is_short=trade.is_short, leverage=trade.leverage
            )
        # 盈利 5% 之后：相对现价的 ATR 移动止损，夹在 2.5%~6%
        trail = min(max(atr_pct * self.atr_trail_mult.value, 0.025), 0.06)
        return -trail

    plot_config = {
        "main_plot": {"ema200": {"color": "red"}},
        "subplots": {
            "rank": {"xs_rank": {"color": "blue"}},
            "breadth": {"breadth": {"color": "green"}},
        },
    }
