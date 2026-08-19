"""
PullbackTrend —— 顺势回调买入策略
====================================================================
适用：现货 / 只做多 / 15m / OKX

设计出发点（针对模板策略「86% 胜率却亏钱」的问题）：

1. 亏损来自盈亏比，不是胜率。模板是「赚 1% 就跑、亏 10% 才砍」，
   所以这里把顺序反过来先定出场：ATR 自适应止损 + 盈利后移动止损，
   让单笔亏损被压在 3~7%，而盈利单允许跑到 10%+。

2. 只做多的现货，最大的风险是「在下跌趋势里不断抄底」。
   所以加了 1h 级别的趋势过滤：只有 1h EMA50 > EMA200 且价格在
   EMA200 上方时才允许开仓。震荡下行市里这个策略会几乎不交易，
   这是刻意的——不交易也是一种正确的决策。

3. 「稳定」靠的是 protections 而不是参数。连续止损、回撤超限、
   某个币对持续亏损时，自动停一段时间，避免在不利环境里连续放血。

注意：下面所有参数都是「起始假设」，不是已验证的最优值。
必须在你自己的币对和时间段上回测验证后再使用。
====================================================================
"""

from datetime import datetime

import talib.abstract as ta
from pandas import DataFrame, isna

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    DecimalParameter,
    IntParameter,
    IStrategy,
    informative,
    stoploss_from_open,
)


class PullbackTrend(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short = False

    # ---------------- 出场结构（策略的核心） ----------------
    # ROI 只作为兜底，主要出场交给 ATR 移动止损，让盈利单能跑
    minimal_roi = {
        "0": 0.10,      # 开仓即 10% 目标（基本不会立刻触发，等于放开上限）
        "240": 0.05,    # 4 小时后降到 5%
        "720": 0.025,   # 12 小时后 2.5%
        "1440": 0.012,  # 24 小时后 1.2%
        "2880": 0,      # 48 小时后有利润就走，避免长期占用仓位
    }

    # 硬止损上限。custom_stoploss 算出的值不会超过这个
    stoploss = -0.08
    use_custom_stoploss = True
    trailing_stop = False  # 用 custom_stoploss 代替，粒度更细

    # ---------------- 运行参数 ----------------
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # 1h EMA200 需要 200 根 1h = 800 根 15m，再留 100 根缓冲
    # OKX 单次返回 300 根、允许 5 次调用，上限 1499，这里安全
    startup_candle_count = 900

    # ---------------- 可调参数（将来 hyperopt 直接可用） ----------------
    buy_rsi_max = IntParameter(28, 50, default=42, space="buy", optimize=True)
    atr_stop_mult = DecimalParameter(1.5, 3.5, default=2.2, decimals=1, space="sell", optimize=True)
    atr_trail_mult = DecimalParameter(0.8, 2.0, default=1.3, decimals=1, space="sell", optimize=True)
    exit_rsi = IntParameter(65, 85, default=76, space="sell", optimize=True)

    # ---------------- 保护机制（「稳定」主要靠这一段） ----------------
    @property
    def protections(self):
        return [
            {
                # 每次平仓后该币对冷却 4 根 K 线，避免同一波行情里反复进出
                "method": "CooldownPeriod",
                "stop_duration_candles": 4,
            },
            {
                # 24 小时内全局止损 3 次 → 全部停 12 小时
                # 这是防「环境突变时连续放血」的主闸门
                "method": "StoplossGuard",
                "lookback_period_candles": 96,
                "trade_limit": 3,
                "stop_duration_candles": 48,
                "only_per_pair": False,
            },
            {
                # 48 小时内回撤超过 10% → 停 12 小时
                "method": "MaxDrawdown",
                "lookback_period_candles": 192,
                "trade_limit": 10,
                "stop_duration_candles": 48,
                "max_allowed_drawdown": 0.10,
            },
            {
                # 某个币对 3 天内累计亏损超过 2% → 单独停它 12 小时
                "method": "LowProfitPairs",
                "lookback_period_candles": 288,
                "trade_limit": 2,
                "stop_duration_candles": 48,
                "required_profit": -0.02,
                "only_per_pair": True,
            },
        ]

    # ---------------- 指标 ----------------
    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """1h 级别只做一件事：判断现在是不是可以做多的环境"""
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)

        # ATR 用来做「按当前波动率缩放」的止损，而不是固定百分比
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]

        bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lower"] = bb["lower"]
        dataframe["bb_mid"] = bb["mid"]

        # 成交量地板，过滤掉深夜的流动性真空
        dataframe["volume_mean"] = dataframe["volume"].rolling(96).mean()

        return dataframe

    # ---------------- 入场 ----------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 环境过滤：1h 必须是多头结构，否则一律不开仓
        uptrend = (dataframe["ema50_1h"] > dataframe["ema200_1h"]) & (
            dataframe["close_1h"] > dataframe["ema200_1h"]
        )
        liquid = (dataframe["volume"] > dataframe["volume_mean"] * 0.5) & (dataframe["volume"] > 0)

        # A. 假跌破布林下轨后收回 —— 典型的「洗盘结束」形态
        dataframe.loc[
            uptrend
            & liquid
            & (dataframe["close"].shift(1) < dataframe["bb_lower"].shift(1))
            & (dataframe["close"] > dataframe["bb_lower"])
            & (dataframe["rsi"] < self.buy_rsi_max.value),
            ["enter_long", "enter_tag"],
        ] = (1, "bb_reclaim")

        # B. RSI 超卖拐头 + 仍站在 15m EMA50 上方 —— 浅回调
        dataframe.loc[
            uptrend
            & liquid
            & (dataframe["rsi"] < self.buy_rsi_max.value)
            & (dataframe["rsi"] > dataframe["rsi"].shift(1))
            & (dataframe["rsi"].shift(1) <= dataframe["rsi"].shift(2))
            & (dataframe["close"] > dataframe["ema50"]),
            ["enter_long", "enter_tag"],
        ] = (1, "rsi_turn")

        return dataframe

    # ---------------- 出场信号 ----------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 超买离场
        dataframe.loc[
            (dataframe["rsi"] > self.exit_rsi.value) & (dataframe["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "rsi_overbought")

        # 大环境破位：1h 转空头结构，不管盈亏先撤
        dataframe.loc[
            (dataframe["ema50_1h"] < dataframe["ema200_1h"]) & (dataframe["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "regime_break")

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
        """
        两段式：
          未盈利阶段 —— 相对开仓价的固定止损，宽度 = ATR × 倍数，夹在 3%~7%
          盈利 2% 后 —— 转为相对当前价的移动止损，宽度随波动率收缩
        返回值的正负号不影响结果，freqtrade 内部取绝对值。
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None

        atr_pct = dataframe["atr_pct"].iloc[-1]
        if isna(atr_pct) or atr_pct <= 0:
            return None

        if current_profit < 0.02:
            # 初始止损：波动大的币给更宽的空间，避免被正常噪音扫掉
            initial = min(max(atr_pct * self.atr_stop_mult.value, 0.03), 0.07)
            return stoploss_from_open(
                -initial, current_profit, is_short=trade.is_short, leverage=trade.leverage
            )

        # 盈利后移动止损，盈利越多收得越紧
        trail = min(max(atr_pct * self.atr_trail_mult.value, 0.010), 0.05)
        if current_profit > 0.05:
            trail = max(trail * 0.7, 0.008)
        return -trail

    # ---------------- FreqUI 绘图 ----------------
    plot_config = {
        "main_plot": {
            "ema50": {"color": "orange"},
            "bb_lower": {"color": "grey"},
            "bb_mid": {"color": "grey"},
            "ema200_1h": {"color": "red"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "blue"}},
            "ATR%": {"atr_pct": {"color": "purple"}},
        },
    }
