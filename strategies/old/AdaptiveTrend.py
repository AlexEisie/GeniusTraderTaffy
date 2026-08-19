"""
AdaptiveTrend —— 论文趋势跟踪机制的 freqtrade 移植版
====================================================================
来源：arXiv 2602.11708
      "Systematic Trend-Following with Adaptive Portfolio Construction:
       Enhancing Risk-Adjusted Alpha in Cryptocurrency Markets" (2026-02)

论文原始规则（单资产部分，本文件忠实移植）：
    时间周期  6h
    动量      MOM_t = (P_t - P_{t-L}) / P_{t-L}
    入场      MOM > θ_entry 做多；MOM < -θ_entry 做空
    移动止损  S_t = max(S_{t-1}, P_t - α × ATR_t)，α = 2.5（论文最优区间 2.0~3.5）
    出场      价格跌破移动止损，除此之外没有任何止盈

--------------------------------------------------------------------
我丢掉了论文的哪些部分，以及为什么
--------------------------------------------------------------------
1. **每月网格搜索重优化 L 和 θ_entry** —— 丢弃。
   论文自己在 Limitations 里写了 "Monthly reoptimization introduces
   look-ahead bias risk"。它报告的 Sharpe 2.41 / 回撤 12.7% 极可能
   来自这个环节。这里把 L 和 θ 固定死，谁也别想偷看未来。
   **代价：实际表现一定比论文数字差很多。这是诚实的代价。**

2. **按市值取前 15 做多 + 上月 Sharpe ≥ 1.3 过滤** —— 丢弃。
   freqtrade 拿不到市值数据；"上月夏普高的才买"是典型的追逐近期
   表现，很容易在回测里制造假象。改用你自己的静态币对列表。

3. **70/30 多空配比** —— 丢弃。现货只能做多。
   文件里保留了做空逻辑，切到合约时把 can_short 改成 True 即可。

--------------------------------------------------------------------
一处需要你知道的解释性选择
--------------------------------------------------------------------
论文写的是 "Long entry when momentum exceeds threshold"，字面意思是
只要 MOM > θ 就持有。这样实现的话，止损出场后只要动量还在阈值上方
就会立刻重新入场，容易反复被扫。
这里改成**动量向上穿越阈值的那一根 K 线才入场**。想换回字面版本，
把 populate_entry_trend 里的 `& (mom.shift(1) <= theta)` 删掉即可。

--------------------------------------------------------------------
真实预期（先说清楚，免得白跑一轮）
--------------------------------------------------------------------
这类策略的合理量级是：年化 10~30%、夏普 0.5~1.0、最大回撤 20~40%，
胜率约 35~45%，收益高度集中在少数几笔大赢家上，中间会有连续
三四个月不赚钱的时期。**它不是"稳定正收益"，是"长期为正但过程难受"。**

因为胜率低、靠少数大赢家，**币对数量和仓位数是关键**：
建议 max_open_trades 提到 8~10，币对扩到 20~30 个。
仓位太少会系统性错过那几笔撑起全部收益的大行情。
====================================================================
"""

from datetime import datetime

import talib.abstract as ta
from pandas import DataFrame, isna

from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy


class AdaptiveTrend(IStrategy):
    INTERFACE_VERSION = 3

    # 论文用 6h。config 里不要设 timeframe，否则会覆盖这里
    timeframe = "6h"

    # 现货保持 False。切到永续合约时改成 True，做空逻辑会自动生效
    can_short = False

    # 论文没有止盈，只有移动止损。设成 10（1000%）等于关闭 ROI
    minimal_roi = {"0": 10}

    # 论文没有固定止损。freqtrade 强制要求一个硬上限，
    # 它同时充当 α×ATR 的封顶值（freqtrade 不允许止损比这个更宽）
    stoploss = -0.15
    use_custom_stoploss = True
    trailing_stop = False

    # 出场完全交给移动止损，不使用出场信号（忠实于论文）
    use_exit_signal = False
    exit_profit_only = False
    process_only_new_candles = True

    # 最长回看 60 根 + ATR(14) + 缓冲
    startup_candle_count = 120

    # ---------------- 论文的三个参数 ----------------
    # L：动量回看根数（6h × 28 = 7 天）
    mom_lookback = IntParameter(8, 60, default=28, space="buy", optimize=True)
    # θ_entry：动量阈值
    mom_threshold = DecimalParameter(0.01, 0.12, default=0.04, decimals=3, space="buy", optimize=True)
    # α：ATR 倍数，论文最优 2.5，区间 2.0~3.5
    atr_mult = DecimalParameter(1.5, 4.0, default=2.5, decimals=1, space="sell", optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        lookback = self.mom_lookback.value

        # MOM_t = (P_t - P_{t-L}) / P_{t-L}
        prev = dataframe["close"].shift(lookback)
        dataframe["mom"] = (dataframe["close"] - prev) / prev

        # ATR，论文用标准周期
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        theta = self.mom_threshold.value
        mom = dataframe["mom"]
        alive = (dataframe["volume"] > 0) & (~dataframe["atr_pct"].isna())

        # 做多：动量向上穿越 +θ
        dataframe.loc[
            (mom > theta) & (mom.shift(1) <= theta) & alive,
            ["enter_long", "enter_tag"],
        ] = (1, "mom_up")

        # 做空：动量向下穿越 -θ（can_short=False 时 freqtrade 会自动忽略）
        dataframe.loc[
            (mom < -theta) & (mom.shift(1) >= -theta) & alive,
            ["enter_short", "enter_tag"],
        ] = (1, "mom_down")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 论文没有出场信号，出场只由移动止损触发
        return dataframe

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
        论文的移动止损：S_t = max(S_{t-1}, P_t - α × ATR_t)

        freqtrade 的 adjust_stop_loss 只会把止损往有利方向移动、
        从不放宽，所以每根 K 线返回「相对当前价 α×ATR」就等价于
        论文里那个 max() 棘轮。返回值的正负号不影响结果。

        止损宽度的上限由 self.stoploss(-0.15) 自动封顶，
        这是 freqtrade 的硬性要求，不是论文的设定。
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None

        atr_pct = dataframe["atr_pct"].iloc[-1]
        if isna(atr_pct) or atr_pct <= 0:
            return None

        return -(atr_pct * self.atr_mult.value)

    plot_config = {
        "main_plot": {},
        "subplots": {
            "Momentum": {"mom": {"color": "blue"}},
            "ATR%": {"atr_pct": {"color": "purple"}},
        },
    }
