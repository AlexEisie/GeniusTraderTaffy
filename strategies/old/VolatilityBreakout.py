"""
VolatilityBreakout —— Larry Williams 波动率突破策略的 freqtrade 移植版
====================================================================
来源：Larry Williams 的 Volatility Break-out（《短线交易秘诀》）
      加密圈流行的版本来自韩国社区（"변동성 돌파 전략"），
      参考实现 https://github.com/sharebook-kr/larry_simple

原始规则（完全外部定义，本文件只做移植，不做原创改动）：
    range  = 前一日最高价 - 前一日最低价
    target = 当日开盘价 + K × range          （K 通常取 0.5）
    入场   = 当日价格向上突破 target 时买入
    出场   = 当日收盘（≈次日开盘）无条件平仓
    过滤   = 当日开盘价需在 5 日均线上方（噪音过滤，社区标准加法）

移植时做的三处工程性调整（都已尽量保持中性，不改变策略性质）：
  1. 执行粒度用 15m K 线判断突破。原策略在实盘中是 tick 级触发，
     15m 会让实际成交价略高于 target，这是真实的执行成本，没有美化。
  2. "前一日高低" 通过 1d informative 取得。freqtrade 的
     merge_informative_pair 会自动做时间对齐，不存在未来函数。
     "当日开盘价" 由 15m 数据自行分组计算，同样只用已知信息。
  3. 原策略没有止损（当日平仓就是风控）。freqtrade 必须设 stoploss，
     这里设 -0.08 作为断路器。**收紧这个值会改变策略性质**，
     想忠实复现就别动它。

注意：这个策略在 2017-2021 年的加密市场表现很好，此后边缘明显衰减。
必须自己在近期数据上验证，不要采信任何旧的收益截图。
====================================================================
"""

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IStrategy, informative


class VolatilityBreakout(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short = False

    # 原策略靠「当日平仓」控制风险，不用 ROI 止盈。
    # 设成 10（1000%）等于关闭 ROI 出场。
    minimal_roi = {"0": 10}

    # 原策略无止损，这里仅作断路器
    stoploss = -0.08
    trailing_stop = False
    use_custom_stoploss = False

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # 需要 5 根日线做均线过滤：5 × 96 根 15m = 480，留足缓冲
    startup_candle_count = 600

    # ---------------- 唯一的核心参数 ----------------
    # K 越小越容易触发（交易更多、噪音更多），越大越挑剔
    k_value = DecimalParameter(0.2, 0.9, default=0.5, decimals=2, space="buy", optimize=True)

    @property
    def protections(self):
        return [
            # 平仓后冷却 2 小时，避免同一天内被反复扫进扫出
            {"method": "CooldownPeriod", "stop_duration_candles": 8},
        ]

    # ---------------- 日线指标：前一日高低 + 5 日均线 ----------------
    @informative("1d")
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 这里的 high / low / close 都是「已收盘的日线」，
        # merge 时 freqtrade 会自动错位对齐，取到的就是前一日的值
        dataframe["ma5"] = ta.SMA(dataframe, timeperiod=5)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 前一日振幅
        dataframe["prev_range"] = dataframe["high_1d"] - dataframe["low_1d"]

        # 当日开盘价：按 UTC 自然日分组取第一根 15m 的 open
        # transform("first") 只回填组内第一个值，不涉及未来数据
        day = dataframe["date"].dt.floor("1D")
        dataframe["day_start"] = day
        dataframe["day_open"] = dataframe.groupby(day)["open"].transform("first")

        # 突破目标价
        dataframe["target"] = dataframe["day_open"] + dataframe["prev_range"] * self.k_value.value

        return dataframe

    # ---------------- 入场：当日首次向上突破 target ----------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        crossed = (
            (dataframe["close"] > dataframe["target"])
            & (dataframe["close"].shift(1) <= dataframe["target"].shift(1))
            & (dataframe["prev_range"] > 0)
            & (dataframe["volume"] > 0)
        )

        # 噪音过滤：当日开盘价站在 5 日均线上方才允许做多
        trend_ok = dataframe["day_open"] > dataframe["ma5_1d"]

        # 每天只取第一次突破，避免当日反复触发
        first_of_day = crossed.groupby(dataframe["day_start"]).cumsum() == 1

        dataframe.loc[
            crossed & trend_ok & first_of_day,
            ["enter_long", "enter_tag"],
        ] = (1, "vb_breakout")

        return dataframe

    # ---------------- 出场：当日最后一根 15m K 线无条件平仓 ----------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["date"].dt.hour == 23)
            & (dataframe["date"].dt.minute >= 45)
            & (dataframe["volume"] > 0),
            ["exit_long", "exit_tag"],
        ] = (1, "day_close")

        return dataframe

    plot_config = {
        "main_plot": {
            "target": {"color": "red"},
            "day_open": {"color": "grey"},
            "ma5_1d": {"color": "orange"},
        },
    }
