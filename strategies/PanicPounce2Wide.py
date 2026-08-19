"""
PanicPounce2Wide — PanicPounce2 的宽出场变体 (15% / -8% / 72h)
====================================================================
gate50 子样本出场网格 (analysis/gate50_grid.py) 显示宽出场在每一年
的事件均值都占优 (全史组合重放 +228% vs 窄出场 +146%)；
但 2026 年段窄出场更强 (+17.5% vs +6.7%)。
本文件用于 freqtrade 系统级 A/B 终审, 与 PanicPounce2 仅三处不同:
  minimal_roi 0.10->0.15, stoploss -0.05->-0.08, 强平 36h->72h。
====================================================================
"""

from PanicPounce2 import PanicPounce2


class PanicPounce2Wide(PanicPounce2):
    minimal_roi = {
        "0": 0.15,
        "4320": -1,
    }
    stoploss = -0.08
