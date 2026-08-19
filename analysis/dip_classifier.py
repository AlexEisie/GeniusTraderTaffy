"""v3b: LightGBM classifier on panic-dip events — can ML beat the hand-made vol>=50 gate?

Walk-forward: train<=2024 -> test 2025; train<=2025 -> test 2026. Label = deployed-exit
trade outcome (tgt10/stop5/36h, conservative fills). If model top-half doesn't beat the
gate50 rule out-of-sample in both folds, the ML branch closes with a documented negative.
"""
import glob
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
DATADIR = "/freqtrade/user_data/data/okx"
COST = 0.002
MAJORS = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "TRX", "SUI", "WLD", "OKB", "LTC",
    "LINK", "AVAX", "ADA", "DOT", "BCH", "ATOM", "FIL", "NEAR", "APT", "ARB",
    "OP", "INJ", "TIA", "AAVE", "UNI", "HYPE", "PUMP",
]

data = {}
for f in sorted(glob.glob(f"{DATADIR}/*-1h.feather")):
    base = f.split("/")[-1].replace("-1h.feather", "").split("_")[0]
    if base in MAJORS:
        data[base] = pd.read_feather(f).set_index("date").sort_index()

btc = data["BTC"].close
btc_vol7 = btc.pct_change().rolling(168, min_periods=100).std() * np.sqrt(24 * 365) * 100
btc_ret24 = btc.pct_change(24)
btc_dd30 = btc / btc.rolling(24 * 30).max() - 1

rows = []
for p, d in data.items():
    c = d.close
    r1 = c.pct_change()
    r4 = c.pct_change(4)
    r24 = c.pct_change(24)
    r168 = c.pct_change(168)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    bbz = (c - mid) / (2 * sd + 1e-12)
    rsi_up = r1.clip(lower=0).rolling(14).mean()
    rsi_dn = (-r1.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + rsi_up / (rsi_dn + 1e-12))
    ownvol = r1.rolling(168, min_periods=100).std() * np.sqrt(24 * 365) * 100
    volr = d.volume / d.volume.rolling(168).mean()
    lower = mid - 2 * sd
    sig = (r24 < -0.10) & (c < lower)
    idxs = np.where(sig.values)[0]
    last_i = -999
    for i in idxs:
        if i - last_i < 4:
            continue
        last_i = i
        if i + 38 >= len(d) or i < 200:
            continue
        t = d.index[i]
        entry = d.open.values[i + 1]
        hi = d.high.values[i + 1 : i + 38]
        lo = d.low.values[i + 1 : i + 38]
        cl = d.close.values[i + 1 : i + 38]
        pnl = None
        for k in range(36):
            if lo[k] <= entry * 0.95:
                pnl = -0.05
                break
            if hi[k] >= entry * 1.10:
                pnl = 0.10
                break
        if pnl is None:
            pnl = cl[35] / entry - 1
        pnl -= COST
        rows.append({
            "t": t, "year": t.year, "pair": p, "pnl": pnl, "win": int(pnl > 0),
            "btcvol": btc_vol7.asof(t), "ownvol": ownvol.values[i],
            "ret4": r4.values[i], "ret24": r24.values[i], "ret168": r168.values[i],
            "bbz": bbz.values[i], "rsi": rsi.values[i], "volr": volr.values[i],
            "btc_ret24": btc_ret24.asof(t), "btc_dd30": btc_dd30.asof(t),
            "hour": t.hour, "dow": t.dayofweek,
        })

ev = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
# causal wave count: events in trailing 12h
ev["cnt12"] = [((ev.t > t - pd.Timedelta(hours=12)) & (ev.t <= t)).sum() for t in ev.t]
ev = ev.dropna()
print(f"events: {len(ev)}  by year: {ev.groupby('year').size().to_dict()}")
print(f"overall: avg pnl {ev.pnl.mean()*100:+.2f}%  win {ev.win.mean()*100:.1f}%")

FEATS = ["btcvol", "ownvol", "ret4", "ret24", "ret168", "bbz", "rsi", "volr",
         "btc_ret24", "btc_dd30", "hour", "dow", "cnt12"]

import lightgbm as lgb
from sklearn.metrics import roc_auc_score


def fold(train_end, test_year):
    tr = ev[ev.t < pd.Timestamp(train_end, tz="UTC")]
    te = ev[ev.year == test_year]
    if len(tr) < 150 or len(te) < 30:
        print(f"fold {test_year}: insufficient data ({len(tr)}/{len(te)})")
        return
    m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=15,
                           max_depth=5, min_child_samples=25, subsample=0.9,
                           colsample_bytree=0.8, n_jobs=1, verbosity=-1)
    m.fit(tr[FEATS], tr.win)
    score = m.predict_proba(te[FEATS])[:, 1]
    te = te.assign(score=score)
    auc = roc_auc_score(te.win, score) if te.win.nunique() > 1 else float("nan")
    med = te.score.median()
    top = te[te.score >= med]
    bot = te[te.score < med]
    q70 = te.score.quantile(0.7)
    top30 = te[te.score >= q70]
    gate = te[te.btcvol >= 50]
    print(f"\n=== fold: train<{train_end} ({len(tr)} ev) -> test {test_year} ({len(te)} ev) ===")
    print(f"AUC {auc:.3f}")
    print(f"all events:      avg {te.pnl.mean()*100:+.2f}%  win {te.win.mean()*100:.0f}%  n={len(te)}")
    print(f"model top-50%:   avg {top.pnl.mean()*100:+.2f}%  win {top.win.mean()*100:.0f}%  n={len(top)}")
    print(f"model bottom50%: avg {bot.pnl.mean()*100:+.2f}%  win {bot.win.mean()*100:.0f}%  n={len(bot)}")
    print(f"model top-30%:   avg {top30.pnl.mean()*100:+.2f}%  win {top30.win.mean()*100:.0f}%  n={len(top30)}")
    print(f"gate50 rule:     avg {gate.pnl.mean()*100:+.2f}%  win {gate.win.mean()*100:.0f}%  n={len(gate)}")
    both = te[(te.score >= med) & (te.btcvol >= 50)]
    print(f"gate50 AND top50: avg {both.pnl.mean()*100:+.2f}%  win {both.win.mean()*100:.0f}%  n={len(both)}")
    imp = sorted(zip(FEATS, m.feature_importances_), key=lambda x: -x[1])[:6]
    print("top features:", ", ".join(f"{k}:{v}" for k, v in imp))


fold("2025-01-01", 2025)
fold("2026-01-01", 2026)
print("\nDone.")
