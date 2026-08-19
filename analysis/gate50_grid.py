"""Exit-grid re-optimization under the gate50 subsample + stability by year."""
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

events = []
for p, d in data.items():
    c = d.close
    r24 = c.pct_change(24)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    lower = mid - 2 * sd
    sig = (r24 < -0.10) & (c < lower)
    idxs = np.where(sig.values)[0]
    last_i = -999
    for i in idxs:
        if i - last_i < 4:
            continue
        last_i = i
        if i + 73 >= len(d):
            continue
        t = d.index[i]
        bv = btc_vol7.asof(t)
        if np.isnan(bv) or bv < 50:
            continue
        events.append({
            "pair": p, "t": t, "year": t.year, "entry": d.open.values[i + 1],
            "hi": d.high.values[i + 1 : i + 73], "lo": d.low.values[i + 1 : i + 73],
            "cl": d.close.values[i + 1 : i + 73],
        })
events.sort(key=lambda e: e["t"])
print(f"gate50 events: {len(events)}")


def ev(events, tgt, stp, maxh):
    pnl = []
    for e in events:
        out = None
        for k in range(maxh):
            if e["lo"][k] <= e["entry"] * (1 - stp):
                out = -stp
                break
            if e["hi"][k] >= e["entry"] * (1 + tgt):
                out = tgt
                break
        if out is None:
            out = e["cl"][maxh - 1] / e["entry"] - 1
        pnl.append(out - COST)
    a = np.array(pnl)
    return len(a), a.mean() * 100, (a > 0).mean() * 100


print("\n=== exit grid on gate50 events (all years) ===")
rows = []
for tgt in [0.08, 0.10, 0.12, 0.15]:
    for stp in [0.04, 0.05, 0.06, 0.08]:
        for maxh in [24, 36, 48, 72]:
            n, avg, win = ev(events, tgt, stp, maxh)
            rows.append([tgt * 100, stp * 100, maxh, n, avg, win])
g = pd.DataFrame(rows, columns=["tgt%", "stop%", "maxh", "n", "avg%", "win%"])
print(g.sort_values("avg%", ascending=False).head(14).to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

print("\n=== chosen cell (10/5/36) vs top cell, stability by year ===")
top = g.sort_values("avg%", ascending=False).iloc[0]
for (tgt, stp, maxh, tag) in [(0.10, 0.05, 36, "current 10/5/36"),
                              (top["tgt%"] / 100, top["stop%"] / 100, int(top["maxh"]), f"top {top['tgt%']:.0f}/{top['stop%']:.0f}/{int(top['maxh'])}")]:
    print(f"-- {tag}")
    for yr in [2023, 2024, 2025, 2026]:
        sub = [e for e in events if e["year"] == yr]
        n, avg, win = ev(sub, tgt, stp, maxh)
        print(f"   {yr}: n={n:3d} avg {avg:+.2f}% win {win:.0f}%")
print("Done.")
