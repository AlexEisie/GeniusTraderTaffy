"""Clean re-validation of vol-gated panic-dip: fixed vol series, year x gate cross, exit grid."""
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

# clean BTC vol series on BTC's own index
btc = data["BTC"].close
btc_vol7 = (btc.pct_change().rolling(168, min_periods=100).std() * np.sqrt(24 * 365) * 100)
print(f"btc_vol7: valid {btc_vol7.notna().sum()}/{len(btc_vol7)}, range {btc_vol7.min():.0f}-{btc_vol7.max():.0f}%, now {btc_vol7.iloc[-1]:.0f}%")
print("btc_vol7 percentiles:", {q: round(btc_vol7.quantile(q), 0) for q in [0.25, 0.5, 0.75, 0.9]})

events = []
for p, d in data.items():
    c = d.close
    r24 = c.pct_change(24)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    lower = mid - 2 * sd
    own_vol = c.pct_change().rolling(168, min_periods=100).std() * np.sqrt(24 * 365) * 100
    sig = (r24 < -0.10) & (c < lower)
    idxs = np.where(sig.values)[0]
    last_i = -999
    for i in idxs:
        if i - last_i < 4:
            continue
        last_i = i
        if i + 49 >= len(d):
            continue
        t = d.index[i]
        bv = btc_vol7.asof(t)
        events.append({
            "pair": p, "t": t, "year": t.year, "entry": d.open.values[i + 1],
            "hi": d.high.values[i + 1 : i + 49], "lo": d.low.values[i + 1 : i + 49],
            "cl": d.close.values[i + 1 : i + 49],
            "btcvol": bv if not np.isnan(bv) else -1,
            "ownvol": own_vol.values[i] if not np.isnan(own_vol.values[i]) else -1,
        })
print(f"total events: {len(events)}; btcvol missing: {sum(1 for e in events if e['btcvol']<0)}")


def ev(events, tgt=0.12, stp=0.06, maxh=24):
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
    return (len(a), a.mean() * 100, (a > 0).mean() * 100, a.sum() * 100) if len(a) else (0, 0, 0, 0)


print("\n=== YEAR x BTC-VOL-GATE cross (tgt12/stop6/24h) ===")
print(f"{'':14s}" + "".join(f"  gate>={g:<3d}      " for g in [0, 40, 50, 60]))
for yr in [2023, 2024, 2025, 2026]:
    row = f"{yr}:         "
    for g in [0, 40, 50, 60]:
        sub = [e for e in events if e["year"] == yr and e["btcvol"] >= g]
        n, avg, win, tot = ev(sub)
        row += f"  n={n:3d} {avg:+5.2f}%  "
    print(row)

print("\n=== own-coin vol gate instead ===")
for yr in [2023, 2024, 2025, 2026]:
    row = f"{yr}:         "
    for g in [0, 80, 100, 120]:
        sub = [e for e in events if e["year"] == yr and e["ownvol"] >= g]
        n, avg, win, tot = ev(sub)
        row += f"  n={n:3d} {avg:+5.2f}%  "
    print(row)

print("\n=== exit grid under btcvol>=40 gate (all years pooled) ===")
sub40 = [e for e in events if e["btcvol"] >= 40]
rows = []
for tgt in [0.06, 0.08, 0.10, 0.12, 0.15]:
    for stp in [0.05, 0.06, 0.08, 0.10]:
        for maxh in [24, 36, 48]:
            n, avg, win, tot = ev(sub40, tgt, stp, maxh)
            rows.append([tgt * 100, stp * 100, maxh, n, avg, win, tot])
g = pd.DataFrame(rows, columns=["tgt%", "stop%", "maxh", "n", "avg%", "win%", "sum%"])
print(g.sort_values("avg%", ascending=False).head(15).to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

print("\n=== monthly distribution of gated events (btcvol>=40) ===")
mm = pd.Series([e["t"].strftime("%Y-%m") for e in sub40]).value_counts().sort_index()
print(mm.to_string())

print("\n=== per-event-avg by year under btcvol>=40, best cell re-check ===")
for yr in [2023, 2024, 2025, 2026]:
    sub = [e for e in sub40 if e["year"] == yr]
    n, avg, win, tot = ev(sub, 0.12, 0.06, 24)
    print(f"{yr}: n={n:3d} avg {avg:+.2f}% win {win:.0f}% sum {tot:+.1f}%")

# concurrency: how many simultaneous positions would this need?
print("\n=== concurrency profile (btcvol>=40, 24h hold) ===")
times = sorted([e["t"] for e in sub40])
import collections
active = []
maxc = 0
counts = collections.Counter()
for t in times:
    active = [a for a in active if (t - a).total_seconds() < 24 * 3600]
    active.append(t)
    counts[len(active)] += 1
    maxc = max(maxc, len(active))
print(f"max concurrent (24h window): {maxc}; distribution: {dict(sorted(counts.items()))}")
print("Done.")
