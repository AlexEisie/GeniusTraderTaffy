"""Achievable-ceiling sims: (A) top-N momentum rotation with costs, (B) dip-event exit grids."""
import glob
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
DATADIR = "/freqtrade/user_data/data/okx"
COST = 0.002  # round-trip taker
MAJORS = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "TRX", "SUI", "WLD", "OKB", "LTC",
    "LINK", "AVAX", "ADA", "DOT", "BCH", "ATOM", "FIL", "NEAR", "APT", "ARB",
    "OP", "INJ", "TIA", "AAVE", "UNI", "HYPE", "PUMP",
]


def load_1h():
    out = {}
    for f in sorted(glob.glob(f"{DATADIR}/*-1h.feather")):
        pair = f.split("/")[-1].replace("-1h.feather", "")
        base = pair.split("_")[0]
        if base not in MAJORS:
            continue
        df = pd.read_feather(f).set_index("date").sort_index()
        out[base] = df
    return out


data = load_1h()
closes = pd.DataFrame({p: d.close for p, d in data.items()})
print(f"pairs {len(closes.columns)}, span {closes.index.min()} -> {closes.index.max()}")

# ============ A) TOP-N ROTATION SIM ============
def T(x):
    if x is None or isinstance(x, pd.Timestamp):
        return x
    return pd.Timestamp(x, tz="UTC")


def rotation(closes, look, n, step, start, end, trend_filter=False, breadth_gate=0.0):
    start, end = T(start), T(end)
    c = closes.loc[:end]
    mom = c.pct_change(look, fill_method=None)
    ema200 = c.ewm(span=200, min_periods=100).mean()
    above = (c > ema200)
    breadth = above.where(c.notna()).mean(axis=1)
    idx = c.loc[start:end].index[::step]
    w_prev = pd.Series(0.0, index=c.columns)
    eq = [1.0]
    dates = [idx[0]]
    for i in range(len(idx) - 1):
        t0, t1 = idx[i], idx[i + 1]
        m = mom.loc[t0].dropna()
        if trend_filter:
            m = m[above.loc[t0, m.index]]
        sel = m.nlargest(n)
        w = pd.Series(0.0, index=c.columns)
        if len(sel) > 0 and breadth.loc[t0] >= breadth_gate:
            w[sel.index] = 1.0 / n  # rest stays cash if fewer than n qualify
        turn = (w - w_prev).abs().sum() / 2
        r = (c.loc[t1] / c.loc[t0] - 1).fillna(0.0)
        pr = (w * r).sum() - turn * COST
        eq.append(eq[-1] * (1 + pr))
        dates.append(t1)
        w_prev = w
    s = pd.Series(eq, index=pd.DatetimeIndex(dates))
    monthly = s.resample("ME").last().pct_change() * 100
    first = s.resample("ME").last().iloc[0] / s.iloc[0] - 1
    monthly.iloc[0] = first * 100
    dd = (s / s.cummax() - 1).min() * 100
    return s.iloc[-1] - 1, dd, monthly


def show(title, res):
    tot, dd, monthly = res
    ms = " ".join(f"{d.strftime('%y-%m')}:{v:+.0f}" for d, v in monthly.items() if not np.isnan(v))
    print(f"{title:46s} tot {tot*100:+7.1f}%  maxDD {dd:6.1f}%  | {ms}")


print("\n=== A) rotation sims, window 2026-02-27 -> end (the 6m regime) ===")
S, E = "2026-02-27", None
E = closes.index.max()
for look in [72, 168, 336]:
    for n in [3, 5]:
        show(f"look{look}h top{n} 24h-rebal", rotation(closes, look, n, 24, S, E))
print("-- with own-trend filter (only hold coins > EMA200, else cash) --")
for look in [72, 168, 336]:
    for n in [3, 5]:
        show(f"look{look}h top{n} 24h-rebal +trend", rotation(closes, look, n, 24, S, E, trend_filter=True))
print("-- trend filter + breadth gate 0.4 --")
for look in [168, 336]:
    show(f"look{look}h top3 +trend +breadth.4", rotation(closes, look, 3, 24, S, E, trend_filter=True, breadth_gate=0.4))
print("-- rebal every 12h vs 48h (look168 top3 +trend) --")
show("look168 top3 12h-rebal +trend", rotation(closes, 168, 3, 12, S, E, trend_filter=True))
show("look168 top3 48h-rebal +trend", rotation(closes, 168, 3, 48, S, E, trend_filter=True))

print("\n=== A2) long history 2023-01 -> end (majors only, robustness) ===")
S2 = "2023-03-01"
for look in [168, 336]:
    for n in [3, 5]:
        r = rotation(closes, look, n, 24, S2, E, trend_filter=True)
        tot, dd, monthly = r
        yearly = (1 + monthly / 100).groupby(monthly.index.year).prod() - 1
        ys = " ".join(f"{y}:{v*100:+.0f}%" for y, v in yearly.items())
        print(f"look{look}h top{n} +trend: tot {tot*100:+8.1f}%  maxDD {dd:6.1f}%  yearly {ys}")
for look in [168]:
    r = rotation(closes, look, 3, 24, S2, E, trend_filter=True, breadth_gate=0.4)
    tot, dd, monthly = r
    yearly = (1 + monthly / 100).groupby(monthly.index.year).prod() - 1
    ys = " ".join(f"{y}:{v*100:+.0f}%" for y, v in yearly.items())
    print(f"look{look}h top3 +trend +breadth.4: tot {tot*100:+8.1f}%  maxDD {dd:6.1f}%  yearly {ys}")

# EW benchmark
r = rotation(closes, 168, len(closes.columns), 24, S, E)
show("EW basket buy&hold-ish (all, daily)", r)

# ============ B) DIP EVENT EXIT GRID ============
print("\n=== B) dip event (ret24<-10% & close<BB_lower) exit grid, conservative fills ===")


def dip_grid(start, end):
    start, end = T(start), T(end)
    events = []
    for p, d in data.items():
        d = d.loc[start:end]
        if len(d) < 300:
            continue
        c = d.close
        r24 = c.pct_change(24)
        mid = c.rolling(20).mean()
        sd = c.rolling(20).std()
        lower = mid - 2 * sd
        sig = (r24 < -0.10) & (c < lower)
        idxs = np.where(sig.values)[0]
        last_i = -999
        for i in idxs:
            if i - last_i < 4:  # 4h cooldown dedupe
                continue
            last_i = i
            if i + 49 >= len(d):
                continue
            entry = d.open.values[i + 1]  # next candle open
            path = d.iloc[i + 1 : i + 49]
            events.append((p, d.index[i], entry, path.high.values, path.low.values, path.close.values))
    print(f"events: {len(events)}  ({start} -> {end})")
    if not events:
        return
    rows = []
    for tgt in [0.03, 0.05, 0.08, 0.12]:
        for stp in [0.04, 0.06, 0.08, 0.12]:
            for maxh in [24, 48]:
                pnl = []
                for (_p, _t, e, hi, lo, cl) in events:
                    out = None
                    for k in range(maxh):
                        if lo[k] <= e * (1 - stp):  # conservative: stop checked first
                            out = -stp
                            break
                        if hi[k] >= e * (1 + tgt):
                            out = tgt
                            break
                    if out is None:
                        out = cl[maxh - 1] / e - 1
                    pnl.append(out - COST)
                pnl = np.array(pnl)
                rows.append([tgt, stp, maxh, pnl.mean() * 100, (pnl > 0).mean() * 100, pnl.sum() * 100])
    g = pd.DataFrame(rows, columns=["tgt", "stop", "maxh", "avg%", "win%", "sum%"])
    print(g.sort_values("avg%", ascending=False).head(12).to_string(index=False, float_format=lambda x: f"{x:6.2f}"))


dip_grid("2026-02-20", None)
dip_grid("2023-01-01", "2026-02-20")
print("Done.")
