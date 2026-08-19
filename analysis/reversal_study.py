"""Reversal/liquidity-provision studies: XS reversal rotation, dip regime split, vol gating."""
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


def T(x):
    return None if x is None else pd.Timestamp(x, tz="UTC")


data = {}
for f in sorted(glob.glob(f"{DATADIR}/*-1h.feather")):
    pair = f.split("/")[-1].replace("-1h.feather", "")
    base = pair.split("_")[0]
    if base in MAJORS:
        data[base] = pd.read_feather(f).set_index("date").sort_index()
closes = pd.DataFrame({p: d.close for p, d in data.items()})
btc_vol7 = closes["BTC"].pct_change().rolling(168).std() * np.sqrt(24 * 365) * 100  # ann %

print("=== 1) XS REVERSAL rotation: buy bottom-N by lookback ret, hold `step`h ===")


def reversal_rot(look, n, step, start, end, vol_gate=None):
    start, end = T(start), T(end)
    c = closes.loc[:end]
    r_look = c.pct_change(look, fill_method=None)
    idx = c.loc[start:end].index[::step]
    eq, dates = [1.0], [idx[0]]
    w_prev = pd.Series(0.0, index=c.columns)
    for i in range(len(idx) - 1):
        t0, t1 = idx[i], idx[i + 1]
        m = r_look.loc[t0].dropna()
        w = pd.Series(0.0, index=c.columns)
        gate_ok = True if vol_gate is None else (btc_vol7.loc[:t0].iloc[-1] >= vol_gate)
        if len(m) >= n and gate_ok:
            sel = m.nsmallest(n)
            w[sel.index] = 1.0 / n
        turn = (w - w_prev).abs().sum() / 2
        r = (c.loc[t1] / c.loc[t0] - 1).fillna(0.0)
        eq.append(eq[-1] * (1 + (w * r).sum() - turn * COST))
        dates.append(t1)
        w_prev = w
    s = pd.Series(eq, index=pd.DatetimeIndex(dates))
    monthly = s.resample("ME").last().pct_change() * 100
    monthly.iloc[0] = (s.resample("ME").last().iloc[0] / s.iloc[0] - 1) * 100
    dd = (s / s.cummax() - 1).min() * 100
    return s.iloc[-1] - 1, dd, monthly


def show(title, res, yearly_only=False):
    tot, dd, monthly = res
    if yearly_only:
        yearly = (1 + monthly / 100).groupby(monthly.index.year).prod() - 1
        ms = " ".join(f"{y}:{v*100:+.0f}%" for y, v in yearly.items())
    else:
        ms = " ".join(f"{d.strftime('%y-%m')}:{v:+.0f}" for d, v in monthly.items() if not np.isnan(v))
    print(f"{title:44s} tot {tot*100:+8.1f}%  maxDD {dd:6.1f}%  | {ms}")


for look in [24, 72]:
    for n in [3, 5]:
        for step in [24, 48]:
            show(f"rev look{look}h bot{n} hold{step}h [2026]", reversal_rot(look, n, step, "2026-02-27", None))
print("-- long history --")
for look in [24, 72]:
    for n in [3]:
        show(f"rev look{look}h bot{n} hold24h [2023-]", reversal_rot(look, n, 24, "2023-03-01", None), yearly_only=True)
print("-- with BTC 7d vol>=40% gate, long history --")
for look in [24, 72]:
    show(f"rev look{look}h bot3 hold24 volGate40 [2023-]", reversal_rot(look, 24 and 3, 24, "2023-03-01", None, vol_gate=40) if False else reversal_rot(look, 3, 24, "2023-03-01", None, vol_gate=40), yearly_only=True)

print()
print("=== 2) DIP events by year + vol gate + volume filter (best cell: tgt12/stop6, 24h) ===")


def dip_events(start, end):
    start, end = T(start), T(end)
    ev = []
    for p, d in data.items():
        d2 = d.loc[start:end] if start is not None else d.loc[:end]
        if len(d2) < 300:
            continue
        c = d2.close
        r24 = c.pct_change(24)
        mid = c.rolling(20).mean()
        sd = c.rolling(20).std()
        lower = mid - 2 * sd
        volr = d2.volume / d2.volume.rolling(168).mean()
        sig = (r24 < -0.10) & (c < lower)
        idxs = np.where(sig.values)[0]
        last_i = -999
        for i in idxs:
            if i - last_i < 4:
                continue
            last_i = i
            if i + 49 >= len(d2) or i < 1:
                continue
            entry = d2.open.values[i + 1]
            path = d2.iloc[i + 1 : i + 49]
            ev.append({
                "pair": p, "t": d2.index[i], "entry": entry,
                "hi": path.high.values, "lo": path.low.values, "cl": path.close.values,
                "volr": volr.values[i],
                "btcvol": btc_vol7.reindex([d2.index[i]], method="ffill").iloc[0],
            })
    return ev


def grid_eval(events, tgt=0.12, stp=0.06, maxh=24):
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
    pnl = np.array(pnl)
    return len(pnl), pnl.mean() * 100 if len(pnl) else 0, (pnl > 0).mean() * 100 if len(pnl) else 0, pnl.sum() * 100 if len(pnl) else 0


allev = dip_events(None, None)
df_ev = pd.DataFrame([{"t": e["t"], "year": e["t"].year, "volr": e["volr"], "btcvol": e["btcvol"]} for e in allev])
for yr in [2023, 2024, 2025, 2026]:
    sub = [e for e in allev if e["t"].year == yr]
    n, avg, win, tot = grid_eval(sub)
    print(f"year {yr}: n={n:4d}  avg {avg:+.2f}%  win {win:4.1f}%  sum {tot:+7.1f}%")

print("\n-- vol gate: only take events when BTC 7d ann vol >= X --")
for gate in [0, 30, 40, 50, 60]:
    sub = [e for e in allev if e["btcvol"] >= gate]
    n, avg, win, tot = grid_eval(sub)
    print(f"btcvol>={gate:3d}%: n={n:4d}  avg {avg:+.2f}%  win {win:4.1f}%  sum {tot:+7.1f}%")

print("\n-- volume-spike filter (event candle volume vs 7d mean) --")
for vf in [0, 1.0, 1.5, 2.0, 3.0]:
    sub = [e for e in allev if e["volr"] >= vf]
    n, avg, win, tot = grid_eval(sub)
    print(f"volr>={vf:.1f}: n={n:4d}  avg {avg:+.2f}%  win {win:4.1f}%  sum {tot:+7.1f}%")

print("\n-- 2026 only, vol gate x volume filter --")
sub26 = [e for e in allev if e["t"].year == 2026]
for gate in [0, 40, 50]:
    for vf in [0, 1.5]:
        sub = [e for e in sub26 if e["btcvol"] >= gate and e["volr"] >= vf]
        n, avg, win, tot = grid_eval(sub)
        print(f"2026 btcvol>={gate} volr>={vf}: n={n:4d}  avg {avg:+.2f}%  win {win:4.1f}%  sum {tot:+7.1f}%")

print("\n-- deeper entry: limit fill 2% below signal close (maker), tgt12/stop6/24h --")


def grid_eval_limit(events, disc=0.02, tgt=0.12, stp=0.06, maxh=24, cost=0.0016):
    pnl, filled = [], 0
    for e in events:
        lim = e["cl"][0] * 0  # placeholder
    for e in events:
        limit_price = e["entry"] * (1 - disc)
        # check fill within first 6h
        fill_k = None
        for k in range(6):
            if e["lo"][k] <= limit_price:
                fill_k = k
                break
        if fill_k is None:
            continue
        filled += 1
        out = None
        for k in range(fill_k, maxh):
            if e["lo"][k] <= limit_price * (1 - stp) and k > fill_k:
                out = -stp
                break
            if e["hi"][k] >= limit_price * (1 + tgt):
                out = tgt
                break
        if out is None:
            out = e["cl"][maxh - 1] / limit_price - 1
        pnl.append(out - cost)
    pnl = np.array(pnl)
    if len(pnl) == 0:
        return 0, 0, 0, 0
    return len(pnl), pnl.mean() * 100, (pnl > 0).mean() * 100, pnl.sum() * 100


for disc in [0.01, 0.02, 0.03, 0.05]:
    n, avg, win, tot = grid_eval_limit(sub26, disc=disc)
    print(f"2026 limit -{disc*100:.0f}%: filled n={n:4d}  avg {avg:+.2f}%  win {win:4.1f}%  sum {tot:+7.1f}%")
print("Done.")
