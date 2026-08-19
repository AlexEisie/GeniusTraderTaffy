"""Deeper edge conditioning: can stacked filters beat 0.2% round-trip costs?"""
import glob
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
DATADIR = "/freqtrade/user_data/data/okx"

EXCLUDE = ("USDC", "USDG", "XAUT", "XMU", "XSNDK", "XSPCX", "XSKHY", "DOS")


def load(tf, min_bars=1000):
    out = {}
    for f in sorted(glob.glob(f"{DATADIR}/*-{tf}.feather")):
        pair = f.split("/")[-1].replace(f"-{tf}.feather", "")
        if any(pair.startswith(e) for e in EXCLUDE):
            continue
        df = pd.read_feather(f).set_index("date").sort_index()
        if len(df) >= min_bars:
            out[pair] = df
    return out


h1 = load("1h")
print(f"pairs: {len(h1)}", sorted(h1.keys()))

# restrict to common window (the 6-month window all pairs share)
START = "2026-02-20"
pool = []
for p, df in h1.items():
    d = df.loc[START:].copy()
    d["pair"] = p
    r = d.close.pct_change()
    d["r1"] = r
    d["ret4"] = d.close.pct_change(4)
    d["ret24"] = d.close.pct_change(24)
    d["ret168"] = d.close.pct_change(168)
    z = (r - r.rolling(200).mean()) / r.rolling(200).std()
    d["z"] = z
    d["ema200"] = d.close.ewm(span=200).mean()
    d["rsi"] = 100 - 100 / (1 + r.clip(lower=0).rolling(14).mean() / (-r.clip(upper=0)).rolling(14).mean())
    bb_mid = d.close.rolling(20).mean()
    bb_sd = d.close.rolling(20).std()
    d["bbz"] = (d.close - bb_mid) / (2 * bb_sd + 1e-12)
    d["atrpct"] = (d.high - d.low).rolling(14).mean() / d.close
    for k in [2, 4, 8, 12, 24, 48]:
        d[f"fwd{k}"] = d.close.shift(-k) / d.close - 1
    # max adverse excursion of the next 24h (worst dip if entering now)
    d["mae24"] = d.low.rolling(24).min().shift(-24) / d.close - 1
    d["mfe24"] = d.high.rolling(24).max().shift(-24) / d.close - 1
    pool.append(d)
P = pd.concat(pool)
print(f"rows: {len(P)}  window {P.index.min()} -> {P.index.max()}")


def cond(name, mask, hold=(4, 8, 24, 48)):
    s = P[mask]
    if len(s) == 0:
        print(f"{name:52s} n=0")
        return
    parts = "  ".join(f"f{k}h {s[f'fwd{k}'].mean()*100:+.3f}%" for k in hold)
    print(f"{name:52s} n={len(s):5d}  {parts}  mae24 {s.mae24.mean()*100:+.2f}%  mfe24 {s.mfe24.mean()*100:+.2f}%")


print("\n--- DIP STACKS (pooled 1h, Feb-Aug 2026). Costs: ~0.20% round trip taker ---")
cond("baseline", P.fwd4.notna())
cond("ret4 < -3%", P.ret4 < -0.03)
cond("ret4 < -5%", P.ret4 < -0.05)
cond("ret24 < -8%", P.ret24 < -0.08)
cond("ret24 < -12%", P.ret24 < -0.12)
cond("ret4<-4% & rsi<25", (P.ret4 < -0.04) & (P.rsi < 25))
cond("ret24<-10% & bbz<-1", (P.ret24 < -0.10) & (P.bbz < -1))
cond("ret24<-10% & z<-2 (capitulation candle now)", (P.ret24 < -0.10) & (P.z < -2))
cond("ret24<-10% & recovering (r1>0)", (P.ret24 < -0.10) & (P.r1 > 0))
cond("ret24<-10% & r1>0 & rsi<30", (P.ret24 < -0.10) & (P.r1 > 0) & (P.rsi < 30))
cond("ret24<-15% & r1>0", (P.ret24 < -0.15) & (P.r1 > 0))

print("\n--- same but split by 7d relative momentum (cross-sectional rank) ---")
P["rank168"] = P.groupby(P.index)["ret168"].rank(pct=True)
cond("dip24<-10% & weak coin (rank<0.3)", (P.ret24 < -0.10) & (P.rank168 < 0.3))
cond("dip24<-10% & strong coin (rank>0.7)", (P.ret24 < -0.10) & (P.rank168 > 0.7))

print("\n--- MOMENTUM RANK persistence / hold-horizon economics ---")
for q in [(0.8, 1.01, "top20%"), (0.6, 0.8, "60-80%"), (0.2, 0.4, "20-40%"), (0.0, 0.2, "bottom20%")]:
    m = (P.rank168 >= q[0]) & (P.rank168 < q[1])
    s = P[m]
    print(f"rank168 {q[2]:9s} n={len(s):6d}  fwd24 {s.fwd24.mean()*100:+.3f}%  fwd48 {s.fwd48.mean()*100:+.3f}%")

print("\n--- top-rank + trigger combos (entry timing within strong coins) ---")
cond("rank>0.8 & pullback ret4<-2%", (P.rank168 > 0.8) & (P.ret4 < -0.02))
cond("rank>0.8 & pullback ret4<-2% & r1>0", (P.rank168 > 0.8) & (P.ret4 < -0.02) & (P.r1 > 0))
cond("rank>0.8 & bbz<-0.8", (P.rank168 > 0.8) & (P.bbz < -0.8))
cond("rank>0.8 & new 24h high", (P.rank168 > 0.8) & (P.close >= P.close.rolling(24).max()))
cond("rank>0.9 (very strong)", P.rank168 > 0.9)
cond("rank>0.9 & ret24>0", (P.rank168 > 0.9) & (P.ret24 > 0))

print("\n--- SEASONALITY (bug-fixed: per-pair returns) ---")
P["hr"] = P.index.hour
hod = P.groupby("hr").r1.mean() * 1e4
print("hour(UTC) mean bp:", " ".join(f"{h}:{v:+.1f}" for h, v in hod.items()))
P["dow"] = P.index.dayofweek
dow = P.groupby("dow").r1.mean() * 1e4
print("dow(0=Mon) mean bp:", " ".join(f"{d}:{v:+.1f}" for d, v in dow.items()))

print("\n--- BTC-relative behaviour: do alts follow BTC dips? (lead-lag) ---")
btc = h1["BTC_USDT"].loc[START:].close.pct_change()
btc_z = (btc - btc.rolling(200).mean()) / btc.rolling(200).std()
P2 = P.join(btc_z.rename("btc_z"), how="left")
cond2 = P2[(P2.btc_z < -2) & (P2.pair != "BTC_USDT")]
print(f"after BTC 1h crash z<-2: alt fwd4 {cond2.fwd4.mean()*100:+.3f}%  fwd24 {cond2.fwd24.mean()*100:+.3f}%  n={len(cond2)}")
cond3 = P2[(P2.btc_z > 2) & (P2.pair != "BTC_USDT")]
print(f"after BTC 1h pump  z>+2: alt fwd4 {cond3.fwd4.mean()*100:+.3f}%  fwd24 {cond3.fwd24.mean()*100:+.3f}%  n={len(cond3)}")
print("Done.")
