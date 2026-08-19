"""Market regime & edge study on OKX spot data (runs inside freqtrade container)."""
import glob
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
DATADIR = "/freqtrade/user_data/data/okx"


def load(tf):
    out = {}
    for f in sorted(glob.glob(f"{DATADIR}/*-{tf}.feather")):
        pair = f.split("/")[-1].replace(f"-{tf}.feather", "")
        df = pd.read_feather(f)
        df = df.set_index("date").sort_index()
        out[pair] = df
    return out


print("=" * 70)
print("1) BUY & HOLD BENCHMARK  (window of the 1h data)")
print("=" * 70)
h1 = load("1h")
rows = []
for p, df in h1.items():
    if len(df) < 500:
        continue
    ret = df.close.iloc[-1] / df.close.iloc[0] - 1
    dd = (df.close / df.close.cummax() - 1).min()
    vol = df.close.pct_change().std() * np.sqrt(24 * 365)
    rows.append([p, len(df), df.index[0].date(), df.index[-1].date(), ret * 100, dd * 100, vol * 100])
bh = pd.DataFrame(rows, columns=["pair", "bars", "start", "end", "bh_ret%", "maxdd%", "annvol%"])
bh = bh.sort_values("bh_ret%", ascending=False)
print(bh.to_string(index=False, float_format=lambda x: f"{x:8.1f}"))
print(f"\nEqual-weight basket B&H: {bh['bh_ret%'].mean():.1f}%  median {bh['bh_ret%'].median():.1f}%")

print()
print("=" * 70)
print("2) MONTHLY REGIME TIMELINE (BTC, ETH, SOL, equal-weight alt basket) [1h data]")
print("=" * 70)
monthly = {}
for p, df in h1.items():
    monthly[p] = df.close.resample("ME").last().pct_change() * 100
mdf = pd.DataFrame(monthly)
mdf["ALT_EW"] = mdf.drop(columns=[c for c in ["BTC_USDT", "ETH_USDT"] if c in mdf], errors="ignore").mean(axis=1)
cols = [c for c in ["BTC_USDT", "ETH_USDT", "SOL_USDT", "ALT_EW"] if c in mdf.columns]
print(mdf[cols].dropna(how="all").to_string(float_format=lambda x: f"{x:7.1f}"))

print()
print("=" * 70)
print("2b) LONG REGIME TIMELINE  (6h data, 2 years, monthly)")
print("=" * 70)
h6 = load("6h")
monthly6 = {}
for p in ["BTC_USDT", "ETH_USDT", "SOL_USDT"]:
    if p in h6:
        monthly6[p] = h6[p].close.resample("ME").last().pct_change() * 100
m6 = pd.DataFrame(monthly6).dropna(how="all")
print(m6.tail(15).to_string(float_format=lambda x: f"{x:7.1f}"))

print()
print("=" * 70)
print("3) TREND vs CHOP DIAGNOSTICS (1h)")
print("=" * 70)
for p in ["BTC_USDT", "ETH_USDT", "SOL_USDT"]:
    df = h1[p].copy()
    ema200 = df.close.ewm(span=200).mean()
    above = (df.close > ema200).mean() * 100
    r = df.close.pct_change()
    # variance ratio: var(k-period) / (k * var(1-period)); >1 momentum, <1 mean reversion
    vrs = {}
    for k in [4, 24, 96]:
        rk = df.close.pct_change(k)
        vrs[k] = rk.var() / (k * r.var())
    ac1 = r.autocorr(1)
    print(f"{p}: above EMA200 {above:5.1f}% of time | VR(4h)={vrs[4]:.2f} VR(1d)={vrs[24]:.2f} VR(4d)={vrs[96]:.2f} | AC1={ac1:+.3f}")

print()
print("=" * 70)
print("4) CONDITIONAL FORWARD RETURNS (all pairs pooled, 1h)")
print("=" * 70)
pool = []
for p, df in h1.items():
    d = df.copy()
    r = d.close.pct_change()
    z = (r - r.rolling(200).mean()) / r.rolling(200).std()
    d["z"] = z
    d["fwd4"] = d.close.shift(-4) / d.close - 1
    d["fwd24"] = d.close.shift(-24) / d.close - 1
    d["hi20"] = d.close.rolling(20).max().shift(1)
    d["lo20"] = d.close.rolling(20).min().shift(1)
    d["ema200"] = d.close.ewm(span=200).mean()
    d["rsi_proxy"] = r.rolling(14).apply(lambda x: (x[x > 0].sum()) / (abs(x).sum() + 1e-12) * 100, raw=False)
    pool.append(d)
P = pd.concat(pool)
base4, base24 = P.fwd4.mean() * 100, P.fwd24.mean() * 100
print(f"baseline: fwd4h {base4:+.3f}%  fwd24h {base24:+.3f}%   (n={len(P)})")


def cond(name, mask):
    s = P[mask]
    print(f"{name:44s} n={len(s):6d}  fwd4h {s.fwd4.mean()*100:+.3f}%  fwd24h {s.fwd24.mean()*100:+.3f}%  win4h {(s.fwd4>0).mean()*100:4.1f}%")


cond("big DOWN candle z<-2", P.z < -2)
cond("big DOWN z<-2, price>EMA200 (dip in uptrend)", (P.z < -2) & (P.close > P.ema200))
cond("big DOWN z<-2, price<EMA200 (dip in downtr)", (P.z < -2) & (P.close < P.ema200))
cond("big UP candle z>2", P.z > 2)
cond("breakout close>20h high", P.close > P.hi20)
cond("breakout & price>EMA200", (P.close > P.hi20) & (P.close > P.ema200))
cond("breakdown close<20h low", P.close < P.lo20)
cond("RSIproxy<30 (oversold)", P.rsi_proxy < 30)
cond("RSIproxy<30 & >EMA200", (P.rsi_proxy < 30) & (P.close > P.ema200))
cond("RSIproxy>70 (overbought)", P.rsi_proxy > 70)

print()
print("=" * 70)
print("5) HOUR-OF-DAY / DAY-OF-WEEK SEASONALITY (pooled 1h, mean ret bp)")
print("=" * 70)
P["hr"] = P.index.hour
P["dow"] = P.index.dayofweek
r1 = P.close.pct_change()
P["r1"] = P.groupby(level=0).close.pct_change() if False else r1  # simple pooled
hod = P.groupby("hr").r1.mean() * 1e4
print("hour(UTC):", " ".join(f"{h}:{v:+.1f}" for h, v in hod.items()))
dow = P.groupby("dow").r1.mean() * 1e4
print("dow(0=Mon):", " ".join(f"{d}:{v:+.1f}" for d, v in dow.items()))

print()
print("=" * 70)
print("6) CROSS-SECTIONAL MOMENTUM (daily rebalance, 1h data -> daily)")
print("=" * 70)
closes = pd.DataFrame({p: df.close for p, df in h1.items()}).resample("1D").last()
rets = closes.pct_change()
look7 = closes.pct_change(7)
fwd1 = rets.shift(-1)
ranks = look7.rank(axis=1, pct=True)
top = fwd1[ranks > 0.8].mean(axis=1)
bot = fwd1[ranks < 0.2].mean(axis=1)
mid = fwd1[(ranks >= 0.4) & (ranks <= 0.6)].mean(axis=1)
print(f"7d-momentum top quintile next-day: {top.mean()*100:+.3f}%/d  (cum {(1+top.fillna(0)).prod()-1:+.1%})")
print(f"7d-momentum mid quintile next-day: {mid.mean()*100:+.3f}%/d  (cum {(1+mid.fillna(0)).prod()-1:+.1%})")
print(f"7d-momentum bottom quint next-day: {bot.mean()*100:+.3f}%/d  (cum {(1+bot.fillna(0)).prod()-1:+.1%})")
look1 = closes.pct_change(1)
ranks1 = look1.rank(axis=1, pct=True)
top1 = fwd1[ranks1 > 0.8].mean(axis=1)
bot1 = fwd1[ranks1 < 0.2].mean(axis=1)
print(f"1d-reversal: prev-day losers next-day {bot1.mean()*100:+.3f}%/d, winners {top1.mean()*100:+.3f}%/d")

print()
print("=" * 70)
print("7) SAME TESTS ON 15m (fine-grained MR check, pooled)")
print("=" * 70)
m15 = load("15m")
pool2 = []
for p, df in m15.items():
    d = df.copy()
    r = d.close.pct_change()
    z = (r - r.rolling(400).mean()) / r.rolling(400).std()
    d["z"] = z
    d["fwd8"] = d.close.shift(-8) / d.close - 1   # 2h
    d["fwd96"] = d.close.shift(-96) / d.close - 1  # 24h
    d["ema200"] = d.close.ewm(span=200).mean()
    bb_mid = d.close.rolling(20).mean()
    bb_sd = d.close.rolling(20).std()
    d["bbz"] = (d.close - bb_mid) / (2 * bb_sd)
    pool2.append(d)
Q = pd.concat(pool2)
print(f"baseline fwd2h {Q.fwd8.mean()*100:+.4f}%  fwd24h {Q.fwd96.mean()*100:+.3f}%  (n={len(Q)})")


def cond2(name, mask):
    s = Q[mask]
    print(f"{name:44s} n={len(s):6d}  fwd2h {s.fwd8.mean()*100:+.4f}%  fwd24h {s.fwd96.mean()*100:+.3f}%  win2h {(s.fwd8>0).mean()*100:4.1f}%")


cond2("15m z<-2.5 crash candle", Q.z < -2.5)
cond2("15m z<-2.5 & >EMA200", (Q.z < -2.5) & (Q.close > Q.ema200))
cond2("close below lower BB (bbz<-1)", Q.bbz < -1)
cond2("bbz<-1 & >EMA200", (Q.bbz < -1) & (Q.close > Q.ema200))
cond2("bbz>+1 (above upper BB)", Q.bbz > 1)
print("\nDone.")
