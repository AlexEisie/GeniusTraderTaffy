"""Portfolio-level replay of panic-dip with slot constraints: find why freqtrade result degraded."""
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

# Build event list (2026 window, gate>=40)
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
        if i + 74 >= len(d):
            continue
        t = d.index[i]
        bv = btc_vol7.asof(t)
        if np.isnan(bv) or bv < 40:
            continue
        br = btc_ret24.asof(t)
        events.append({
            "pair": p, "t": t, "entry": d.open.values[i + 1], "ret24": r24.values[i],
            "hi": d.high.values[i + 1 : i + 74], "lo": d.low.values[i + 1 : i + 74],
            "cl": d.close.values[i + 1 : i + 74], "btc_ret24": br, "btcvol": bv,
        })
events.sort(key=lambda e: e["t"])
print(f"gated events: {len(events)}")


def trade_pnl(e, tgt, stp, maxh):
    for k in range(maxh):
        if e["lo"][k] <= e["entry"] * (1 - stp):
            return -stp - COST, k + 1
        if e["hi"][k] >= e["entry"] * (1 + tgt):
            return tgt - COST, k + 1
    return e["cl"][maxh - 1] / e["entry"] - 1 - COST, maxh


def replay(events, slots=5, tgt=0.10, stp=0.05, maxh=36, guard=False,
           extra=None, label=""):
    """Chronological replay with limited slots; equal fraction 1/slots compounding."""
    eq = 1.0
    busy = []  # (free_time, pair)
    taken, skipped_slots = 0, 0
    pnls = []
    guard_until = None
    stop_times = []
    for e in events:
        t = e["t"]
        busy = [b for b in busy if b[0] > t]
        if extra and not extra(e):
            continue
        if guard and guard_until is not None and t < guard_until:
            continue
        if len(busy) >= slots:
            skipped_slots += 1
            continue
        pnl, dur = trade_pnl(e, tgt, stp, maxh)
        pnls.append(pnl)
        taken += 1
        eq *= 1 + pnl / slots
        busy.append((t + pd.Timedelta(hours=dur), e["pair"]))
        if pnl < -stp * 0.9:
            stop_times.append(t)
            stop_times = [s for s in stop_times if (t - s).total_seconds() < 48 * 3600]
            if guard and len(stop_times) >= 5:
                guard_until = t + pd.Timedelta(hours=12)
    a = np.array(pnls)
    if len(a) == 0:
        print(f"{label:52s} no trades")
        return
    print(f"{label:52s} taken {taken:3d} (slotskip {skipped_slots:3d})  avg {a.mean()*100:+5.2f}%  win {(a>0).mean()*100:4.1f}%  port {eq-1:+7.1%}")


print("\n--- replicate freqtrade-ish behavior and variants (2026, gate40) ---")
replay(events, slots=5, guard=True, label="A: slots5 guard tgt10/stp5/36h (≈freqtrade)")
replay(events, slots=5, guard=False, label="B: A without StoplossGuard")
replay(events, slots=5, guard=False, maxh=24, label="C: B with maxh 24")
replay(events, slots=8, guard=False, label="D: B with slots 8")
replay(events, slots=999, guard=False, label="E: unlimited slots (study ceiling)")
replay(events, slots=5, guard=False, extra=lambda e: e["ret24"] < -0.13, label="F: deeper dip only ret24<-13%")
replay(events, slots=5, guard=False, extra=lambda e: e["btc_ret24"] < -0.04, label="G: climax only (BTC ret24<-4%)")
replay(events, slots=5, guard=False, extra=lambda e: e["btc_ret24"] < -0.06, label="H: climax only (BTC ret24<-6%)")
replay(events, slots=5, guard=False, maxh=24, extra=lambda e: e["btc_ret24"] < -0.04, label="I: G + maxh24")
replay(events, slots=3, guard=False, extra=lambda e: e["btc_ret24"] < -0.06, label="J: H + slots3")

print("\n--- monster-wave filters (ex-ante observable) ---")
# trailing signal count within 12h (computed causally: count of events with t in (t-12h, t])
ts = pd.Series(1, index=pd.DatetimeIndex([e["t"] for e in events]))
cnt12 = ts.rolling("12h").sum()
cnt_map = dict(zip(ts.index, cnt12.values))
# NOTE: duplicate timestamps possible -> recompute robustly
tsdf = pd.DataFrame({"t": [e["t"] for e in events]})
tsdf["one"] = 1
tsdf = tsdf.sort_values("t")
counts = []
times = tsdf["t"].values
for i, e in enumerate(events):
    t = e["t"]
    c = ((tsdf["t"] > t - pd.Timedelta(hours=12)) & (tsdf["t"] <= t)).sum()
    e["cnt12"] = c
btc30d_high = btc.rolling(24 * 30).max()
btc_dd30 = btc / btc30d_high - 1
for e in events:
    v = btc_dd30.asof(e["t"])
    e["btc_dd30"] = v if not np.isnan(v) else 0

replay(events, slots=5, guard=True, extra=lambda e: e["cnt12"] <= 8, label="K1: A + cnt12<=8")
replay(events, slots=5, guard=True, extra=lambda e: e["cnt12"] <= 12, label="K2: A + cnt12<=12")
replay(events, slots=5, guard=True, extra=lambda e: e["cnt12"] <= 20, label="K3: A + cnt12<=20")
replay(events, slots=5, guard=True, extra=lambda e: e["btc_dd30"] > -0.15, label="L1: A + BTC dd30 > -15%")
replay(events, slots=5, guard=True, extra=lambda e: e["btc_dd30"] > -0.10, label="L2: A + BTC dd30 > -10%")
replay(events, slots=5, guard=True, extra=lambda e: e["cnt12"] <= 12 and e["btc_dd30"] > -0.15, label="M: A + cnt<=12 + dd30>-15%")

replay(events, slots=5, guard=True, extra=lambda e: e["btcvol"] >= 50, label="N: A + btcvol>=50 (extreme only)")
replay(events, slots=5, guard=True, extra=lambda e: e["btcvol"] >= 55, label="O: A + btcvol>=55")
replay(events, slots=5, guard=True, tgt=0.15, stp=0.08, maxh=72, extra=lambda e: e["btcvol"] >= 50, label="P: vol50 wide exits 15/8/72")
replay(events, slots=8, guard=True, tgt=0.15, stp=0.08, maxh=72, extra=lambda e: e["btcvol"] >= 50, label="P8: same, slots 8")
replay(events, slots=5, guard=True, tgt=0.12, stp=0.06, maxh=48, extra=lambda e: e["btcvol"] >= 50, label="Q: vol50 mid exits 12/6/48")

print("\n--- yearly P&L of A vs M ---")


def replay_yearly(events, extra=None, label="", tgt=0.10, stp=0.05, maxh=36):
    busy = []
    guard_until = None
    stop_times = []
    eqs = {}
    for e in events:
        t = e["t"]
        yr = t.year
        eqs.setdefault(yr, 1.0)
        busy = [b for b in busy if b[0] > t]
        if extra and not extra(e):
            continue
        if guard_until is not None and t < guard_until:
            continue
        if len(busy) >= 5:
            continue
        pnl, dur = trade_pnl(e, tgt, stp, maxh)
        eqs[yr] *= 1 + pnl / 5
        busy.append((t + pd.Timedelta(hours=dur), e["pair"]))
        if pnl < -stp * 0.9:
            stop_times.append(t)
            stop_times = [s for s in stop_times if (t - s).total_seconds() < 48 * 3600]
            if len(stop_times) >= 5:
                guard_until = t + pd.Timedelta(hours=12)
    print(label, " ".join(f"{y}:{(v-1)*100:+.1f}%" for y, v in sorted(eqs.items())))


replay_yearly(events, label="A yearly:")
replay_yearly(events, extra=lambda e: e["cnt12"] <= 12 and e["btc_dd30"] > -0.15, label="M yearly:")
replay_yearly(events, extra=lambda e: e["btcvol"] >= 50, label="N yearly (vol>=50):")
replay_yearly(events, extra=lambda e: e["btcvol"] >= 55, label="O yearly (vol>=55):")
replay_yearly(events, extra=lambda e: e["btcvol"] >= 50, tgt=0.15, stp=0.08, maxh=72, label="P yearly (vol50 15/8/72):")
replay_yearly(events, extra=lambda e: e["btcvol"] >= 50, tgt=0.12, stp=0.06, maxh=48, label="Q yearly (vol50 12/6/48):")

print("\n--- what did slot-constrained selection actually take? timing within waves ---")
# group events into waves (gaps > 24h)
waves, cur = [], []
for e in events:
    if cur and (e["t"] - cur[-1]["t"]).total_seconds() > 24 * 3600:
        waves.append(cur)
        cur = []
    cur.append(e)
if cur:
    waves.append(cur)
print(f"waves: {len(waves)}, sizes: {[len(w) for w in waves]}")
for w in waves:
    if len(w) < 6:
        continue
    pnls = [trade_pnl(e, 0.10, 0.05, 36)[0] for e in w]
    first5 = np.mean(pnls[:5]) * 100
    rest = np.mean(pnls[5:]) * 100
    print(f"wave {w[0]['t'].strftime('%m-%d %H:%M')} n={len(w)}: first5 avg {first5:+.2f}%  rest avg {rest:+.2f}%")
print("Done.")
