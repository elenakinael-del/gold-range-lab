"""Quantitative layer: everything the five physics models are calibrated on."""
import math
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
from .config import RANGE_BARS


def parkinson(h, l):
    return float(np.sqrt(np.mean(np.log(h / l) ** 2) / (4 * np.log(2))))


def garman_klass(o, h, l, c):
    v = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return float(np.sqrt(max(np.mean(v), 0)))


def hurst(logp, lags=(2, 3, 4, 6, 8, 12, 16, 24)):
    """Scaling exponent of std(x[t+k]-x[t]) ~ k^H (aggregated-variance estimator)."""
    lags = [k for k in lags if k < len(logp) // 3]
    sd = np.array([np.std(logp[k:] - logp[:-k]) for k in lags])
    return float(np.clip(np.polyfit(np.log(lags), np.log(sd + 1e-12), 1)[0], 0.05, 0.95))


def perm_entropy(x, m=4):
    """Normalised Bandt-Pompe permutation entropy, 0 = fully ordered, 1 = white noise."""
    w = swv(np.asarray(x), m)
    code = (np.argsort(w, axis=1) * (m ** np.arange(m))).sum(1)
    _, c = np.unique(code, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / math.log(math.factorial(m)))


def rolling(fn, x, win, step):
    idx = np.arange(win, len(x) + 1, step)
    return idx, np.array([fn(x[i - win:i]) for i in idx])


def volume_profile(df, lo, hi, bins=48):
    tp = ((df.High + df.Low + df.Close) / 3).values
    v = df.Volume.values.astype(float) + 1.0
    p, edges = np.histogram(tp, bins=bins, range=(lo, hi), weights=v)
    p = np.convolve(p, np.array([1, 2, 3, 2, 1]) / 9, mode="same") + 1e-9
    p = p / p.sum()
    u = -np.log(p / p.max() + 1e-3)           # Boltzmann inversion  U = -D ln p
    return p, np.clip(u, 0, 4.5) / 4.5, 0.5 * (edges[1:] + edges[:-1])


def compute_all(df, source="", interval="", range_bars=None):
    n = min(range_bars or RANGE_BARS, len(df))
    rg = df.tail(n)
    lo, hi = float(rg.Low.min()), float(rg.High.max())
    close = df.Close.values
    price = float(close[-1])
    pos = (price - lo) / (hi - lo + 1e-9)
    logp = np.log(close[-n:])
    ret = np.diff(logp)

    h_now = hurst(logp[-192:])
    entropy_window = min(96, len(ret))
    ent_idx, ent_ser = rolling(perm_entropy, ret, entropy_window, 6)
    ent_now = float(ent_ser[-1])
    ent_pct = float((ent_ser <= ent_now).mean())
    hurst_window = min(160, len(logp))
    hu_idx, hu_ser = rolling(hurst, logp, hurst_window, 8)

    pk_long = parkinson(rg.High.values, rg.Low.values)
    pk_short = parkinson(rg.High.values[-24:], rg.Low.values[-24:])
    gk_short = garman_klass(rg.Open.values[-24:], rg.High.values[-24:], rg.Low.values[-24:], rg.Close.values[-24:])
    vol_ratio = pk_short / (pk_long + 1e-12)

    sgn = np.sign(rg.Close.values - rg.Open.values)
    cvd = float((sgn * rg.Volume.values)[-96:].sum() / (rg.Volume.values[-96:].sum() + 1e-9))

    p, u, centers = volume_profile(rg, lo, hi)
    poc = float(centers[p.argmax()])
    k_up = float(np.log(p.max() / p[-4:].mean()))   # Kramers index = dU/D toward range high
    k_dn = float(np.log(p.max() / p[:4].mean()))
    comp = float(np.clip(1 - vol_ratio, 0, 1))
    pressure = float(np.clip(0.55 * comp + 0.45 * min(1, 3 * abs(cvd)), 0, 1))
    D_norm = float((np.std(ret) / (np.log(hi / lo) + 1e-9)) ** 2 / 2)   # diffusion coeff in range units

    if h_now > 0.58 and vol_ratio > 1.2:
        regime = "EXPANSION"
    elif ent_pct < 0.35 and vol_ratio < 0.95:
        regime = "COMPRESSION"
    else:
        regime = "BALANCED"

    return dict(
        source=source, interval=interval, n_bars=int(len(df)), n_range=int(n),
        price=price, prev=float(close[-2]), lo=lo, hi=hi, mid=0.5 * (lo + hi), poc=poc, pos=float(pos),
        hurst=h_now, entropy=ent_now, entropy_pct=ent_pct, parkinson=pk_short, parkinson_long=pk_long,
        gk=gk_short, vol_ratio=float(vol_ratio), cvd=cvd, bias=1.0 if cvd >= 0 else -1.0,
        kramers_up=k_up, kramers_dn=k_dn, pressure=pressure, compression=comp, D_norm=D_norm, regime=regime,
        profile=p.tolist(), profile_u=u.tolist(), profile_x=centers.tolist(),
        entropy_series=ent_ser.tolist(), hurst_series=hu_ser.tolist(),
        t0=str(df.index[0]), t1=str(df.index[-1]),
        ohlc=[[str(i), float(o), float(h), float(l), float(c)] for i, o, h, l, c in
              zip(df.index[-200:], df.Open.values[-200:], df.High.values[-200:], df.Low.values[-200:], df.Close.values[-200:])],
    )
