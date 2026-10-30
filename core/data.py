"""Live gold futures loader (yfinance) with a physically-plausible synthetic fallback."""
import numpy as np, pandas as pd
from .config import SYMBOL, INTERVAL, PERIOD


def synthetic_gold(n=1500, seed=11, start=4000.0):
    """Range-bound OU price with volatility compression, used only if Yahoo is unreachable."""
    r = np.random.default_rng(seed)
    vol = 0.00055 * (1 - 0.5 * np.exp(-((np.arange(n) - n * 0.55) / (n * 0.22)) ** 2))
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] - 0.012 * x[i - 1] + vol[i] * r.standard_normal()
    c = start * np.exp(x)
    o = np.r_[c[0], c[:-1]]
    spread = np.abs(r.standard_normal(n)) * vol * start * 0.8
    h, l = np.maximum(o, c) + spread, np.minimum(o, c) - spread
    v = r.lognormal(6.2, 0.45, n)
    idx = pd.date_range(end=pd.Timestamp.now("UTC").floor("5min"), periods=n, freq="5min")
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx)


def load_gold(interval=INTERVAL, period=PERIOD, symbol=SYMBOL):
    """Returns (DataFrame OHLCV, source_label)."""
    try:
        import yfinance as yf
        df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(df) < 300:
            raise ValueError("not enough bars")
        return df, f"{symbol} · yfinance live"
    except Exception as e:  # offline / rate limited
        print(f"[data] yfinance unavailable ({e}); using synthetic gold series")
        return synthetic_gold(), f"{symbol} · synthetic fallback"
