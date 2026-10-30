"""V. Fractal dynamics: Hurst-exponent collapse and trend re-ignition.
An ensemble of 3-D filaments is driven by AR(1) increments  dz_t = phi dz_{t-1} + sigma eps  (variance ratio VR(2)=1+phi, H = 0.5(1+log2(1+phi))).
Range = anti-persistent (H<0.5) tangle. Breakout = persistence flips, H climbs through 0.65 and filaments fan out into a trend.
H(t) shown is MEASURED on the ensemble with the same aggregated-variance estimator used on the live gold series."""
import numpy as np
from core.render import *

L = 70
class Model:
    slug = "05_hurst_fractal_regime"
    def __init__(s, M):
        s.M = M; s.r = np.random.default_rng(12); s.P = 300; s.dirn = M["bias"]
        s.pos = np.zeros((s.P, 3)); s.inc = np.zeros((s.P, 3)); s.buf = np.zeros((L, s.P, 3)); s.tb = 12.0
        s.zs = []; s.H = .5; s.Hs = []

    def phi(s, t):
        a = -.16 * smooth(t, 1, 5); b = .34 * smooth(t, s.tb, s.tb + 2.2)
        return a + (b - a * smooth(t, s.tb, s.tb + 2.2)) if t > s.tb else a

    def frame(s, sc, t):
        ph = s.phi(t); br = smooth(t, s.tb, s.tb + 2.2); sig = .026 * (1 + .9 * br)
        for _ in range(2):
            eps = s.r.standard_normal((s.P, 3)); s.inc = ph * s.inc + sig * eps
            s.inc[:, 2] += .030 * br * s.dirn; s.inc[:, :2] *= (1 - .4 * br)
            s.pos += s.inc - (.05 * (1 - br)) * s.pos * np.array([1, 1, 1.0])
            s.buf = np.roll(s.buf, 1, 0); s.buf[0] = s.pos; s.zs.append(s.pos[:, 2].copy())
        z = np.array(s.zs[-90:])   # measured H via aggregated variance over lags on the ensemble
        if len(z) > 40:
            lags = [1, 2, 3, 4, 6, 8, 12]; sd = [np.sqrt(np.mean((z[k:] - z[:-k]) ** 2)) for k in lags]
            s.H = float(np.clip(np.polyfit(np.log(lags), np.log(np.array(sd) + 1e-9), 1)[0], .05, .95))
        s.Hs.append(s.H)
        zc = float(s.pos[:, 2].mean())
        sc.cam.update(az=np.radians(-35 + 60 * smooth(t, 0, 20)), el=np.radians(18 + 6 * np.sin(t / 4)), dist=8.0 + 7.0 * smooth(t, s.tb, 19), t=np.array([0, 0, .8 * zc]))
        sc.begin(persist=.0, exposure=1.35); sc.background_dust(t)
        age = np.linspace(1, 0, L)[:, None]
        pts = np.transpose(s.buf, (1, 0, 2))                 # (P, L, 3)
        pts = densify(pts, 2); n = pts.shape[1]
        a = np.linspace(1, 0, n)                              # head -> tail
        cvec = ICE(.28 + .6 * a ** 1.4) * (1 - br) + HEAT(.30 + .70 * a ** 1.1) * br
        sc.pts(pts.reshape(-1, 3), np.tile(cvec, (s.P, 1)), np.tile(.35 + 1.1 * a ** 2.2, s.P))
        sc.pts(s.pos, (1, .97, .9), 1.4)
        # range planes + breakout level
        zz = 1.0; th = np.linspace(0, 2 * np.pi, 220)
        for zl, nm, val, dy in ((-zz, "RANGE LOW", s.M["lo"], -1), (zz, "RANGE HIGH", s.M["hi"], 1)):
            sc.pts(np.stack([1.4 * np.cos(th), 1.4 * np.sin(th), np.full_like(th, zl)], 1), (.55, .7, 1), .30 * (1 - .6 * br))
            sc.label3d([1.55, 0, zl], f"{nm}  {val:,.1f}", 17)
        phs = [(0, 5, "I.  Random-walk onset — H ≈ 0.5"), (5, 12, "II.  Range: anti-persistent filaments, fractal compression (H < 0.5)"),
               (12, 15, "III.  Persistence flips — volatility spike, H climbs"), (15, 20.01, "IV.  H > 0.65: trending regime re-activates")]
        rd = [f"Measured Hurst exponent  H(t) = {s.H:4.2f}     AR persistence φ = {ph:+4.2f}     σ = {sig:4.3f}     regime = {'TREND' if s.H > .6 else 'MEAN-REVERTING' if s.H < .46 else 'RANDOM WALK'}",
              f"Live gold series:  H = {s.M['hurst']:.2f}  ·  Parkinson vol ratio {s.M['vol_ratio']:.2f}  ·  Garman-Klass {100 * s.M['gk']:.3f}%  ·  breakout bias {'up' if s.dirn > 0 else 'down'}"]
        hud(sc, s.M, "V.  Fractal Dynamics  ·  Hurst Exponent Collapse  ·  Regime Switch", phs, t, rd)
        return sc.finish(t / 20)
