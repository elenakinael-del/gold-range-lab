"""II. Thermodynamics & information entropy: the range as a cooling lattice that stores latent volatility.
Particles are an overdamped Langevin gas bound to lattice sites by a temperature-dependent spring.
Temperature is modulated by the REAL rolling permutation entropy of the live series; order parameter = entropy collapse.
At the end the lattice releases its latent heat as a shock front along the price axis."""
import numpy as np
from core.render import *

class Model:
    slug = "02_entropy_phase_transition"
    def __init__(s, M):
        s.M = M; s.r = np.random.default_rng(5); n = 14
        g = np.linspace(-1, 1, n); L = np.stack(np.meshgrid(g, g, np.linspace(-1, 1, 13), indexing="ij"), -1).reshape(-1, 3)
        s.N = len(L); s.site = L[s.r.permutation(s.N)]; s.x = s.r.uniform(-1, 1, (s.N, 3)); s.v = s.r.normal(0, .3, (s.N, 3))
        e = np.array(M["entropy_series"]); s.en = (e - e.min()) / (np.ptp(e) + 1e-9); s.hist = []; s.dirn = M["bias"]
        s.tb = 13.5

    def T(s, t):
        base = .06 + .94 * (1 - smooth(t, 1.5, 11.5)) ** 1.6
        k = s.en[int(np.clip(t / s.tb, 0, .999) * len(s.en))]
        return base * (1 + .35 * (k - .5))

    def frame(s, sc, t):
        T = s.T(t); dt = .02; rel = smooth(t, s.tb, s.tb + 1.2)
        k = 55 * (1 - smooth(T, .08, .7)) ** 1.3 * (1 - rel) + .4
        for _ in range(3):
            a = -k * (s.x - s.site) - 2.2 * s.v
            a[:, 2] += rel * 5.0 * s.dirn * (1 + .6 * s.r.random(s.N)) * (s.x[:, 2] * s.dirn < 2.6 + 0 * s.N)
            a[:, :2] += rel * .6 * s.x[:, :2]
            s.v += a * dt + np.sqrt(2 * 2.2 * T * dt) * s.r.standard_normal(s.v.shape); s.x += s.v * dt
            if rel < .02:
                hit = np.abs(s.x) > 1.05; s.x[hit] = np.clip(s.x[hit], -1.05, 1.05); s.v[hit] *= -.5
        ke = np.clip((s.v ** 2).sum(1) / 3.2, 0, 1); phi = float(np.mean(np.exp(-((s.x - s.site) ** 2).sum(1) / .1 ** 2) ))
        S = 1 - phi; s.hist.append(S)
        sc.cam.update(az=np.radians(-30 + 55 * smooth(t, 0, 20)), el=np.radians(22 + 8 * np.sin(t / 3)), dist=11.5 + 5.5 * smooth(t, s.tb, 19), t=np.array([0, 0, 1.6 * smooth(t, s.tb, 20) * s.dirn]))
        sc.begin(persist=.8, exposure=2.0); sc.background_dust(t)
        col = ICE(.38 + .38 * (1 - ke) ** 2) * (1 - ke)[:, None] + HEAT(.45 + .55 * ke) * ke[:, None]
        sc.pts(s.x * [1.4, 1.4, 1.4], col, 3.0, trail=True)
        # range container: fades when it is shattered
        a = 1 - rel
        if a > .02:
            c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * 1.45
            for i in range(8):
                for j in range(i + 1, 8):
                    if np.sum(np.abs(c[i] - c[j]) > 0) == 1: sc.pts(polyline(np.array([c[i], c[j]]), .03), (.35, .55, 1), .22 * a)
        if t > s.tb:   # latent-heat shock front
            z = -1.5 * s.dirn + (t - s.tb) * 1.5 * s.dirn; th = np.linspace(0, 2 * np.pi, 360)
            for rr, w in ((1.5, 1.2), (1.85, .5), (2.3, .25)):
                sc.pts(np.stack([rr * np.cos(th), rr * np.sin(th), np.full_like(th, z * (1 if s.dirn > 0 else 1))], 1), HEAT(.9), w * float(1 - smooth(t, s.tb + 3, s.tb + 6.5)), trail=True)
        for sz, nm, val in ((-1, "RANGE LOW", s.M["lo"]), (1, "RANGE HIGH", s.M["hi"])):
            sc.label3d([-2.6, -1.9, sz * 1.45], f"{nm}  {val:,.1f}", 17, alpha=a)
        # entropy meter (screen space, minimal L-axes)
        mx, my, mw, mh = W - 330, 150, 250, 110
        sc.screen(np.linspace(mx, mx, 40), np.linspace(my, my + mh, 40), (.7, .75, 1), .25); sc.screen(np.linspace(mx, mx + mw, 80), np.full(80, my + mh), (.7, .75, 1), .25)
        hs = np.array(s.hist[::2]); hx = mx + mw * np.arange(len(hs)) / (10 * 24 / 1.0 * 2 * 1.0) * 1.0 / 1.0
        hx = mx + mw * (np.arange(len(hs)) * 2 / 480); sc.screen(hx, my + mh - mh * np.clip(hs, 0, 1) * .98, HEAT(.8), 1.3)
        sc.text(mx + mw, my - 8, "configurational entropy  S(t)", 15, anchor="ra", style="i", alpha=.85)
        ph = [(0, 4.5, "I.  Hot, disordered order flow — high entropy gas"), (4.5, 12.5, "II.  Cooling: entropy collapses, liquidity crystallises on volume nodes"),
              (12.5, 14.5, "III.  Critical temperature — latent volatility released"), (14.5, 20.01, "IV.  Phase change: the lattice shatters and propagates")]
        rd = [f"Order parameter φ = {phi:4.2f}     S/S₀ = {S:4.2f}     bath temperature T = {T:4.2f}",
              f"Live permutation entropy {s.M['entropy']:.3f}  (percentile {100 * s.M['entropy_pct']:.0f} of window)  ·  Parkinson vol ratio {s.M['vol_ratio']:.2f}  ·  latent-volatility index {s.M['compression']:.2f}"]
        hud(sc, s.M, "II.  Thermodynamics  ·  Entropy Collapse  ·  Latent Volatility", ph, t, rd)
        return sc.finish(t / 20)
