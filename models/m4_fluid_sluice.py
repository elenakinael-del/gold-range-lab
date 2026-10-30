"""IV. Fluid dynamics: viscous damping and pressure build-up behind a sluice gate.
A particle fluid is injected into a chamber (inventory accumulation). Pressure force = -grad(rho), strong viscous drag keeps
velocity ~ 0 while pressure rises. When the gate opens (range break) the stored pressure converts to kinetic energy through a
converging nozzle (Bernoulli) and becomes a jet = trend propagation. Inflow, drag and gate timing are calibrated from live data."""
import numpy as np
from scipy.ndimage import gaussian_filter
from core.render import *

NX, NY = 52, 14
XL, XR = -3.2, 4.2

class Model:
    slug = "04_fluid_pressure_sluice"
    def __init__(s, M):
        s.M = M; s.r = np.random.default_rng(4); s.N = 9000
        s.p = np.zeros((s.N, 3)); s.v = np.zeros((s.N, 3)); s.on = np.zeros(s.N, bool)
        s.tg = 11.5; s.gamma = 2.6; s.inflow = int(55 + 90 * M["pressure"]); s.dirn = M["bias"]; s.rho_ref = 1.0

    def R(s, x):
        x = np.asarray(x); n = np.where(x < 0, 1.0, np.where(x < 1.3, .36 + .64 * .5 * (1 + np.cos(np.pi * x / 1.3)), .36 + .09 * (x - 1.3)))
        return n

    def density(s):
        a = s.p[s.on]; ix = np.clip(((a[:, 0] - XL) / (XR - XL) * NX).astype(int), 0, NX - 1)
        iy = np.clip(((a[:, 1] + 1) / 2 * NY).astype(int), 0, NY - 1); iz = np.clip(((a[:, 2] + 1) / 2 * NY).astype(int), 0, NY - 1)
        rho = np.bincount((ix * NY + iy) * NY + iz, minlength=NX * NY * NY).reshape(NX, NY, NY).astype(float)
        return gaussian_filter(rho, 1.4), (ix, iy, iz)

    def frame(s, sc, t):
        dt = .035; open_ = smooth(t, s.tg, s.tg + 1.2)
        for _ in range(2):
            free = np.where(~s.on)[0]; nin = min(len(free), int(s.inflow * (1 if t < s.tg else .55)))
            if nin:
                i = free[:nin]; a = s.r.uniform(0, 2 * np.pi, nin); rr = np.sqrt(s.r.random(nin)) * .95
                s.p[i] = np.c_[np.full(nin, XL + .05) + s.r.random(nin) * .3, rr * np.cos(a), rr * np.sin(a)]; s.v[i] = [.8, 0, 0]; s.on[i] = True
            rho, (ix, iy, iz) = s.density(); gx, gy, gz = np.gradient(rho)
            act = np.where(s.on)[0]; F = -np.stack([gx[ix, iy, iz], gy[ix, iy, iz], gz[ix, iy, iz]], 1) * 5.2 / (NX / (XR - XL)) * 1.8
            # nozzle makes outflow possible only when gate is open
            rn = rho[ix, iy, iz] / (rho.max() + 1e-9); xa = s.p[act, 0]
            F[:, 0] += open_ * (9.0 * rn * (xa < 1.8) + 2.2 * (xa >= 1.8))
            drag = np.where(s.p[act, 0] < 1.3, s.gamma * (1 - .75 * open_), .25)[:, None]
            s.v[act] += (F - drag * s.v[act]) * dt + .25 * np.sqrt(dt) * s.r.standard_normal((len(act), 3)); s.p[act] += s.v[act] * dt
            x = s.p[act, 0]; g = x > -.04
            if open_ < .02: s.p[act[g], 0] = -.04; s.v[act[g], 0] = -np.abs(s.v[act[g], 0]) * .3
            else:
                blocked = (x > -.04) & (x < .06) & (np.hypot(s.p[act, 1], s.p[act, 2]) > .36 + .64 * open_)
                s.p[act[blocked], 0] = -.04; s.v[act[blocked], 0] *= -.2
            s.v[act, 0] += np.where(s.p[act, 0] > 0, 1.6 * open_ * (s.p[act, 0] < 1.8), 0) * dt * (rho[ix, 0 * ix + NY // 2, 0 * ix + NY // 2][np.isin(np.arange(len(s.on)), act)[act]] * 0 + 1) * 0
            x = s.p[act, 0]; rp = np.hypot(s.p[act, 1], s.p[act, 2]); Rl = s.R(x); out = rp > Rl
            if out.any():
                n = s.p[act[out], 1:] / rp[out, None]; s.p[act[out], 1:] = n * Rl[out, None] * .995
                vr = (s.v[act[out], 1:] * n).sum(1); s.v[act[out], 1:] -= 1.2 * np.maximum(vr, 0)[:, None] * n
            gone = act[s.p[act, 0] > XR - .2]; s.on[gone] = False
        act = np.where(s.on)[0]; speed = np.clip(s.v[act, 0] / 3.0, 0, 1); rho_i = rho[ix, iy, iz] if len(ix) == len(act) else np.zeros(len(act))
        press = float(rho[: int(NX * (-XL) / (XR - XL)) - 2].max() / 1.0); s.rho_ref = max(s.rho_ref, press * .05 + .5)
        pn = np.clip(rho_i / (np.percentile(rho_i, 98) + 1e-6) if len(rho_i) else rho_i, 0, 1)
        sc.cam.update(az=np.radians(-22 + 30 * smooth(t, 0, 20)), el=np.radians(14 + 10 * smooth(t, 0, 20)), dist=11.5 - 1.0 * smooth(t, 0, 20), t=np.array([.5, 0, 0]))
        sc.begin(persist=.78, exposure=1.8); sc.background_dust(t)
        sc.pts(s.p[act], HEAT(.18 + .5 * pn + .45 * speed ** .7) * (.65 + .35 * speed)[:, None], 1.7, trail=True)
        # channel wireframe (rings glow with local pressure)
        xs = np.linspace(XL, XR - .2, 62); th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        loc = np.interp(xs, np.linspace(XL, XR, NX), rho.reshape(NX, -1).mean(1)); loc = loc / (loc.max() + 1e-6)
        for xx, lv in zip(xs, loc):
            r_ = float(s.R(xx)); sc.pts(np.stack([np.full_like(th, xx), r_ * np.cos(th), r_ * np.sin(th)], 1), HEAT(.2 + .6 * lv), .14 + .35 * lv)
        for a in th[::9]:
            L = np.stack([xs, s.R(xs) * np.cos(a), s.R(xs) * np.sin(a)], 1); sc.pts(densify(L[None], 3)[0], ICE(.45), .25)
        gate = 1 - open_; gg = np.linspace(0, 2 * np.pi, 120); rg = np.sqrt(np.linspace(0, 1, 14))[:, None] * (1 - .64 * open_ * 0 + 0)
        if gate > .02:
            for rr in np.linspace(.1, 1, 9): sc.pts(np.stack([np.full(120, 0.0), rr * np.cos(gg), rr * np.sin(gg)], 1), GOLD(.8), .35 * gate)
        elif open_ < 1: pass
        sc.label3d([XL, 0, -1.55], "INVENTORY  (range)", 17); sc.label3d([0, 0, -1.55], f"GATE  {s.M['hi'] if s.dirn > 0 else s.M['lo']:,.1f}", 17); sc.label3d([3.4, 0, -1.2], "TREND JET", 17)
        vj = float(np.percentile(s.v[act, 0], 97)) if len(act) else 0.0; P0 = press
        ph = [(0, 5, "I.  Inventory accumulation — large players load the chamber"), (5, 11.5, "II.  Viscous damping: velocity ≈ 0 while pressure builds"),
              (11.5, 14, "III.  The gate opens — range boundary breaks"), (14, 20.01, "IV.  Pressure converts to kinetic energy — trend propagation")]
        rd = [f"Chamber pressure index = {P0:5.2f}     jet velocity v = {vj:4.2f}     drag γ = {s.gamma:3.1f}     gate aperture = {100 * open_:3.0f}%",
              f"Live data:  inventory-pressure index {s.M['pressure']:.2f}  ·  compression {s.M['compression']:.2f}  ·  CVD {s.M['cvd']:+.2f}  ·  release direction {'up' if s.dirn > 0 else 'down'}"]
        hud(sc, s.M, "IV.  Fluid Dynamics  ·  Viscous Damping  ·  Pressure Release", ph, t, rd)
        return sc.finish(t / 20)
