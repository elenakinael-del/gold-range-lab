"""I. Potential well · Langevin dynamics · Kramers escape.
Price is an overdamped particle  dX = -grad U dt + sqrt(2D) dW  in a confining well.
The floor of the well is the Boltzmann-inverted live volume profile (U = -D ln p).
Order-flow imbalance tilts the landscape; the barrier thins until the Kramers rate explodes."""
import numpy as np
from core.render import *

class Model:
    slug = "01_potential_well_kramers"
    def __init__(s, M):
        s.M = M; s.r = np.random.default_rng(7); s.N = 1900
        s.p = s.r.normal(0, .25, (s.N, 2)); s.V0, s.w = 1.25, .55
        s.D = .12 + .10 * float(np.clip(M["vol_ratio"], .5, 1.5)); s.dirn = M["bias"]
        s.xs = np.linspace(-1, 1, len(M["profile_u"])); s.prof = np.array(M["profile_u"]); s.prof = s.prof - s.prof.mean()
        s.xg = np.linspace(-1.9, 2.5, 70); s.yg = np.linspace(-1.5, 1.5, 38); s.S = 1.7
        s.X, s.Y = np.meshgrid(s.xg, s.yg, indexing="ij")

    def U(s, x, y, tau):
        r2 = x * x + y * y
        rip = .10 * np.interp(x, s.xs, s.prof) * np.exp(-r2 / (2 * (1.3 * s.w) ** 2))
        return -s.V0 * np.exp(-r2 / (2 * s.w ** 2)) - tau * x + .17 * y * y + rip

    def vz(s, u): return 1.5 * np.where(u < 0, -1.6 * np.log1p(-u / 1.6), u)   # visual compression of the deep downhill slope

    def tau(s, t): return 1.55 * smooth(t, 2.5, 16.5)

    def kramers(s, tau):
        x = np.linspace(-1.2, 2.0, 500); u = s.U(x, 0 * x, tau); i0 = int(np.argmin(u[x < .6]))
        d = np.diff(u[i0:]); up = np.where(d > 0)[0]
        if len(up) == 0: return 0.0
        dn = np.where(d[up[0]:] < 0)[0]
        return 0.0 if len(dn) == 0 else float(u[i0 + up[0] + dn[0]] - u[i0])

    def frame(s, sc, t):
        tau = s.tau(t); h = 1e-3; dt = .012
        for _ in range(4):
            x, y = s.p[:, 0], s.p[:, 1]
            gx = (s.U(x + h, y, tau) - s.U(x - h, y, tau)) / (2 * h); gy = (s.U(x, y + h, tau) - s.U(x, y - h, tau)) / (2 * h)
            s.p += -np.stack([gx, gy], 1) * dt + np.sqrt(2 * s.D * dt) * s.r.standard_normal(s.p.shape)
            gone = s.p[:, 0] > 3.0; s.p[gone] = s.r.normal(0, .22, (gone.sum(), 2))
        x, y = s.p[:, 0], s.p[:, 1]
        sc.cam.update(az=np.radians(-38 + 34 * smooth(t, 0, 20)), el=np.radians(33 - 8 * smooth(t, 8, 20)), dist=12.0 - 1.8 * smooth(t, 0, 7) + 1.2 * smooth(t, 14, 20),
                      t=np.array([1.2, 0, -0.9]))
        sc.begin(persist=.84, exposure=1.15); sc.background_dust(t)
        Z = s.vz(s.U(s.X, s.Y, tau)); gx, gy = np.gradient(Z, s.xg, s.yg); g = np.hypot(gx, gy)
        v = np.clip(.12 + .50 * g / 2.4 + .30 * np.exp(-(s.X ** 2 + s.Y ** 2) / .6), 0, 1)
        P = np.stack([s.X * s.S, s.Y * s.S, Z], -1); C = HEAT(v)
        for A, B in ((P, C), (P.transpose(1, 0, 2), C.transpose(1, 0, 2))):
            D2 = densify(A, 3); Cc = densify(B, 3); sc.pts(D2.reshape(-1, 3), Cc.reshape(-1, 3), (.55 + .6 * v.mean()))
        # live-volume-profile nodes: glowing rings on the well floor
        zz = s.vz(s.U(x, y, tau)) + .06; esc = np.clip((x - 1.0) / 1.5, 0, 1)
        col = ICE(.55 + .3 * (1 - esc)) * (1 - esc)[:, None] + GOLD(.75 + .25 * esc) * esc[:, None]
        sc.pts(np.stack([x * s.S, y * s.S, zz], 1), col, 1.3, trail=True)
        dU = s.kramers(tau); rate = (np.sqrt(s.V0) / s.w / (2 * np.pi)) * np.exp(-dU / s.D) if dU > 0 else float("inf")
        for xx, nm, val in ((-1, "RANGE LOW", s.M["lo"]), (1, "RANGE HIGH", s.M["hi"])):
            zb = s.vz(s.U(np.array([xx]), np.array([-1.0]), tau))[0] - .45
            sc.pts(polyline(np.array([[xx * s.S, -1.7, zb + .35], [xx * s.S, -1.7, zb]]), .02), (.8, .85, 1), .35)
            sc.label3d([xx * s.S, -1.7, zb - .12], f"{nm}  {val:,.1f}", 17)
        phases = [(0, 6.5, "I.  Confinement — price thermalises inside the volume-profile well"), (6.5, 12, "II.  Order-flow imbalance tilts the landscape; the barrier thins"),
                  (12, 17, "III.  Kramers escape — noise overcomes the barrier"), (17, 20.01, "IV.  Directional propagation down the potential gradient")]
        rd = [f"ΔU / D = {dU / s.D:5.2f}     Kramers rate  r = {sci(rate) if rate != float('inf') else 'barrier collapsed'} / step     tilt τ = {tau:4.2f}     noise D = {s.D:4.2f}",
              f"Calibrated on live data:  Kramers index up {s.M['kramers_up']:.2f}  ·  down {s.M['kramers_dn']:.2f}  ·  volume-weighted POC {s.M['poc']:,.1f}  ·  CVD bias {'up' if s.dirn > 0 else 'down'}"]
        hud(sc, s.M, "I.  Potential Well  ·  Langevin Dynamics  ·  Kramers Escape", phases, t, rd)
        return sc.finish(t / 20)
