"""III. Heat / Fick diffusion with resting-order walls.
u(x,y,t) = liquidity density, solved implicitly (backward Euler, operator splitting) with a permeability field.
Walls sit at the range edges with tiny permeability (reflecting). Stress = density jump across the wall.
When passive liquidity is exhausted the boundary flips from Neumann to Dirichlet-like and flow propagates outward."""
import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import map_coordinates
from core.render import *

class Model:
    slug = "03_diffusion_boundary_stress"
    def __init__(s, M):
        s.M = M; s.r = np.random.default_rng(9); s.nx, s.ny = 118, 40
        s.x = np.linspace(-1.5, 3.1, s.nx); s.y = np.linspace(-1.4, 1.4, s.ny); s.dx, s.dy = s.x[1] - s.x[0], s.y[1] - s.y[0]
        px = np.interp(s.x, np.array(M["profile_x"]) * 0 + np.linspace(-1, 1, len(M["profile"])), np.array(M["profile"]) / max(M["profile"]), left=0, right=0)
        s.u = (px[:, None] * np.exp(-s.y[None, :] ** 2 / .55) * 1.0).astype(float); s.u *= (np.abs(s.x) < .94)[:, None]
        s.D = .022 + 1.0 * min(M["D_norm"] * 40, .02); s.dirn = M["bias"]; s.tb = 12.0
        s.N = 1700; s.q = np.stack([s.r.uniform(-.9, .9, s.N), s.r.normal(0, .45, s.N)], 1)
        s.sparks = np.zeros((0, 4)); s.fired = False; s.stress = 0.

    def perm(s, t):
        xf = .5 * (s.x[1:] + s.x[:-1]); eps = .02
        wl = np.exp(-((xf + 1) / .05) ** 2); wh = np.exp(-((xf - 1) / .05) ** 2)
        k = smooth(t, s.tb, s.tb + 2.5)
        return 1 - (1 - eps) * wl - (1 - eps) * wh * (1 - k)

    def step(s, t, dt):
        D = s.D * s.perm(t); v = (smooth(t, s.tb + .5, s.tb + 3) * .9 if True else 0)
        n = s.nx; ab = np.zeros((3, n)); lo = D / s.dx ** 2; up = lo
        diag = np.ones(n); diag[:-1] += dt * up; diag[1:] += dt * lo; diag += dt * v / s.dx
        ab[1] = diag; ab[0, 1:] = -dt * up; ab[2, :-1] = -dt * lo - dt * v / s.dx
        s.u = solve_banded((1, 1), ab, s.u)
        m = s.ny; ay = np.zeros((3, m)); c = s.D * dt / s.dy ** 2; ay[1] = 1 + 2 * c; ay[1, 0] = ay[1, -1] = 1 + c; ay[0, 1:] = -c; ay[2, :-1] = -c
        s.u = solve_banded((1, 1), ay, s.u.T).T
        s.u[-20:] *= (1 - dt * np.linspace(0, 4, 20))[:, None]   # open outflow sink at the far end

    def frame(s, sc, t):
        for _ in range(3): s.step(t, .06)
        i1 = int(np.searchsorted(s.x, 1.0)); s.stress = abs(s.u[i1 - 2:i1 - 1].mean() - s.u[i1 + 3:i1 + 4].mean())
        # random walkers obeying the same permeability
        k = smooth(t, s.tb, s.tb + 2.5); v = smooth(t, s.tb + .5, s.tb + 3) * .9
        new = s.q + np.sqrt(2 * s.D * .18) * s.r.standard_normal(s.q.shape); new[:, 0] += v * .18
        for edge, leak in ((-1, .004), (1, .004 + .996 * k)):
            cross = (np.sign(s.q[:, 0] - edge) != np.sign(new[:, 0] - edge)) & (s.r.random(s.N) > leak); new[cross, 0] = s.q[cross, 0]
        new[:, 1] = np.clip(new[:, 1], -1.4, 1.4); s.q = new; s.q[s.q[:, 0] > 3.0, 0] = 3.0
        if t > s.tb and not s.fired:
            s.fired = True; n = 500; s.sparks = np.stack([np.full(n, 1.0), s.r.uniform(-1.3, 1.3, n), s.r.uniform(0, 1.8, n), s.r.uniform(.4, 2.0, n)], 1)
        sc.cam.update(az=np.radians(-52 + 42 * smooth(t, 0, 20)), el=np.radians(31), dist=12.0 - 2.0 * smooth(t, 0, 10), t=np.array([.9, 0, .6]))
        sc.begin(persist=.8, exposure=1.35); sc.background_dust(t)
        Z = 2.6 * np.clip(s.u, 0, None) ** .6; GX, GY = np.meshgrid(s.x, s.y, indexing="ij"); P = np.stack([GX, GY, Z], -1); C = HEAT(.12 + .88 * np.clip(Z / 2.2, 0, 1))
        for A, B in ((P, C), (P.transpose(1, 0, 2), C.transpose(1, 0, 2))):
            sc.pts(densify(A, 2).reshape(-1, 3), densify(B, 2).reshape(-1, 3), .5)
        # tracers sit on the surface
        zz = map_coordinates(Z, [(s.q[:, 0] - s.x[0]) / s.dx, (s.q[:, 1] - s.y[0]) / s.dy], order=1) + .04
        sc.pts(np.c_[s.q, zz], ICE(.8), 1.5, trail=True)
        # resting-order walls: brightness = stress
        for xe, up, nm, val in ((-1, False, "RANGE LOW", s.M["lo"]), (1, True, "RANGE HIGH", s.M["hi"])):
            st = float(np.clip(s.stress * 3.2, .05, 1)); a = (1 - k) if up else 1.0
            gy, gz = np.meshgrid(np.linspace(-1.4, 1.4, 56), np.linspace(0, 1.9, 22), indexing="ij")
            keep = np.ones_like(gy, bool) if not up else s.r.random(gy.shape) > smooth(t, s.tb, s.tb + 2.0) * .9
            W3 = np.stack([np.full_like(gy, xe), gy, gz], -1)[keep]
            sc.pts(W3, HEAT(.55 + .4 * st), .45 + 1.2 * st * a, trail=False)
            sc.label3d([xe, -1.9, -.15], f"{nm}  {val:,.1f}", 17, alpha=1.0)
        if len(s.sparks):
            s.sparks[:, 0] += s.sparks[:, 3] * .05 * (1 + t - s.tb) * .6; s.sparks[:, 2] += (s.r.random(len(s.sparks)) - .5) * .03
            sc.pts(s.sparks[:, :3], GOLD(.95), 1.6, trail=True)
        ph = [(0, 5, "I.  Initial liquidity landscape — volume-profile peaks"), (5, 11.5, "II.  Diffusion flattens the interior while walls absorb the stress"),
              (11.5, 15, "III.  Passive liquidity exhausted — boundary flips reflecting → absorbing"), (15, 20.01, "IV.  Directional flux propagates beyond the range")]
        rd = [f"Fick:  ∂u/∂t = D ∂²u/∂x² − v ∂u/∂x     D = {s.D:4.3f}     wall stress |Δu| = {s.stress:4.2f}     permeability of range high = {.03 + .97 * k:4.2f}",
              f"Live data:  diffusion coefficient (range units) {s.M['D_norm']:.2e}  ·  volume POC {s.M['poc']:,.1f}  ·  price at {100 * s.M['pos']:.0f}% of range  ·  flux bias {'up' if s.dirn > 0 else 'down'}"]
        hud(sc, s.M, "III.  Diffusion  ·  Resting-Order Walls  ·  Boundary Stress", ph, t, rd)
        return sc.finish(t / 20)
