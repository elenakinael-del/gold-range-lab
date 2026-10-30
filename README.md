# Gold Range Lab — physics models of what happens *inside* a range

Five physics frameworks mapped onto gold futures (GC=F) accumulation / distribution, each as a 20 s 3D MP4,
plus a live dashboard that re-runs all five on the latest yfinance prices.

| # | Model | File | Video |
|---|-------|------|-------|
| I   | Potential well · Langevin · Kramers escape | `models/m1_potential_well.py`   | `videos/01_potential_well_kramers.mp4` |
| II  | Thermodynamics · entropy collapse · latent volatility | `models/m2_entropy_phase.py`    | `videos/02_entropy_phase_transition.mp4` |
| III | Heat / Fick diffusion · resting-order wall stress | `models/m3_diffusion_walls.py`  | `videos/03_diffusion_boundary_stress.mp4` |
| IV  | Fluid dynamics · viscous damping · sluice-gate pressure | `models/m4_fluid_sluice.py`     | `videos/04_fluid_pressure_sluice.mp4` |
| V   | Fractal dynamics · Hurst collapse · regime switch | `models/m5_hurst_fractal.py`    | `videos/05_hurst_fractal_regime.mp4` |

## Setup (VS Code)
```
python -m venv .venv && .venv\Scripts\activate      # mac/linux: source .venv/bin/activate
pip install -r requirements.txt
python run_all.py            # all five MP4s with LIVE yfinance data  (~2 min each on one CPU core)
python run_all.py 1 4        # only models 1 and 4
python run_all.py --preview 14 3   # a single PNG still at t=14 s for look-dev
python -m dashboard.app      # live dashboard -> http://localhost:8050
```
Run/debug entries are pre-configured in `.vscode/launch.json`. Resolution/FPS/range length/brand watermark: `core/config.py`.

**The MP4s in `/videos` were rendered in a sandbox with no Yahoo access, so they use a synthetic gold series (labelled on-screen).
Run `python run_all.py` on your machine to re-render them on real, current GC=F data.**

## What is calibrated on live data
- **Range**: high/low of the last `RANGE_BARS` bars (`core/config.py`); price position inside it.
- **I**: floor of the well = Boltzmann-inverted volume profile (`U = −D ln p`); noise D from Parkinson vol ratio; tilt direction from signed-volume CVD; Kramers index `ln(p_POC / p_edge)` each side.
- **II**: temperature modulated by rolling permutation entropy (Bandt–Pompe, m=4); latent-volatility index = compression.
- **III**: diffusion coefficient from return variance in range units; the live price is the heat source in the dashboard; wall leak rises as price tests an edge.
- **IV**: inflow ∝ inventory-pressure index (compression + |CVD|); gate aperture follows volatility expansion.
- **V**: persistence φ from live Hurst (`φ = 2^(2H−1) − 1`); H on screen is *measured* on the simulated ensemble with the same aggregated-variance estimator used on gold.
- The act structure of each video (e.g. the breakout at ~12 s) is a **scenario illustration** of the mechanism, not a prediction. The dashboard, by contrast, runs continuously from the current state.

## Public link
`python -m dashboard.app` serves on your LAN (`0.0.0.0:8050`). For a public URL: push this folder to GitHub and deploy on
Render / Railway / Fly (build `pip install -r requirements.txt`, start `gunicorn dashboard.app:app`; the `Procfile` is included), or tunnel locally with `ngrok http 8050` / `cloudflared tunnel --url http://localhost:8050`.

## Notes
- Text in all visuals is Times New Roman, white (falls back to Liberation Serif if TNR isn't installed). Override with env `TNR_PATH`.
- Yahoo limits: 1m ≤ 7 days, 5m/15m/30m ≤ 60 days; GC=F is delayed ~10 min. Metrics are descriptive analytics, not trading signals.
