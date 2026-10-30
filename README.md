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


___________________

The math has two layers: the statistics measured on real gold candles, and the equations that drive the five animations.

## Layer 1: measured from the candles

**Range and position.** The range is `lo = min(Low)` and `hi = max(High)` over the last 288 bars. Position is `(price − lo) / (hi − lo)`, so 0 is the low and 1 is the high.

**Volume profile and POC.** Each bar's typical price is `(H + L + C) / 3`. These are binned into 48 slices of the range, weighted by volume, lightly smoothed, and normalised so they sum to 1. That gives `p(x)`, and the tallest bin is the POC.

**Kramers index.** `ln(p_POC / p_edge)`, where `p_edge` is the average of the 4 bins at that edge. A bigger number means the edge is much "thinner" than the centre, so it is a taller barrier in the analogy.

**Volatility.** These are two high/low-based estimators:
- Parkinson: `σ = √( mean[ln(H/L)²] / (4 ln 2) )`
- Garman-Klass: `σ = √( mean[ 0.5·ln(H/L)² − (2 ln 2 − 1)·ln(C/O)² ] )`
- Vol ratio: Parkinson over the last 24 bars divided by Parkinson over the whole range. Below 1 is calmer than the range average, above 1 is hotter.

**CVD bias.** `Σ sign(C − O) · Volume / Σ Volume` over the last 96 bars. It runs from −1 to +1. It is a candle-based proxy, not true bid/ask delta.

**Permutation entropy.**
1. Take 4 consecutive returns and record their ordering (24 possible patterns).
2. Count how often each pattern `pᵢ` appears over 96 returns.
3. Compute `S = −Σ pᵢ ln pᵢ / ln 24`.

0 means very orderly and 1 means pure noise. It measures randomness, not direction. The percentile is how many of the rolling values are at or below the current one.

**Hurst exponent.** Price changes over a lag `k` scale roughly as `std(P(t+k) − P(t)) ∝ k^H`. The code takes the slope of log(std) against log(lag) for lags up to 24:
- H below 0.5 means mean-reverting.
- H around 0.5 means random walk.
- H above 0.5 means persistent.

**Derived scores.**
- Compression: `clip(1 − vol_ratio, 0, 1)`.
- Pressure: `0.55·compression + 0.45·min(1, 3·|CVD|)`.
- Diffusion coefficient: `(std(returns) / ln(hi/lo))² / 2`.

**Regime rules.** Expansion needs H above 0.58 and vol ratio above 1.2. Compression needs entropy percentile below 35% and vol ratio below 0.95. Anything else is "balanced".

## Layer 2: the physics behind each animation

1. **Potential well.** Price follows the Langevin equation `dX = −∇U dt + √(2D) dW`. The potential is `U = −V₀ e^{−r²/2w²} − τx + …`, where `V₀` is the well depth, `w` its width and `τ` the order-flow tilt. The well floor is also ruffled by the volume profile, using `U = −D ln p`. The well disappears when `τ > V₀ / (w√e)`. Escape speed follows Kramers: `rate ≈ (ω/2π)·e^{−ΔU/D}`, so a lower barrier or more noise means exponentially faster escape.
2. **Entropy.** Each particle is pulled toward a lattice site by a spring `k`, with friction `γ` and thermal noise set by temperature `T`: `dv = [−k(x − x_site) − γv]dt + noise`. Higher `T` weakens `k`, so the lattice melts. The order parameter is `φ = mean(e^{−|x − x_site|²/a²})` and the disorder is `S = 1 − φ`.
3. **Diffusion.** Liquidity density `u` follows `∂u/∂t = ∇·(D·permeability·∇u) − v ∂u/∂x`. The range edges have near-zero permeability (about 0.02), so they reflect. Wall stress is the density jump across the wall. At the scripted "break" the permeability rises and a drift `v` carries the density outward.
4. **Fluid.** Particles feel a pressure force `−∇ρ` pushing from dense to sparse regions, plus drag `γ`. With the gate shut, drag keeps speed near zero while density rises. When it opens, a Bernoulli-style drive through the narrowing nozzle turns stored pressure into jet speed.
5. **Fractal.** Each filament's increments follow `dz_t = φ·dz_{t−1} + σε`. For this process, `VR(2) = 1 + φ`, so `H = 0.5·(1 + log₂(1 + φ))`, or inversely `φ = 2^{2H−1} − 1`. Negative `φ` gives tangled mean-reversion and positive `φ` gives trending filaments. The H shown on screen is measured on the simulated ensemble with the same lag-scaling estimator.

## Limits to keep in mind

- The weights in the pressure score (0.55/0.45) and the regime thresholds are my heuristics, not fitted or backtested.
- Hurst and entropy are noisy on short windows.
- The physics is analogy. Real price does not obey these equations, and the dashboard describes state without forecasting.

__________________________________________________

## The five models
| # | Model | Idea |
|---|-------|------|
| I | Potential well · Kramers escape | Price as a particle in a well shaped by the volume profile |
| II | Entropy · phase transition | Order builds in the range and is released at breakout |
| III | Diffusion · boundary stress | Liquidity spreads while resting orders act as walls |
| IV | Fluid · sluice gate | Inventory pressure builds behind a gate, then releases |
| V | Hurst · fractal regime | Mean-reverting to trending persistence flip |

## What it measures (from live 5m candles)
Range and position, volume profile / POC, Parkinson and Garman-Klass volatility, candle-based CVD, permutation entropy, Hurst exponent, and a rule-based regime label (compression / balanced / expansion).

## Quick start
```bash
pip install -r requirements.txt
python run_all.py          # render the five 20 s MP4s on live data
python -m dashboard.app    # live dashboard at localhost:8050
```

## Limits (please read)
- The physics is a **visual analogy**, not a claim that markets behave like fluids or particles.
- Video "breakouts" are illustrative sequences, not predictions. The dashboard describes the current state and does not forecast.
- CVD is a candle-based proxy, not true bid/ask delta. Pressure weights and regime thresholds are heuristics, not backtested.
- Yahoo data is delayed. Not financial advice.

## Notes
- Text in all visuals is Times New Roman, white (falls back to Liberation Serif if TNR isn't installed). Override with env `TNR_PATH`.
- Yahoo limits: 1m ≤ 7 days, 5m/15m/30m ≤ 60 days; GC=F is delayed ~10 min. Metrics are descriptive analytics, not trading signals.
