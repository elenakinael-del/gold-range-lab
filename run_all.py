"""python run_all.py            -> renders all five 20 s MP4s into /videos using LIVE yfinance data
   python run_all.py 1 3        -> only models 1 and 3
   python run_all.py --preview 14 1   -> one PNG still at t=14 s (fast look-dev)"""
import sys, importlib, time
from core.data import load_gold
from core.metrics import compute_all
from core.render import render
from core.config import INTERVAL

MODELS = {1: "m1_potential_well", 2: "m2_entropy_phase", 3: "m3_diffusion_walls", 4: "m4_fluid_sluice", 5: "m5_hurst_fractal"}
if __name__ == "__main__":
    a = sys.argv[1:]; prev = None
    if "--preview" in a: i = a.index("--preview"); prev = float(a[i + 1]); a = a[:i] + a[i + 2:]
    pick = [int(x) for x in a] or list(MODELS)
    df, src = load_gold(); M = compute_all(df, src, INTERVAL); print(f"data: {src}  bars={len(df)}  range {M['lo']:.1f}-{M['hi']:.1f}  H={M['hurst']:.2f}")
    for k in pick:
        t0 = time.time(); mod = importlib.import_module(f"models.{MODELS[k]}")
        print("->", render(mod.Model, M, preview=prev), f"({time.time() - t0:.0f}s)")
