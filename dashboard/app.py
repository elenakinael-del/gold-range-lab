"""Live dashboard server.   python -m dashboard.app   ->  http://localhost:8050
Production / public link:  gunicorn dashboard.app:app   (see README for Render / Railway one-click deploy)."""
import os, sys, time, math, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, jsonify, request, send_file
from core.data import load_gold
from core.metrics import compute_all

app = Flask(__name__)
HERE = os.path.dirname(os.path.abspath(__file__))
RANGE = {"1m": 720, "2m": 360, "5m": 288, "15m": 96, "30m": 64}
PERIOD = {"1m": "5d", "2m": "5d", "5m": "5d", "15m": "5d", "30m": "10d"}
TTL, _cache, _lock = 40, {}, threading.Lock()

def clean(o):
    if isinstance(o, float): return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    return o

@app.route("/")
def index(): return send_file(os.path.join(HERE, "index.html"))

@app.route("/api/analytics")
def analytics():
    iv = request.args.get("interval", "5m"); iv = iv if iv in RANGE else "5m"
    with _lock:
        hit = _cache.get(iv)
        if hit and time.time() - hit[0] < TTL: return jsonify(hit[1])
        df, src = load_gold(interval=iv, period=PERIOD[iv])
        M = clean(compute_all(df, src, iv, RANGE[iv])); M["server_time"] = time.strftime("%H:%M:%S UTC", time.gmtime())
        _cache[iv] = (time.time(), M)
    return jsonify(M)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"\n  Gold Range Lab  ->  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
