"""Global settings. Edit here, every model + the dashboard follows."""
SYMBOL   = "GC=F"          # COMEX Gold futures on Yahoo Finance
INTERVAL = "5m"            # 1m / 2m / 5m / 15m / 30m / 60m
PERIOD   = "5d"            # yfinance lookback (1m supports max 7d)
RANGE_BARS = 288           # bars that define the current structural range (288 x 5m ~ 1 session)
W, H, FPS, DUR = 1280, 720, 24, 20
BRAND = "Elena Hysa  ·  Quantitative Research"   # bottom-right watermark in videos ("" to hide)
