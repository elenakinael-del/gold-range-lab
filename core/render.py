"""Tiny cinematic 3D engine: perspective camera, sub-pixel additive splatting, multi-scale bloom,
trail persistence, film grade, Times New Roman HUD, H.264 export. Pure numpy + OpenCV + Pillow."""
import os, math, subprocess, shutil
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from .config import W, H, FPS, DUR, BRAND

# ---------- colour ramps (custom, not library colormaps) ----------
def _ramp(stops):
    pos = np.array([s[0] for s in stops]); col = np.array([s[1] for s in stops], float)
    def f(v):
        v = np.clip(np.asarray(v, float), 0, 1)
        return np.stack([np.interp(v, pos, col[:, i]) for i in range(3)], -1)
    return f

HEAT = _ramp([(0, (.02, 0, .10)), (.2, (.22, .03, .55)), (.42, (.75, .08, .45)), (.62, (1, .38, .08)), (.82, (1, .78, .30)), (1, (1, .97, .88))])
ICE  = _ramp([(0, (0, .02, .12)), (.3, (.05, .25, .75)), (.6, (.10, .70, 1)), (.85, (.6, .95, 1)), (1, (1, 1, 1))])
GOLD = _ramp([(0, (.10, .03, 0)), (.4, (.70, .35, .05)), (.7, (1, .75, .25)), (1, (1, .97, .80))])

def smooth(x, a=0.0, b=1.0):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)

def sci(x):
    return "0" if x <= 0 else f"{x:.1e}"

# ---------- fonts (Times New Roman everywhere, Liberation Serif = metric clone fallback) ----------
_FONTS = {
    "r": ["C:/Windows/Fonts/times.ttf", "/System/Library/Fonts/Supplemental/Times New Roman.ttf", "/Library/Fonts/Times New Roman.ttf",
          "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"],
    "i": ["C:/Windows/Fonts/timesi.ttf", "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf", "/Library/Fonts/Times New Roman Italic.ttf",
          "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf", "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"],
}
_cache = {}
def font(size, style="r"):
    k = (size, style)
    if k not in _cache:
        for p in _FONTS[style] + [os.environ.get("TNR_PATH", "")]:
            if p and os.path.exists(p):
                _cache[k] = ImageFont.truetype(p, size); break
        else:
            _cache[k] = ImageFont.load_default()
    return _cache[k]

def _accum(acc, items):
    if not items: return
    sx = np.concatenate([i[0] for i in items]); sy = np.concatenate([i[1] for i in items])
    col = np.concatenate([i[2] for i in items]); w = np.concatenate([i[3] for i in items])
    x0 = np.floor(sx).astype(np.int64); y0 = np.floor(sy).astype(np.int64)
    ok = (x0 >= 0) & (x0 < W - 1) & (y0 >= 0) & (y0 < H - 1)
    x0, y0, fx, fy, col, w = x0[ok], y0[ok], (sx - np.floor(sx))[ok], (sy - np.floor(sy))[ok], col[ok], w[ok]
    base = y0 * W + x0; WH = W * H
    for off, wt in ((0, (1 - fx) * (1 - fy)), (1, fx * (1 - fy)), (W, (1 - fx) * fy), (W + 1, fx * fy)):
        idx = np.concatenate([base + off, base + off + WH, base + off + 2 * WH])
        wgt = np.concatenate([wt * w * col[:, 0], wt * w * col[:, 1], wt * w * col[:, 2]])
        acc += np.bincount(idx, weights=wgt, minlength=3 * WH).astype(np.float32).reshape(acc.shape)

def densify(P, sub):
    """P (m,n,3) grid -> (m,(n-1)*sub+1,3) linear interpolation along axis 1."""
    t = np.linspace(0, 1, sub, endpoint=False)
    a, b = P[:, :-1, None, :], P[:, 1:, None, :]
    out = (a + (b - a) * t[None, None, :, None]).reshape(P.shape[0], -1, 3)
    return np.concatenate([out, P[:, -1:, :]], 1)

def polyline(P, step=0.02):
    """Dense samples along one polyline (n,3)."""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1); n = np.maximum((seg / step).astype(int), 1)
    pts = [P[i] + (P[i + 1] - P[i]) * np.linspace(0, 1, n[i], endpoint=False)[:, None] for i in range(len(P) - 1)]
    return np.concatenate(pts + [P[-1:]], 0)

class Scene:
    def __init__(s, seed=3):
        s.cam = dict(az=0., el=.5, dist=9., fov=40., t=np.zeros(3))
        s.trail = np.zeros((3, H, W), np.float32); s.exposure = 1.0; s.persist = 0.0
        r = np.random.default_rng(seed)
        s.dust = r.uniform(-14, 14, (260, 3)); s.dustc = r.uniform(.15, .6, 260)
        yy, xx = np.mgrid[0:H, 0:W]; d = np.hypot((xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2))
        s.vig = (1 - 0.55 * d ** 2.2).clip(0.15, 1).astype(np.float32)[..., None]
        s.bg = (np.array([.006, .008, .020]) * (1 - .6 * d.clip(0, 1))[..., None]).astype(np.float32)
        s.rng = np.random.default_rng(seed + 1)

    # -- camera --
    def project(s, P):
        c = s.cam; P = np.asarray(P, float).reshape(-1, 3) - c["t"]
        ca, sa, ce, se = math.cos(c["az"]), math.sin(c["az"]), math.cos(c["el"]), math.sin(c["el"])
        x1 = ca * P[:, 0] + sa * P[:, 1]; y1 = -sa * P[:, 0] + ca * P[:, 1]; z1 = P[:, 2]
        y2 = ce * y1 - se * z1; z2 = ce * z1 + se * y1
        dep = y2 + c["dist"]; f = H * .5 / math.tan(math.radians(c["fov"]) / 2); dep = np.where(dep < .3, .3, dep)
        return W / 2 + f * x1 / dep, H / 2 - f * z2 / dep, dep

    def begin(s, persist=0.0, exposure=1.0):
        s.persist, s.exposure = persist, exposure
        s.qs, s.qt, s.labels, s.texts, s.bar = [], [], [], [], None

    def pts(s, P, col, w=1.0, trail=False, depthfx=True):
        P = np.asarray(P, float).reshape(-1, 3); n = len(P)
        if n == 0: return
        sx, sy, dep = s.project(P)
        col = np.broadcast_to(np.asarray(col, float), (n, 3)); w = np.broadcast_to(np.asarray(w, float), (n,))
        if depthfx: w = w * np.clip((s.cam["dist"] / dep) ** 1.3, .15, 3.0)
        (s.qt if trail else s.qs).append((sx, sy, col, w))

    def screen(s, sx, sy, col, w=1.0, trail=False):
        sx = np.asarray(sx, float).ravel(); n = len(sx)
        (s.qt if trail else s.qs).append((sx, np.asarray(sy, float).ravel(), np.broadcast_to(np.asarray(col, float), (n, 3)), np.broadcast_to(np.asarray(w, float), (n,))))

    def label3d(s, P, text, size=20, anchor="mm", alpha=1.0, style="r"):
        sx, sy, _ = s.project(P); s.labels.append((sx[0], sy[0], text, size, anchor, alpha, style))

    def text(s, x, y, text, size=20, anchor="la", alpha=1.0, style="r"):
        s.texts.append((x, y, text, size, anchor, alpha, style))

    def background_dust(s, t):
        d = s.dust.copy(); d[:, 2] += (t * .15) % 28 - 14; d[:, 2] = (d[:, 2] + 14) % 28 - 14
        s.pts(d, np.outer(s.dustc, [.35, .5, .9]), .5, depthfx=False)

    # -- frame finish --
    def finish(s, progress=None):
        if getattr(s, 'skip', False): return None
        acc_s = np.zeros((3, H, W), np.float32); _accum(acc_s, s.qs)
        new_t = np.zeros((3, H, W), np.float32); _accum(new_t, s.qt)
        prev = s.trail * s.persist; s.trail = prev + new_t
        tot = acc_s + new_t + 0.5 * prev
        img = np.ascontiguousarray(tot.transpose(1, 2, 0)); t = 1 - np.exp(-img * s.exposure)
        out = t.copy()
        for ds, sig, wt in ((1, 2, .35), (2, 3, .45), (4, 4, .55), (8, 4, .65)):
            sm = cv2.resize(t, (W // ds, H // ds), interpolation=cv2.INTER_AREA) if ds > 1 else t
            b = cv2.GaussianBlur(sm, (0, 0), sig)
            out += wt * (cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR) if ds > 1 else b)
        out = 1 - np.exp(-out * 1.15)
        out = out * s.vig + s.bg + s.rng.normal(0, .0025, out.shape).astype(np.float32)
        im = Image.fromarray((np.clip(out, 0, 1) ** (1 / 1.04) * 255).astype(np.uint8)).convert("RGBA")
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
        for sx, sy, tx, sz, an, al, st in s.labels: d.text((sx, sy), tx, font=font(sz, st), fill=(255, 255, 255, int(255 * al)), anchor=an)
        for x, y, tx, sz, an, al, st in s.texts: d.text((x, y), tx, font=font(sz, st), fill=(255, 255, 255, int(255 * al)), anchor=an)
        if progress is not None: d.rectangle([0, H - 3, int(W * progress), H], fill=(255, 255, 255, 110))
        return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))

def hud(sc, M, title, phases, t, readouts):
    sc.text(48, 34, title, 30)
    sc.text(48, 76, f"Gold futures  ·  {M['source']}  ·  {M['interval']}  ·  {M['n_range']}-bar structural range  ·  {M['lo']:,.1f} – {M['hi']:,.1f}", 16, alpha=.7, style="i")
    for t0, t1, txt in phases:
        a = float(min(smooth(t, t0, t0 + .8), 1 - smooth(t, t1 - .8, t1)))
        if a > .01: sc.text(48, H - 98, txt, 25, alpha=a, style="i")
    for i, r in enumerate(readouts): sc.text(48, H - 62 + i * 22, r, 17, alpha=.9)
    if BRAND: sc.text(W - 48, H - 38, BRAND, 16, anchor="ra", alpha=.6, style="i")

def _ffmpeg():
    try:
        import imageio_ffmpeg; return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"

def render(model_cls, M, out_dir="videos", preview=None, fps=FPS, dur=DUR):
    """Render a Model to MP4 (or a single PNG if preview=t_seconds)."""
    os.makedirs(out_dir, exist_ok=True)
    m = model_cls(M); sc = Scene()
    n = int(fps * dur)
    if preview is not None:
        last = int(preview * fps)
        for k in range(last + 1):
            sc.skip = k < last; fr = m.frame(sc, k / fps)
        path = os.path.join(out_dir, f"{m.slug}_preview.png"); Image.fromarray(fr).save(path); return path
    path = os.path.join(out_dir, f"{m.slug}.mp4")
    p = subprocess.Popen([_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-preset", "medium", "-crf", "15", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    for k in range(n):
        p.stdin.write(m.frame(sc, k / fps).tobytes())
        if k % 48 == 0: print(f"   {m.slug}  {k}/{n}", flush=True)
    p.stdin.close(); p.wait(); return path
