"""Prometheus v3: mix and master the stems rendered by score.py.

    python3 mix.py <stems_dir> <out.wav> [passes]

Pass 1 mixes with static settings. Each further pass measures the loudness
contour of the mix and nudges a smooth gain automation toward cues.REF_CONTOUR.
"""
import os
import sys
import wave

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

import cues as C

SR = 44100

#           gain   lowcut highcut reverb-send  ducked-by-perc
BUS = {
    "strings":  (1.00, 110, 12000, 0.55, 0.30),
    "ostinato": (1.70, 120, 15000, 0.18, 0.00),
    "low":      (1.00,  18,   140, 0.00, 0.00),
    "brass":    (0.90,  60,  9000, 0.35, 0.20),
    "choir":    (0.90, 180, 11000, 0.70, 0.30),
    "organ":    (0.95,  25, 10000, 0.70, 0.00),
    "perc":     (1.45,  30, 16000, 0.22, 0.00),
    "sparkle":  (1.20, 400, 16000, 0.55, 0.00),
    "sfx":      (0.55,  30, 16000, 0.30, 0.00),
    "air":      (1.00, 800, 16000, 0.30, 0.00),
}


def band(x, lo, hi):
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(len(x), 1 / SR)[:, None]
    H = (1 / np.sqrt(1 + (lo / np.maximum(f, 1)) ** 4)) * (1 / np.sqrt(1 + (f / hi) ** 4))
    return np.fft.irfft(X * H, len(x), axis=0)


def make_ir(sec=4.2, seed=0):
    r = np.random.default_rng(seed)
    n = int(sec * SR)
    t = np.arange(n) / SR
    ir = r.standard_normal((n, 2)) * np.exp(-t * 6.9 / sec)[:, None]
    ir = band(ir, 200, 5500)
    ir[: int(0.025 * SR)] = 0
    # early reflections
    for d, g in ((0.031, .5), (0.047, .4), (0.063, .35), (0.089, .3)):
        ir[int(d * SR)] += g * np.array([1, .8]) if r.random() > .5 else g * np.array([.8, 1])
    return ir / np.sqrt(np.sum(ir ** 2) / 2)


def convolve(x, ir):
    n = len(x) + len(ir)
    size = 1 << int(np.ceil(np.log2(n)))
    return np.stack([np.fft.irfft(np.fft.rfft(x[:, c], size) * np.fft.rfft(ir[:, c], size), size)[:len(x)]
                     for c in range(2)], 1)


def envelope(x, ms):
    m = np.abs(x).max(1) if x.ndim == 2 else np.abs(x)
    return uniform_filter1d(m, int(ms * SR / 1000))


def compress(x, thresh_db=-18, ratio=2.5, ms=60):
    e = np.sqrt(uniform_filter1d((x ** 2).mean(1), int(ms * SR / 1000))) + 1e-9
    lvl = 20 * np.log10(e)
    over = np.maximum(lvl - thresh_db, 0)
    g = 10 ** (-(over - over / ratio) / 20)
    return x * g[:, None]


def limiter(x, ceiling=0.95, look_ms=5, release_ms=80):
    peak = np.abs(x).max(1)
    g = np.minimum(1.0, ceiling / (peak + 1e-9))
    w = int(look_ms * SR / 1000)
    g = minimum_filter1d(g, 2 * w + 1)
    g = uniform_filter1d(g, w)
    # release smoothing: only let gain rise slowly
    rel = np.exp(-1 / (release_ms * SR / 1000))
    out = np.empty_like(g)
    cur = 1.0
    for i in range(0, len(g), 64):          # block-wise for speed
        tgt = g[i:i + 64].min()
        cur = tgt if tgt < cur else tgt + (cur - tgt) * rel ** 64
        out[i:i + 64] = cur
    return x * np.minimum(out, g)[:, None]


def contour(x, win=4.0, dur=C.DUR):
    m = x.mean(1) if x.ndim == 2 else x
    v = []
    for s in np.arange(0, len(C.REF_CONTOUR) * win, win):
        seg = m[int(s * SR):int((s + win) * SR)]
        v.append(10 * np.log10(np.mean(seg ** 2) + 1e-12))
    v = np.array(v)
    return v - v.max()


def automation_curve(diff_db, n, win=4.0):
    centers = (np.arange(len(diff_db)) + 0.5) * win * SR
    curve = np.interp(np.arange(n), centers, diff_db)
    curve = uniform_filter1d(curve, int(1.5 * SR))      # smooth, no zipper
    return 10 ** (curve / 20)


def mix(stems, auto=None):
    n = len(next(iter(stems.values())))
    perc_env = envelope(stems["perc"] + stems["low"], 30)
    perc_env /= perc_env.max() + 1e-9
    dry = np.zeros((n, 2))
    wet_in = np.zeros((n, 2))
    for k, (g, lo, hi, send, duck) in BUS.items():
        x = band(stems[k].astype(np.float64), lo, hi) * g
        if duck:
            x *= (1 - duck * perc_env)[:, None]
        if k in ("strings", "choir", "sfx", "sparkle"):    # widen
            mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
            x = np.stack([mid + side * 1.9, mid - side * 1.9], 1)
        dry += x
        wet_in += x * send
    wet = convolve(wet_in, IR)
    bus = dry + wet * 0.75
    bus = bus + band(bus, 4500, 20000) * 1.6            # air shelf
    bus = compress(bus / (np.abs(bus).max() + 1e-9), -16, 2.2, 80)
    if auto is not None:
        bus *= auto[:, None]
    bus /= np.abs(bus).max() + 1e-9
    bus = np.tanh(bus * 1.2) / np.tanh(1.2)               # gentle saturation
    bus = limiter(bus * 1.15, 0.94)
    end = int(C.DUR * SR)
    bus = bus[:end]
    fade = int(1.8 * SR)
    bus[-fade:] *= np.linspace(1, 0, fade)[:, None]
    return bus


def write(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


if __name__ == "__main__":
    d, out = sys.argv[1], sys.argv[2]
    passes = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    stems = {k: np.load(os.path.join(d, k + ".npy")) for k in BUS}
    IR = make_ir()
    ref = np.array(C.REF_CONTOUR)
    auto = None
    n = len(stems["strings"])
    total_db = np.zeros(len(ref))
    for p in range(passes):
        m = mix(stems, auto)
        c = contour(m)
        diff = ref - c
        print(f"pass {p + 1}: contour corr {np.corrcoef(ref, c)[0, 1]:.3f}  "
              f"mean abs error {np.mean(np.abs(diff)):.2f} dB")
        total_db = np.clip(total_db + 0.8 * diff, -9, 6)
        auto = automation_curve(total_db, n)
    write(out, m)
    np.save(out + ".auto.npy", total_db)
