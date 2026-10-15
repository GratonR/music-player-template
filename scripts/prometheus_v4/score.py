"""Prometheus v3: score and SFX pass. Renders one stereo stem per group.

    python3 score.py <out_dir>

Timing comes from cues.py (measured on the reference). Melody, harmony and
sounds are original; the only quotations are public domain (Bach BWV 565
opening, Beethoven Symphony No. 5 motif).
"""
import os
import sys

import numpy as np

import cues as C
import instruments as I
from instruments import SR

N = int((C.DUR + 3) * SR)
STEMS = {k: np.zeros((N, 2), dtype=np.float32) for k in
         ("strings", "ostinato", "low", "brass", "choir", "organ", "perc", "sparkle", "sfx", "air")}
rng = np.random.default_rng(21)


def add(stem, sig, t, gain=1.0, pan=0.0):
    i = int(t * SR)
    if i >= N or i < 0 or len(sig) == 0:
        return
    sig = sig[: N - i] * gain
    if sig.ndim == 2:
        STEMS[stem][i:i + len(sig)] += sig
        return
    STEMS[stem][i:i + len(sig), 0] += sig * np.cos((pan + 1) * np.pi / 4)
    STEMS[stem][i:i + len(sig), 1] += sig * np.sin((pan + 1) * np.pi / 4)


# ----------------------------------------------------------------- harmony
# chord = (bass, [upper voices])
Dm = (38, [62, 65, 69]); Bb = (34, [62, 65, 70]); Gm = (43, [62, 67, 70]); A = (45, [61, 64, 69])
C_ = (36, [60, 64, 67]); F = (41, [60, 65, 69])
D = (38, [62, 66, 69, 76]); Bm = (35, [62, 66, 69, 71]); G = (43, [62, 66, 67, 71]); Amaj = (45, [61, 64, 69, 71])

# Original theme, in quarter-beats of the 72 BPM feel (0.833 s)
THEME_MIN = [(69, 1), (74, 1), (76, .5), (77, .5), (76, 1), (74, .5), (72, .5), (69, 2),
             (70, 1), (69, 1), (67, .5), (65, .5), (64, 1), (62, 2)]
MAJ = {77: 78, 72: 73, 70: 71, 65: 66, 60: 61}
THEME_MAJ = [(MAJ.get(m, m), d) for m, d in THEME_MIN]
Q = 60 / 72


def theme(stem, t0, notes, inst, transpose=0, gain=1.0, pan=0.1, q=Q):
    t = t0
    for m, d in notes:
        add(stem, inst(m + transpose, d * q * 1.02), t, gain, pan)
        t += d * q


def pads(t0, t1, chords, chord_len, gain=1.0, choir_gain=0.0, sub_gain=1.0, a=0.8):
    t, i = t0, 0
    while t < t1 - 0.3:
        bass, upper = chords[i % len(chords)]
        d = min(chord_len, t1 - t) + 0.9
        for j, m in enumerate(upper):
            add("strings", I.string_section(m, d, a=a, gain=gain), t, 1.0, -0.6 + 0.6 * j)
            add("strings", I.string_section(m - 12, d, a=a, voices=4, gain=gain * 0.7), t, 1.0, 0.5 - 0.5 * j)
            if choir_gain:
                add("choir", I.choir(m, d, gain=choir_gain), t, 1.0, 0.6 - 0.6 * j)
        if sub_gain:
            add("low", I.sub(bass - 12 if bass >= 34 else bass, d, sub_gain * 1.2), t)
            add("low", I.sub(bass, d, sub_gain * 0.35), t)
        t += chord_len
        i += 1


def ostinato(t0, t1, step, anchor, chords, notes_per_chord, pattern, accents, gain=1.0,
             octave=0, pan_spread=0.35):
    """pattern: indices into (bass+24, upper...) cycled; None = rest."""
    k0 = int(np.ceil((t0 - anchor) / step - 1e-6))
    k = k0
    while True:
        t = anchor + k * step
        if t >= t1:
            break
        if t >= t0:
            idx = k - k0
            bass, upper = chords[(idx // notes_per_chord) % len(chords)]
            p = pattern[idx % len(pattern)]
            if p is not None:
                pool = [bass + 24] + upper
                m = pool[p % len(pool)] + octave
                acc = accents[idx % len(accents)]
                add("ostinato", I.spiccato(m, step * 0.7, acc), t, gain,
                    pan_spread if idx % 2 else -pan_spread)
        k += 1


def grid(t0, t1, step, anchor, every=1, offset=0):
    k = int(np.ceil((t0 - anchor) / step - 1e-6))
    out = []
    while anchor + k * step < t1:
        if (k - offset) % every == 0 and anchor + k * step >= t0:
            out.append(anchor + k * step)
        k += 1
    return out


# =============================================================== MUSIC
def music():
    # ---- 0:00-0:07.6 Prometheus: drone, hum, distant pulse
    add("strings", I.string_section(38, 7.9, a=1.0, r=0.4, gain=2.0), 0.0)
    add("strings", I.string_section(45, 7.9, a=1.4, r=0.4, gain=1.4), 0.0, 1.0, 0.3)
    add("choir", I.choir(62, 7.7, vowel=(400, 800, 2600), gain=1.3, r=0.4), 0.3)
    add("low", I.sub(26, 7.7, 1.4, a=1.5, r=0.3), 0.0)
    for t in (4.48, 6.66):
        add("perc", I.taiko(55, 1.8, 0.9), t)
    for t in grid(1.0, 7.4, 0.5556, 1.0):
        add("sparkle", I.celesta(86 if int(t * 10) % 3 else 81, 1.0, 0.6), t, 1.0, 0.5)
    add("air", I.air_bed(7.7, 1.8), 0.0)
    # drop 7.6 -> 9.7 (SFX only)

    # ---- 0:09.9-0:22 Greece: 72 BPM triplet ostinato
    s0, s1, step, anc = C.PULSE["greece"]
    prog = [Dm, Bb, Gm, A]
    pads(9.7, 22.3, prog, 4 * Q, gain=1.0, sub_gain=1.0)
    k = 0
    for t in grid(s0, s1, step, anc):
        bass, upper = prog[(k // 12) % 4]
        m = ([bass + 24] + upper)[[3, 2, 1, 2, 3, 0][k % 6]]
        add("ostinato", I.pluck(m + 12, 0.6, [1.3, .7, .9, 1.1, .7, .9][k % 6]), t, 1.4, 0.3 if k % 2 else -0.3)
        add("ostinato", I.pluck(m, 0.6, 0.6), t, 1.0, -0.3 if k % 2 else 0.3)
        k += 1
    theme("strings", 13.27, THEME_MIN, lambda m, d: I.string_section(m, d, a=0.08, r=0.5, voices=5,
                                                                    bright=1.15, gain=1.5))
    for t in grid(s0, s1, step, anc, every=6):
        add("perc", I.taiko(68, 1.4, 0.7), t, 1.0, -0.2)

    # ---- 0:22.3-0:35.4 Rome: same grid, march
    s0, s1, step, anc = C.PULSE["rome"]
    prog = [Dm, C_, Bb, A]
    pads(22.3, 35.4, prog, 4 * Q, gain=1.1, sub_gain=1.1)
    ostinato(s0, s1, step, anc, prog, 12, [3, None, 2, 3, None, 1], [1.3, 0, .8, 1.1, 0, .8])
    for t in grid(s0, s1, step, anc):
        k = int(round((t - anc) / step)) % 12
        if k in (0, 2, 3, 6, 8, 9, 11):
            add("perc", I.snare(1.4 if k in (3, 9) else 0.7), t, 1.0, 0.25 if k % 2 else -0.25)
    march = [0, 2, 3, 6, 8, 9]  # long-short-short figure on the triplet grid
    for t in grid(s0, s1, step, anc):
        k = int(round((t - anc) / step)) % 12
        if k in march:
            add("perc", I.taiko(60 if k in (0, 6) else 82, 1.2, 1.0 if k in (0, 6) else 0.55), t,
                1.0, rng.uniform(-.3, .3))
    for i, t in enumerate(np.arange(22.5, 35.0, 4 * Q)):
        for m in prog[i % 4][1]:
            add("brass", I.brass(m - 12, 4 * Q * 0.95, 1.3), t, 1.0, -0.3)
    add("air", I.air_bed(13.2, 1.6), 22.3)
    theme("brass", 28.9, THEME_MIN[:8], lambda m, d: I.brass(m, d, 1.4), transpose=-12)

    # ---- 0:35.4-0:39.3 Monks: thin out, lone choir, three low hits
    add("choir", I.choir(62, 4.3, vowel=(400, 800, 2600), gain=1.6), 35.3, 1.0, -0.1)
    add("choir", I.choir(57, 4.3, vowel=(400, 800, 2600), gain=1.1), 35.3, 1.0, 0.1)
    add("air", I.air_bed(4.6, 0.8), 35.1)
    add("choir", I.choir(74, 4.3, vowel=(400, 800, 2600), gain=0.9), 35.3, 1.0, 0.4)
    for t in (35.57, 35.85, 36.13):
        add("perc", I.snare(1.3, 0.3), t, 1.0, -0.3)

    # ---- 0:39.8-1:04.8 Cathedral: ~258 notes/min, octave bounce, theme
    s0, s1, step, anc = C.PULSE["cathedral"]
    prog = [Dm, Bb, F, C_, Gm, Bb, A, A]
    clen = 16 * step
    pads(39.8, 64.7, prog, clen, gain=1.2, choir_gain=1.0, sub_gain=1.2)
    ostinato(s0, s1, step, anc, prog, 16, [0, 3, 0, 2, 0, 3, 0, 1], [1.1, .85, 1.0, .85, 1.05, .85, 1.0, .85],
             gain=1.1)
    ostinato(47.3, s1, step, anc, prog, 16, [None, 3, None, 2], [0, .7, 0, .7], gain=0.7, octave=12)
    for t in grid(s0, s1, step, anc, every=4):
        add("perc", I.taiko(64, 1.3, 0.8), t)
    for t in grid(s0, s1, step, anc, every=8, offset=6):
        add("perc", I.taiko(90, 0.7, 0.4), t, 1.0, 0.4)
    theme("brass", 43.6, THEME_MIN, lambda m, d: I.brass(m, d, 1.4))
    theme("strings", 54.7, THEME_MIN, lambda m, d: I.string_section(m + 12, d, a=0.06, r=0.4, voices=6,
                                                                    bright=1.15, gain=1.4))
    for t in grid(s0, s1, step, anc, every=2):
        add("perc", I.clock_tick(0.35), t, 1.0, 0.6)

    # ---- 1:04.9-1:11 Bach and Beethoven (public domain)
    t = 66.9
    for m, d in [(81, .12), (79, .12), (81, .9), (79, .14), (77, .14), (76, .14), (74, .14), (73, .6),
                 (74, 1.4)]:
        add("organ", I.organ(m, d + 0.05, 2.0), t)
        add("organ", I.organ(m - 12, d + 0.05, 1.5), t)
        t += d
    for m, g in ((26, 1.8), (38, 2.0), (50, 1.6), (53, 1.3), (57, 1.3), (61, 1.0), (62, 1.0)):
        add("organ", I.organ(m, 3.0, g), 69.08)
    add("low", I.low_boom(3.0, 0.8), 66.9)
    t = 70.6
    for m, d in ((69, .2), (69, .2), (69, .2), (65, 0.6)):
        for o in (-12, 0):
            add("strings", I.string_section(m + o, d + 0.1, a=0.01, r=0.12, voices=6, gain=2.2), t)
        t += d + 0.02

    # ---- 1:11-1:31 Industry: 72 BPM sixteenths, clock, build
    s0, s1, step, anc = C.PULSE["industry"]
    prog = [Dm, Bb, F, A]
    clen = 16 * step
    pads(71.2, 91.0, prog, clen, gain=1.3, choir_gain=1.1, sub_gain=1.3)
    ostinato(s0, s1, step, anc, prog, 16, [0, 3, 0, 2, 0, 3, 0, 1], [1.1, .85, 1.0, .85, 1.05, .85, 1.0, .85],
             gain=1.2)
    ostinato(79.6, s1, step, anc, prog, 16, [3, 2, 1, 2], [.8, .6, .7, .6], gain=0.7, octave=12)
    for t in grid(s0, s1, step, anc, every=4):
        add("perc", I.taiko(58, 1.2, 0.9), t)
    for t in grid(s0, s1, step, anc, every=2):
        add("perc", I.clock_tick(0.25), t, 1.0, -0.5)
    theme("brass", 81.2, THEME_MIN, lambda m, d: I.brass(m, d, 1.7), q=4 * step)
    for t in (86.1, 87.83, 88.27, 88.67, 89.1, 89.53):
        add("perc", I.taiko(52, 1.0, 1.1), t)

    # ---- 1:31-1:34.5 Countdown
    add("strings", I.string_section(57, 3.6, a=1.5, r=0.2, gain=1.6), 91.0)
    add("strings", I.string_section(45, 3.6, a=1.5, r=0.2, gain=1.6), 91.0)
    for t in (92.55, 93.34, 94.0):
        add("perc", I.taiko(56, 1.0, 1.1), t)

    # ---- 1:34.6-2:04.4 Space: ~304 notes/min, D major
    s0, s1, step, anc = C.PULSE["space"]
    prog = [D, Bm, G, Amaj]
    clen = 16 * step
    pads(94.57, 124.3, prog, clen, gain=1.4, choir_gain=1.3, sub_gain=1.4, a=0.3)
    ostinato(s0, s1, step, anc, prog, 16, [0, 3, 0, 2, 0, 3, 0, 1], [1.1, .85, 1.0, .85, 1.05, .85, 1.0, .85],
             gain=1.3)
    ostinato(s0, s1, step, anc, prog, 16, [3, 2, 1, 2], [.8, .6, .7, .6], gain=0.8, octave=12)
    for t in grid(s0, s1, step, anc, every=4):
        add("perc", I.taiko(60, 1.1, 1.0), t, 1.0, rng.uniform(-.3, .3))
    for t in grid(s0, s1, step, anc, every=8, offset=2):
        add("perc", I.taiko(88, 0.6, 0.5), t, 1.0, 0.4)
    for t in grid(s0, s1, step, anc, every=2):
        add("perc", I.clock_tick(0.25, 4200), t, 1.0, 0.5)
    q = 4 * step
    theme("brass", 97.7, THEME_MAJ, lambda m, d: I.brass(m, d, 1.8), q=q)
    theme("choir", 107.2, THEME_MAJ, lambda m, d: I.choir(m + 12, d, gain=1.6, a=0.1, r=0.4), q=q)
    theme("strings", 116.4, THEME_MAJ, lambda m, d: I.string_section(m + 12, d, a=0.05, r=0.3, voices=6,
                                                                    bright=1.1, gain=1.7), q=q)
    for t in C.MACHINE_HITS:
        add("low", I.braam(38, 1.5, 0.8), t)
        add("perc", I.taiko(45, 1.3, 1.4), t)
    for t in (120.07, 121.67, 122.47, 122.87, 123.37, 123.76):
        add("perc", I.taiko(54, 0.9, 1.1), t)

    # ---- 2:04.4-2:10 One fire: almost nothing
    add("sparkle", I.bell(74, 6.0, 0.5), 125.0, 1.0, 0.2)
    add("low", I.sub(26, 5.8, 1.3, a=1.0, r=1.0), 124.9)
    add("air", I.air_bed(6.0, 1.4), 124.7)
    add("strings", I.string_section(50, 5.5, a=2.5, r=1.0, gain=0.9), 125.0)

    # ---- 2:10.5-2:16.5 Accelerate: heartbeat pulse, D major swell, last hit
    for t in grid(130.57, 133.2, 0.5, 130.57):
        add("perc", I.taiko(46, 1.0, 0.45), t)
        add("low", I.sub(26, 0.45, 1.3, a=0.005, r=0.2), t)
    for j, m in enumerate((50, 57, 62, 69, 74)):
        add("strings", I.string_section(m, 6.0, a=2.2, r=1.2, gain=1.3), 130.4, 1.0, -0.6 + 0.24 * j)
        add("choir", I.choir(m, 6.0, gain=1.2, a=2.0, r=1.2), 130.6, 1.0, 0.6 - 0.24 * j)
    add("brass", I.braam(50, 3.0, 0.9), 133.47)
    add("low", I.sub(26, 3.0, 1.4, a=0.01, r=1.5), 133.47)
    add("sparkle", I.bell(86, 3.0, 1.0), 133.47, 1.0, -0.3)


# =============================================================== SFX
def bandsweep(dur, f0, f1, q=6.0, gain=1.0, shape=1.0):
    """Noise through a swept resonant band: whooshes and risers."""
    n = int(dur * SR)
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    out = np.zeros(n)
    segs = 24
    edges = np.linspace(0, n, segs + 1).astype(int)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    for s in range(segs):
        fc = f0 * (f1 / f0) ** (((s + .5) / segs) ** shape)
        H = np.exp(-((freqs - fc) / (fc / q)) ** 2)
        y = np.fft.irfft(X * H, n)
        w = np.zeros(n)
        a, b = edges[s], edges[s + 1]
        w[a:b] = 1
        out += y * w
    out = np.convolve(out, np.ones(256) / 256, "same")
    return out / (np.max(np.abs(out)) + 1e-9) * gain


def whoosh(dur=0.6, up=True, gain=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    e = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** (1.5 if up else 0.7)
    sw = bandsweep(dur, 400, 5000, 3, 1.0) if up else bandsweep(dur, 5000, 300, 3, 1.0)
    return sw * e * 0.25 * gain


def riser(dur, gain=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = bandsweep(dur, 300, 9000, 4, 1.0, shape=2.0)
    tone = np.sin(2 * np.pi * np.cumsum(120 + 1400 * (t / dur) ** 3) / SR) * 0.15
    return (s * 0.25 + tone) * (t / dur) ** 2.2 * gain


def reverse_cymbal(dur, gain=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = I.hp(I.hp(rng.standard_normal(n)))
    return x * np.exp(-(dur - t) * 3.0) * 0.06 * gain


def fire_bed(dur, density=30, gain=1.0):
    n = int(dur * SR)
    x = np.zeros(n)
    idx = rng.integers(0, n, int(dur * density))
    x[idx] = rng.uniform(-1, 1, len(idx)) * rng.uniform(.2, 1, len(idx))
    pops = np.convolve(I.hp(x), np.exp(-np.arange(200) / 30), "same")
    roar = np.convolve(rng.standard_normal(n), np.ones(400) / 400, "same") * 3.0
    flicker = 1 + 0.4 * np.sin(2 * np.pi * 0.7 * np.arange(n) / SR) * np.sin(2 * np.pi * 0.23 * np.arange(n) / SR)
    return (pops * 0.35 + roar * flicker * 0.15) * gain


def digital_ticks(t0, dur, rate0=28, rate1=6, gain=1.0):
    t = t0
    while t < t0 + dur:
        p = (t - t0) / dur
        add("sfx", I.clock_tick(0.9 * gain, 5200 + 800 * rng.random()), t, 1.0, rng.uniform(-.2, .2))
        t += 1 / (rate0 + (rate1 - rate0) * p)


def beep(f, dur=0.12, gain=1.0):
    t = np.arange(int(dur * SR)) / SR
    return np.sin(2 * np.pi * f * t) * I.env(len(t), 0.003, 0.01, 0.8, 0.02) * 0.08 * gain


def shing(gain=1.0):
    t = np.arange(int(1.2 * SR)) / SR
    s = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, a, d in
            ((2310, 1, 3), (3570, .7, 4), (5120, .5, 5), (6830, .3, 6), (1480, .3, 2.5)))
    scrape = I.hp(rng.standard_normal(len(t))) * np.exp(-t * 14) * 0.6
    return (s * 0.05 + scrape * 0.05) * gain


def radio_static(dur, gain=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = I.hp(rng.standard_normal(n)) * (0.5 + 0.5 * (rng.random(n) > 0.97))
    whistle = np.sin(2 * np.pi * np.cumsum(900 + 700 * np.sin(2 * np.pi * 0.8 * t)) / SR) * 0.3
    return (x * 0.06 + whistle * 0.04) * I.env(n, 0.05, 0.1, 0.9, 0.3) * gain


def rumble(dur, gain=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = np.convolve(rng.standard_normal(n), np.ones(600) / 600, "same") * 25
    crackle = I.hp(rng.standard_normal(n)) * (rng.random(n) > 0.9) * 0.3
    return (x + crackle * 0.2) * np.exp(-t / (dur * 0.45)) * I.env(n, 0.05, 0.1, 1, 0.5) * 0.4 * gain


def servo(dur=0.6, gain=1.0):
    t = np.arange(int(dur * SR)) / SR
    f = 300 + 500 * np.sin(np.pi * t / dur)
    s = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * 0.3 + np.sin(2 * np.pi * np.cumsum(f * 2) / SR)
    hiss = I.hp(rng.standard_normal(len(t))) * np.exp(-((t - dur * 0.7) / 0.1) ** 2) * 0.5
    return (s * 0.03 + hiss * 0.08) * np.sin(np.pi * t / dur) * gain


def paper(dur=0.8, gain=1.0):
    n = int(dur * SR)
    x = I.hp(rng.standard_normal(n)) * (rng.random(n) > 0.6)
    return np.convolve(x, np.ones(30) / 30, "same") * np.sin(np.pi * np.arange(n) / n) * 0.5 * gain


def shutter(gain=1.0):
    t = np.arange(int(0.08 * SR)) / SR
    a = I.hp(rng.standard_normal(len(t))) * np.exp(-t * 120)
    b = np.zeros_like(a)
    k = int(0.035 * SR)
    b[k:] = a[: len(a) - k] * 0.7
    return (a + b) * 0.12 * gain


def sfx():
    # Prometheus flame
    add("sfx", fire_bed(9.3, 40, 0.9), 0.0)
    add("sfx", whoosh(0.9, True, 1.4), 0.0, 1.0, -0.2)
    for t in (0.47, 1.07, 2.2, 2.53):
        add("sfx", whoosh(0.35, True, 0.5), t - 0.15, 1.0, rng.uniform(-.5, .5))
    add("sfx", whoosh(0.5, False, 1.0), 4.3)
    # suck-out at 7.6, rush back into 9.7
    add("sfx", reverse_cymbal(0.7, 1.4), 6.9)
    add("sfx", riser(1.4, 1.5), 8.3)
    add("sfx", I.low_boom(3.5, 1.2), 9.7)
    digital_ticks(8.6, 2.4, 30, 8, 0.9)                 # "4,029" counter
    # Rome
    for t in (25.75, 30.12, 34.48):
        add("sfx", shing(1.0), t, 1.0, 0.4)
    add("sfx", whoosh(0.5, True, 0.8), 23.8)
    add("sfx", whoosh(0.5, False, 0.7), 34.8)
    # Monks -> cathedral
    add("sfx", paper(0.9, 1.0), 36.3, 1.0, 0.3)
    add("sfx", reverse_cymbal(2.6, 1.6), 37.2)
    add("sfx", riser(2.9, 1.8), 36.9)
    for t in (37.5, 37.78):
        add("sfx", I.celesta(93, 1.2, 1.2), t, 1.0, -0.4)
    add("sfx", I.low_boom(4.0, 1.4), 39.8)
    add("sfx", whoosh(0.4, True, 0.6), 41.0)
    add("sfx", whoosh(0.4, True, 0.6), 46.6)
    digital_ticks(48.0, 2.6, 32, 10, 0.9)               # "9,454,457" copies
    # Bach drop
    add("sfx", whoosh(0.7, False, 1.6), 65.05)
    add("sfx", reverse_cymbal(1.4, 1.2), 65.5)
    add("sfx", whoosh(0.4, True, 0.6), 68.8)
    # Industry: radio, montage shutters, fire
    add("sfx", radio_static(1.8, 1.2), 75.8, 1.0, -0.3)
    for a, b in C.MONTAGE_BURSTS:
        for t in [c for c in C.SCENE_CUTS if a <= c <= b]:
            add("sfx", shutter(1.0), t, 1.0, rng.uniform(-.4, .4))
    add("sfx", whoosh(0.4, True, 0.7), 85.9)
    add("sfx", I.low_boom(4.0, 1.0, 70, 28), 90.07)
    # Countdown + liftoff
    for t, f in ((92.55, 880), (93.34, 880), (94.0, 1320)):
        add("sfx", beep(f, 0.15, 1.2), t)
    add("sfx", riser(2.0, 1.4), 92.6)
    add("sfx", rumble(7.0, 1.5), 94.37)
    add("sfx", I.low_boom(4.5, 1.3), 94.57)
    digital_ticks(96.2, 2.6, 34, 8, 0.9)                # "404,911,442"
    # Voyager: transmission bleeps
    for i, t in enumerate(np.arange(99.8, 103.4, 0.3)):
        if rng.random() > 0.35:
            add("sfx", beep(1760 if i % 3 else 2349, 0.06, 0.8), t, 1.0, 0.5)
    digital_ticks(104.1, 1.8, 26, 8, 0.8)               # "4,022"
    for t in (107.6, 108.1):
        add("sfx", shutter(0.6), t, 1.0, 0.2)
    # Machines
    for t in C.MACHINE_HITS:
        add("sfx", servo(0.7, 1.2), t - 0.55, 1.0, rng.uniform(-.5, .5))
    add("sfx", whoosh(1.2, False, 0.8), 116.3)          # Mars wind
    for t in (118.47, 120.07, 121.67, 122.47, 122.87, 123.37, 123.97):
        add("sfx", whoosh(0.3, True, 0.7), t - 0.12, 1.0, rng.uniform(-.5, .5))
    add("sfx", riser(1.6, 1.3), 123.0)
    # One fire
    add("sfx", whoosh(0.8, False, 1.6), 124.55)
    add("sfx", fire_bed(12.0, 25, 1.0), 124.6)
    add("sfx", riser(1.2, 0.9), 125.2)
    add("sfx", reverse_cymbal(1.2, 1.0), 126.3)
    add("sfx", whoosh(1.0, True, 1.3), 133.2)
    add("sfx", I.low_boom(3.0, 1.2), 133.47)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    music()
    sfx()
    for k, v in STEMS.items():
        np.save(os.path.join(out, k + ".npy"), v)
        print(f"{k:9s} peak {np.max(np.abs(v)):.3f}")
