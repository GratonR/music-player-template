"""Synth instruments for Prometheus v3 (numpy only)."""
import numpy as np

SR = 44100
_rng = np.random.default_rng(3)


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def env(n, a, d, s, r):
    a, d, r = max(1, int(a * SR)), max(1, int(d * SR)), max(1, int(r * SR))
    sus = max(n - a - d - r, 0)
    e = np.concatenate([np.linspace(0, 1, a, False), np.linspace(1, s, d, False),
                        np.full(sus, s), np.linspace(s, 0, r)])
    return np.pad(e, (0, max(0, n - len(e))))[:n]


def _t(dur):
    return np.arange(int(dur * SR)) / SR


def bandlimited_saw(f0, t, rolloff=1.0, fmax=9000.0, phase=None, pm=None):
    out = np.zeros_like(t)
    kmax = int(fmax / f0)
    for k in range(1, max(2, kmax)):
        ph = _rng.uniform(0, 6.28) if phase is None else phase * k
        arg = 2 * np.pi * f0 * k * t + ph
        if pm is not None:
            arg = arg + k * pm
        out += np.sin(arg) / k ** rolloff
    return out


# ---------------------------------------------------------------- strings
def string_section(m, dur, a=0.8, r=1.5, voices=6, bright=1.25, gain=1.0):
    t = _t(dur)
    f0 = midi(m)
    out = np.zeros_like(t)
    for v in range(voices):
        det = 1 + _rng.uniform(-0.0045, 0.0045)
        vib = 0.0035 * f0 * np.sin(2 * np.pi * _rng.uniform(4.6, 5.8) * t + _rng.uniform(0, 6.28))
        pm = np.cumsum(vib) / SR * 2 * np.pi * np.clip(t / 0.7, 0, 1)
        out += bandlimited_saw(f0 * det, t, bright, 7000, pm=pm)
    return out * env(len(t), a, 0.4, 0.92, r) * 0.045 * gain / np.sqrt(voices)


def spiccato(m, dur, accent=1.0):
    """Short bowed ostinato note: saw body plus a bow-noise attack."""
    t = _t(dur)
    f0 = midi(m)
    body = sum(bandlimited_saw(f0 * (1 + d), t, 1.15, 6500) for d in (-0.003, 0.003))
    body *= env(len(t), 0.002, 0.045, 0.22, 0.03)
    bow = np.diff(_rng.standard_normal(len(t)), prepend=0) * np.exp(-t * 70) * 0.35
    return (body + bow) * 0.05 * accent


def choir(m, dur, vowel=(650, 1100, 2700), gain=1.0, a=1.0, r=1.6):
    f0 = midi(m)
    t = _t(dur)
    out = np.zeros_like(t)
    for d in (-0.004, 0.0, 0.004):
        vib = 0.005 * f0 * np.sin(2 * np.pi * _rng.uniform(4.8, 5.6) * t)
        pm = np.cumsum(vib) / SR * 2 * np.pi
        for k in range(1, 40):
            f = f0 * k
            if f > 5500:
                break
            w = sum(np.exp(-((f - F) / (80 + F * .08)) ** 2) * g for F, g in zip(vowel, (1, .55, .25)))
            out += (0.1 / k + w) * np.sin(2 * np.pi * f * (1 + d) * t + k * pm + _rng.uniform(0, 6.28))
    breath = _rng.standard_normal(len(t)) * 0.004
    return (out + breath) * env(len(t), a, 0.3, 0.9, r) * 0.03 * gain


def organ(m, dur, gain=1.0):
    f = midi(m)
    t = _t(dur)
    s = sum(a * np.sin(2 * np.pi * f * r * t) for r, a in
            ((0.5, .7), (1, 1), (2, .6), (3, .35), (4, .3), (6, .15), (8, .1)))
    chiff = np.diff(_rng.standard_normal(len(t)), prepend=0) * np.exp(-t * 60) * 0.1
    return (s + chiff) * env(len(t), 0.015, 0.05, 0.9, 0.15) * 0.065 * gain


def brass(m, dur, gain=1.0, a=0.15, r=0.6):
    t = _t(dur)
    e = env(len(t), a, 0.4, 0.8, r)
    f0 = midi(m)
    out = np.zeros_like(t)
    for k in range(1, 24):
        if f0 * k > 6000:
            break
        out += e ** (0.5 + 0.22 * k) * np.sin(2 * np.pi * f0 * k * t + _rng.uniform(0, 6.28)) / k ** .9
    return out * e * 0.075 * gain


def sub(m, dur, gain=1.0, a=0.05, r=0.4):
    t = _t(dur)
    s = np.sin(2 * np.pi * midi(m) * t) + 0.2 * np.sin(4 * np.pi * midi(m) * t)
    return s * env(len(t), a, 0.2, 0.9, r) * 0.22 * gain


def celesta(m, dur=1.6, gain=1.0):
    t = _t(dur)
    f = midi(m)
    s = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * d) for r, a, d in
            ((1, 1, 2.5), (4, .3, 6), (2.0, .2, 4), (7.1, .08, 9)))
    return s * 0.05 * gain


def bell(m, dur=5.0, gain=1.0):
    t = _t(dur)
    f = midi(m)
    s = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * d) for r, a, d in
            ((1, 1, .8), (2.76, .45, 1.6), (5.4, .25, 2.8), (.5, .35, .6), (8.93, .1, 4)))
    return s * 0.055 * gain


# ---------------------------------------------------------------- percussion
def taiko(pitch=62, dur=1.6, gain=1.0):
    t = _t(dur)
    f = pitch * (1 + 1.1 * np.exp(-t * 22))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.5)
    skin = _rng.standard_normal(len(t)) * np.exp(-t * 45) * 0.3
    room = _rng.standard_normal(len(t)) * np.exp(-t * 6) * 0.03
    return (body + skin + room) * 0.42 * gain


def low_boom(dur=4.0, gain=1.0, f_start=90, f_end=32):
    t = _t(dur)
    f = f_end + (f_start - f_end) * np.exp(-t * 3)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.1) * 0.6 * gain


def braam(root, dur=3.5, gain=1.0):
    s = sum(brass(m, dur, 1.6, a=0.03, r=1.2) for m in (root - 24, root - 12, root - 5, root))
    s += low_boom(dur, 0.9)[:len(s)]
    s += _rng.standard_normal(len(s)) * np.exp(-_t(dur)[:len(s)] * 16) * 0.07
    return s * gain


def clock_tick(gain=1.0, pitch=3200):
    t = _t(0.05)
    return (np.sin(2 * np.pi * pitch * t) * 0.5 + np.diff(_rng.standard_normal(len(t)), prepend=0) * 0.5) \
        * np.exp(-t * 140) * 0.05 * gain


def noise(dur):
    return _rng.standard_normal(int(dur * SR))


def hp(x):
    return np.diff(x, prepend=0)
