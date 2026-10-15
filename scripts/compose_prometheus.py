"""Compose "Prometheus" — an original 2:16 cinematic piece tracing the West,
from ancient Greece to space. Pure numpy synthesis, no samples.

    python3 scripts/compose_prometheus.py out.wav

Tempo is fixed at 90 BPM (51 bars of 4/4 = 136 s); intensity comes from
rhythmic density growing section by section, not from tempo changes.
"""
import sys
import wave

import numpy as np

SR = 44100
BPM = 90
BEAT = 60 / BPM
BAR = 4 * BEAT
BARS = 51
DUR = BARS * BAR + 4.0  # reverb tail
N = int(DUR * SR)
rng = np.random.default_rng(7)

L = np.zeros(N)
R = np.zeros(N)

# Section map (start bar, length in bars)
GREECE, ROME, MEDIEVAL, RENAISSANCE, INDUSTRY, SPACE, OUTRO = (
    (0, 8), (8, 8), (16, 6), (22, 8), (30, 10), (40, 8), (48, 3))


def t_of(bar, beat=0.0):
    return bar * BAR + beat * BEAT


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def add(sig, start, gain=1.0, pan=0.0):
    i = int(start * SR)
    if i >= N:
        return
    sig = sig[: N - i]
    gl = np.cos((pan + 1) * np.pi / 4) * gain
    gr = np.sin((pan + 1) * np.pi / 4) * gain
    L[i:i + len(sig)] += sig * gl
    R[i:i + len(sig)] += sig * gr


def env_adsr(n, a, d, s, r):
    a, d, r = int(a * SR), int(d * SR), int(r * SR)
    sus = max(n - a - d - r, 0)
    e = np.concatenate([
        np.linspace(0, 1, a, endpoint=False),
        np.linspace(1, s, d, endpoint=False),
        np.full(sus, s),
        np.linspace(s, 0, r),
    ])
    return np.pad(e, (0, max(0, n - len(e))))[:n]


def additive(freq, dur, amps, decay=None, vib=0.0, vib_rate=5.0, detune=0.0, inharm=0.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    phase_mod = vib * np.sin(2 * np.pi * vib_rate * t) * np.clip(t / 0.6, 0, 1)
    out = np.zeros(n)
    for k, a in enumerate(amps, start=1):
        f = freq * k * (1 + inharm * k * k) * (1 + detune)
        if f > SR / 2.2 or a == 0:
            continue
        p = 2 * np.pi * f * t + k * phase_mod + rng.uniform(0, 2 * np.pi)
        h = a * np.sin(p)
        if decay is not None:
            h *= np.exp(-t * decay * (1 + 0.35 * k))
        out += h
    return out


def saw_amps(n_h, rolloff=1.0):
    return [1 / k ** rolloff for k in range(1, n_h + 1)]


# ---------- instruments ----------

def lyre(m, dur=2.5):
    s = additive(midi(m), dur, [1, .5, .33, .2, .12, .08, .05], decay=1.6, inharm=1e-4)
    att = np.clip(np.arange(len(s)) / (0.004 * SR), 0, 1)
    return s * att * 0.35


def pad(m, dur, bright=10, a=1.5, r=2.0):
    n = int(dur * SR)
    s = sum(additive(midi(m), dur, saw_amps(bright, 1.3), detune=d) for d in (-0.004, 0, 0.005))
    return s * env_adsr(n, a, 0.5, 0.8, r) * 0.08


def choir(m, dur, vowel=(700, 1220, 2600)):
    f0 = midi(m)
    amps = []
    for k in range(1, 40):
        f = f0 * k
        w = sum(np.exp(-((f - F) / (90 + F * 0.08)) ** 2) * g for F, g in zip(vowel, (1, .5, .25)))
        amps.append((0.15 / k + w) if f < 5000 else 0)
    n = int(dur * SR)
    s = sum(additive(f0, dur, amps, vib=0.004 * f0 / f0, vib_rate=r, detune=d)
            for d, r in ((-0.003, 4.8), (0.003, 5.3)))
    return s * env_adsr(n, 1.2, 0.4, 0.85, 1.8) * 0.06


def brass(m, dur, gain=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    e = env_adsr(n, 0.12, 0.3, 0.75, 0.5)
    out = np.zeros(n)
    f0 = midi(m)
    for k in range(1, 24):
        if f0 * k > 7000:
            break
        # brightness tracks the envelope, like a real brass swell
        out += (e ** (0.6 + 0.25 * k)) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)) / k ** 0.9
    return out * e * 0.09 * gain


def string_stab(m, dur):
    n = int(dur * SR)
    s = sum(additive(midi(m), dur, saw_amps(14, 1.2), detune=d) for d in (-0.003, 0.003))
    return s * env_adsr(n, 0.01, 0.12, 0.45, 0.08) * 0.07


def bell(m, dur=4.0):
    f = midi(m)
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = np.zeros(n)
    for ratio, a, dec in ((1, 1, 1.0), (2.76, .5, 1.8), (5.4, .3, 3.0), (8.93, .2, 4.5), (0.5, .4, 0.7)):
        s += a * np.sin(2 * np.pi * f * ratio * t) * np.exp(-t * dec)
    return s * 0.08


def bass(m, dur):
    n = int(dur * SR)
    s = additive(midi(m), dur, [1, .4, .2, .1])
    return s * env_adsr(n, 0.02, 0.2, 0.8, 0.15) * 0.22


def taiko(dur=1.2, pitch=70):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = pitch * (1 + 1.5 * np.exp(-t * 30))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 5)
    slap = rng.standard_normal(n) * np.exp(-t * 60) * 0.3
    return (body + slap) * 0.5


def kick():
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    f = 48 + 120 * np.exp(-t * 35)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7) * 0.55


def noise_hit(dur, decay, hp=True):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = rng.standard_normal(n)
    if hp:
        x = np.diff(x, prepend=0)  # crude high-pass
    return x * np.exp(-t * decay)


def hat():
    return noise_hit(0.12, 45) * 0.05


def snare():
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 25)
    return (noise_hit(0.35, 14) * 0.6 + tone * 0.5) * 0.22


def anvil():
    # industrial metallic hit
    b = bell(84, 1.2) * 1.5
    b[: int(0.2 * SR)] += noise_hit(0.2, 40) * 0.04
    return b


def riser(dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = rng.standard_normal(n)
    # sweep brightness with cumulative differencing mix
    bright = np.diff(x, prepend=0)
    mix = (t / dur)
    return (x * (1 - mix) * 0.3 + bright * mix) * (t / dur) ** 2 * 0.12


# ---------- harmony ----------
# D minor epic loop: Dm  Bb  F  C   (i VI III VII)
PROG = [(50, [62, 65, 69]), (46, [62, 65, 70]), (41, [60, 65, 69]), (48, [60, 64, 67])]
DORIAN_ARP = [62, 69, 74, 72, 69, 71, 74, 76]   # D dorian lyre figure

# ---------------- GREECE: lyre + soft pad, half-time, bars 0-7 ----------------
b0, nb = GREECE
for bar in range(b0, b0 + nb):
    root, chord = PROG[(bar // 2) % 4] if bar >= 4 else (50, [62, 69])
    for i, m in enumerate(DORIAN_ARP):
        if bar < 2 and i % 2:
            continue
        add(lyre(m + (12 if bar % 4 == 3 and i == 6 else 0)), t_of(bar, i * 0.5), 1.0, -0.3 + 0.08 * i)
    if bar % 2 == 0:
        for m in chord:
            add(pad(m - 12, 2 * BAR + 0.5, bright=6, a=2.5), t_of(bar), 0.7 * min(1, bar / 4 + 0.3))
    if bar >= 4 and bar % 2 == 0:
        add(bass(root - 12, 2 * BAR), t_of(bar), 0.6)

# ---------------- ROME: war drums + brass, bars 8-15 ----------------
b0, nb = ROME
for bar in range(b0, b0 + nb):
    root, chord = PROG[(bar // 2) % 4]
    for beat in (0, 1.5, 2, 3):
        add(taiko(), t_of(bar, beat), 1.0 if beat in (0, 2) else 0.6, rng.uniform(-.3, .3))
    if bar % 2 == 0:
        for j, m in enumerate(chord):
            add(brass(m - 12, 2 * BAR - 0.1, gain=1 + 0.1 * (bar - b0)), t_of(bar), 1.0, -0.4 + 0.4 * j)
        add(bass(root - 12, 2 * BAR), t_of(bar), 0.9)
    # march fanfare on horns
    if bar % 4 == 3:
        for k, m in enumerate([62, 65, 69, 74]):
            add(brass(m, BEAT * 0.9, 1.3), t_of(bar, k), 0.9, 0.2)
    for i, m in enumerate(DORIAN_ARP[::2]):
        add(lyre(m), t_of(bar, i), 0.5, 0.5)

# ---------------- MEDIEVAL: choir drone + bells, bars 16-21 ----------------
b0, nb = MEDIEVAL
add(choir(38, nb * BAR + 1), t_of(b0), 1.2, 0)          # D drone
add(choir(45, nb * BAR + 1), t_of(b0), 0.9, 0.2)        # fifth
melody = [(62, 2), (64, 1), (65, 1), (64, 2), (62, 2), (60, 2), (62, 2), (65, 2), (67, 2),
          (69, 4), (67, 2), (65, 2), (64, 4)]
pos = 0.0
for m, d in melody:
    add(choir(m, d * BEAT + 0.6), t_of(b0, pos), 0.9, -0.2)
    pos += d
for bar in range(b0, b0 + nb):
    add(bell(74 if bar % 2 == 0 else 69), t_of(bar), 0.8, 0.6)
    add(taiko(2.0, 55), t_of(bar), 0.5)

# ---------------- RENAISSANCE: strings ostinato, brighter, bars 22-29 ----------------
b0, nb = RENAISSANCE
REN_PROG = [(53, [65, 69, 72]), (48, [64, 67, 72]), (50, [62, 65, 69]), (46, [62, 65, 70])]  # F C Dm Bb
for bar in range(b0, b0 + nb):
    root, chord = REN_PROG[(bar - b0) // 2 % 4]
    arp = [chord[0], chord[1], chord[2], chord[1] + 12, chord[2], chord[1], chord[0] + 12, chord[1]]
    for i, m in enumerate(arp):
        add(string_stab(m, BEAT * 0.45), t_of(bar, i * 0.5), 1.0, 0.35 if i % 2 else -0.35)
    if (bar - b0) % 2 == 0:
        for m in chord:
            add(pad(m, 2 * BAR + 0.3, bright=12, a=0.8), t_of(bar), 0.8)
        add(bass(root - 12, 2 * BAR), t_of(bar), 0.8)
    add(taiko(1.0, 80), t_of(bar), 0.4)
    add(taiko(1.0, 80), t_of(bar, 2), 0.3)
    if bar >= b0 + 4:   # counter-melody on lyre, now octave up
        for i, m in enumerate(arp[::2]):
            add(lyre(m + 12, 1.5), t_of(bar, i + 0.25), 0.5, 0.0)

# ---------------- INDUSTRY: machine pulse, 16ths, anvils, bars 30-39 ----------------
b0, nb = INDUSTRY
for bar in range(b0, b0 + nb):
    root, chord = PROG[(bar - b0) // 2 % 4]
    intensity = (bar - b0) / nb
    for s in range(16):
        m = [root, root + 12, root + 7, root + 12][s % 4]
        add(string_stab(m, BEAT * 0.22), t_of(bar, s * 0.25), 0.9 + 0.4 * intensity, -0.2 if s % 2 else 0.2)
        add(hat(), t_of(bar, s * 0.25), 0.6 + (0.6 if s % 4 == 2 else 0), 0.4)
    for beat in range(4):
        add(kick(), t_of(bar, beat), 1.0)
    add(snare(), t_of(bar, 1), 0.9, 0.05)
    add(snare(), t_of(bar, 3), 0.9, 0.05)
    add(anvil(), t_of(bar, 2.5), 0.7, -0.5)
    if (bar - b0) % 2 == 0:
        for j, m in enumerate(chord):
            add(brass(m, 2 * BAR - 0.1, gain=1 + intensity), t_of(bar), 0.8, -0.4 + 0.4 * j)
        add(bass(root - 24, 2 * BAR), t_of(bar), 1.0)
add(riser(2 * BAR), t_of(b0 + nb - 2), 1.0)
add(taiko(1.5, 60), t_of(b0 + nb - 1, 3), 1.2)
add(taiko(1.5, 60), t_of(b0 + nb - 1, 3.5), 1.2)

# ---------------- SPACE: full climax, choir + brass + wide pads, bars 40-47 ----------------
b0, nb = SPACE
add(noise_hit(4.0, 0.8) * 0.1, t_of(b0), 1.0)   # crash swell
for bar in range(b0, b0 + nb):
    root, chord = PROG[(bar - b0) // 2 % 4]
    if (bar - b0) % 2 == 0:
        for j, m in enumerate(chord):
            add(pad(m, 2 * BAR + 1.0, bright=16, a=0.4), t_of(bar), 1.1, -0.6 + 0.6 * j)
            add(choir(m, 2 * BAR + 0.5), t_of(bar), 0.9, 0.6 - 0.6 * j)
            add(brass(m - 12, 2 * BAR - 0.1, 1.6), t_of(bar), 0.9)
        add(bass(root - 24, 2 * BAR), t_of(bar), 1.1)
        add(bass(root - 12, 2 * BAR), t_of(bar), 0.6)
    for beat in (0, 0.75, 1.5, 2, 3, 3.5):
        add(taiko(1.2, 65), t_of(bar, beat), 1.0 if beat in (0, 2) else 0.65, rng.uniform(-.4, .4))
    add(kick(), t_of(bar), 0.8)
    for s in range(8):
        add(lyre(chord[s % 3] + 24, 1.2), t_of(bar, s * 0.5), 0.45, 0.7 if s % 2 else -0.7)
# soaring theme
theme = [(74, 3), (72, 1), (69, 4), (70, 3), (72, 1), (74, 4), (77, 2), (76, 2), (74, 2), (72, 2), (74, 8)]
pos = 0.0
for m, d in theme:
    add(brass(m, d * BEAT, 1.4), t_of(b0, pos), 1.0, 0.1)
    add(choir(m, d * BEAT + 0.4), t_of(b0, pos), 0.8, -0.1)
    pos += d

# ---------------- OUTRO: D major resolution (Picardy) + bell, bars 48-50 ----------------
b0, nb = OUTRO
for m in (50, 57, 62, 66, 69, 74):
    add(pad(m, nb * BAR + 3, bright=8, a=0.3, r=3.5), t_of(b0), 0.9)
add(choir(62, nb * BAR + 2), t_of(b0), 0.7)
add(choir(66, nb * BAR + 2), t_of(b0), 0.6)
add(taiko(3.0, 50), t_of(b0), 1.3)
add(bell(74, 8.0), t_of(b0), 1.2, 0.3)
add(bell(81, 8.0), t_of(b0, 2), 0.7, -0.3)
for i, m in enumerate(DORIAN_ARP[:4]):
    add(lyre(m + 12, 3.0), t_of(b0 + 1, i), 0.5, -0.3 + 0.2 * i)


# ---------- mix: tone shaping + reverb + master ----------

def fft_lowpass(x, cutoff):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / np.sqrt(1 + (f / cutoff) ** 4)
    return np.fft.irfft(X, len(x))


def reverb(x, seconds=3.2, seed=0):
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = r.standard_normal(n) * np.exp(-t * 6.9 / seconds)
    ir = fft_lowpass(ir, 5000)
    ir[: int(0.02 * SR)] = 0  # pre-delay
    ir /= np.sqrt(np.sum(ir ** 2))
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[: len(x)]
    return y


L = fft_lowpass(L, 9000)
R = fft_lowpass(R, 9000)
wetL, wetR = reverb(L, seed=1), reverb(R, seed=2)
L = L * 0.8 + wetL * 0.45
R = R * 0.8 + wetR * 0.45

out = np.stack([L, R], axis=1)
out /= np.max(np.abs(out)) + 1e-9
out = np.tanh(out * 1.6) / np.tanh(1.6)       # gentle glue / saturation
fade = int(3.5 * SR)
out[-fade:] *= np.linspace(1, 0, fade)[:, None]
out[: int(0.05 * SR)] *= np.linspace(0, 1, int(0.05 * SR))[:, None]
out *= 0.89

pcm = (out * 32767).astype("<i2")
with wave.open(sys.argv[1] if len(sys.argv) > 1 else "prometheus.wav", "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print(f"{BARS} bars @ {BPM} BPM, {N / SR:.1f}s rendered")
