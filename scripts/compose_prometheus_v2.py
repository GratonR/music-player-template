"""Prometheus v2: original cinematic score built on the measured map of the
reference video (tempo, key, drops, impacts, acceleration).

    python3 scripts/compose_prometheus_v2.py out.wav

Map (seconds) taken from the analysis:
  pulse 144 BPM (felt at 72), 152 BPM from 94.6 s to 124.9 s
  D minor -> D major
  drops at 7.6, 64.9, 124.9 ; impacts at 9.7, 39.8, 66.9, 94.6
  machine pulses every 1.6 s from 110.2 s to 115 s
The only quoted material is public domain: the opening figure of Bach's
Toccata in D minor BWV 565 (at "BACH") and the four-note motif of
Beethoven's Fifth Symphony (at "BEETHOVEN").
"""
import sys
import wave

import numpy as np

SR = 44100
DUR = 136.5
N = int((DUR + 4) * SR)
rng = np.random.default_rng(11)
L = np.zeros(N)
R = np.zeros(N)


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def add(sig, start, gain=1.0, pan=0.0):
    i = int(start * SR)
    if i >= N or i < 0:
        return
    sig = sig[: N - i]
    L[i:i + len(sig)] += sig * gain * np.cos((pan + 1) * np.pi / 4)
    R[i:i + len(sig)] += sig * gain * np.sin((pan + 1) * np.pi / 4)


def env(n, a, d, s, r):
    a, d, r = max(1, int(a * SR)), max(1, int(d * SR)), max(1, int(r * SR))
    sus = max(n - a - d - r, 0)
    e = np.concatenate([np.linspace(0, 1, a, False), np.linspace(1, s, d, False),
                        np.full(sus, s), np.linspace(s, 0, r)])
    return np.pad(e, (0, max(0, n - len(e))))[:n]


def partials(f0, dur, amps, decay=None, detune=0.0, vib=0.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    pm = vib * np.sin(2 * np.pi * 5.2 * t) * np.clip(t / 0.8, 0, 1)
    out = np.zeros(n)
    for k, a in enumerate(amps, 1):
        f = f0 * k * (1 + detune)
        if a == 0 or f > 9000:
            continue
        h = a * np.sin(2 * np.pi * f * t + k * pm + rng.uniform(0, 6.28))
        if decay:
            h *= np.exp(-t * decay * (1 + 0.4 * k))
        out += h
    return out


SAW = [1 / k for k in range(1, 30)]


# ---------- instruments ----------
def strings(m, dur, a=0.6, r=1.2, voices=5, gain=1.0):
    n = int(dur * SR)
    s = sum(partials(midi(m), dur, [x ** 1.25 for x in SAW], detune=d, vib=0.003)
            for d in np.linspace(-0.006, 0.006, voices))
    return s * env(n, a, 0.3, 0.9, r) * 0.05 * gain / np.sqrt(voices)


def ostinato_note(m, dur):
    n = int(dur * SR)
    s = sum(partials(midi(m), dur, [x ** 1.1 for x in SAW[:18]], detune=d) for d in (-0.004, 0.004))
    return s * env(n, 0.005, 0.08, 0.35, 0.05) * 0.05


def choir(m, dur, gain=1.0):
    f0 = midi(m)
    amps = []
    for k in range(1, 35):
        f = f0 * k
        w = sum(np.exp(-((f - F) / (80 + F * .08)) ** 2) * g for F, g in ((650, 1), (1100, .55), (2700, .25)))
        amps.append(0.12 / k + w if f < 5000 else 0)
    n = int(dur * SR)
    s = sum(partials(f0, dur, amps, detune=d, vib=0.005) for d in (-0.004, 0, 0.004))
    return s * env(n, 1.0, 0.3, 0.9, 1.5) * 0.035 * gain


def organ(m, dur):
    # drawbar-style pipe organ: 16', 8', 4', 2 2/3', 2'
    f = midi(m)
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = np.zeros(n)
    for ratio, a in ((0.5, .7), (1, 1), (2, .6), (3, .35), (4, .3), (6, .15), (8, .1)):
        s += a * np.sin(2 * np.pi * f * ratio * t)
    return s * env(n, 0.02, 0.05, 0.9, 0.12) * 0.07


def brass(m, dur, gain=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    e = env(n, 0.15, 0.4, 0.8, 0.6)
    f0 = midi(m)
    out = np.zeros(n)
    for k in range(1, 22):
        if f0 * k > 6000:
            break
        out += e ** (0.5 + 0.22 * k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)) / k ** .9
    return out * e * 0.08 * gain


def braam(root, dur=3.5):
    # low brass cluster + sub drop: the trailer "impact"
    s = sum(brass(m, dur, 1.6) for m in (root - 24, root - 12, root - 5, root))
    n = len(s)
    t = np.arange(n) / SR
    sub = np.sin(2 * np.pi * np.cumsum(55 * (1 + np.exp(-t * 4))) / SR) * np.exp(-t * 1.2) * 0.5
    hit = rng.standard_normal(n) * np.exp(-t * 18) * 0.08
    return s + sub + hit


def sub(m, dur, gain=1.0):
    n = int(dur * SR)
    return partials(midi(m), dur, [1, .25, .08]) * env(n, 0.05, 0.2, 0.9, 0.3) * 0.25 * gain


def drum(pitch=60, dur=1.4, gain=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = pitch * (1 + 1.2 * np.exp(-t * 25))
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 4) +
            rng.standard_normal(n) * np.exp(-t * 50) * 0.25) * 0.45 * gain


def tick():
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    return np.diff(rng.standard_normal(n), prepend=0) * np.exp(-t * 80) * 0.03


def riser(dur, gain=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    x = rng.standard_normal(n)
    sweep = np.sin(2 * np.pi * np.cumsum(200 + 1800 * (t / dur) ** 2) / SR) * 0.05
    return (np.diff(x, prepend=0) * 0.1 + sweep) * (t / dur) ** 2.5 * gain


def fire(dur):
    # crackle bed for the Prometheus flame
    n = int(dur * SR)
    x = np.zeros(n)
    idx = rng.integers(0, n, int(dur * 25))
    x[idx] = rng.uniform(-1, 1, len(idx))
    k = np.exp(-np.arange(int(0.004 * SR)) / 40)
    return np.convolve(x, k, 'same') * 0.08


def bell(m, dur=5.0):
    f = midi(m)
    n = int(dur * SR)
    t = np.arange(n) / SR
    return sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * d)
               for r, a, d in ((1, 1, .8), (2.76, .45, 1.6), (5.4, .25, 2.8), (.5, .35, .6))) * 0.06


# ---------- harmony ----------
Dm, Bb, F, C, Gm, A = [50, 62, 65, 69], [46, 62, 65, 70], [41, 60, 65, 69], [48, 60, 64, 67], [43, 62, 67, 70], [45, 61, 64, 69]
D, G, Bm, Amaj = [50, 62, 66, 69], [43, 62, 67, 71], [47, 62, 66, 71], [45, 61, 64, 69]
MINOR = [Dm, Bb, F, C]
MAJOR = [D, Bm, G, Amaj]


def pulse_grid(t0, t1, bpm):
    step = 60 / bpm / 2  # eighth notes
    return np.arange(t0, t1 - 1e-6, step), step


def ostinato(t0, t1, bpm, chords, bar_beats=8, gain=1.0, octave=0, density=1):
    grid, step = pulse_grid(t0, t1, bpm)
    pat = [0, 2, 1, 2, 0, 3, 1, 2]
    for i, t in enumerate(grid):
        if density == 0 and i % 2:
            continue
        ch = chords[(i // (bar_beats * 2)) % len(chords)]
        m = ch[1 + pat[i % 8] % 3] + octave
        add(ostinato_note(m, step * 0.9), t, gain * (1.1 if i % 4 == 0 else 0.8), -0.3 if i % 2 else 0.3)


def pad_prog(t0, t1, chords, chord_len, gain=1.0, with_choir=False, voices=5):
    t = t0
    i = 0
    while t < t1 - 0.5:
        ch = chords[i % len(chords)]
        d = min(chord_len, t1 - t) + 0.8
        for j, m in enumerate(ch[1:]):
            add(strings(m, d, voices=voices, gain=gain), t, 1.0, -0.5 + 0.5 * j)
            if with_choir:
                add(choir(m, d, gain), t, 1.0, 0.5 - 0.5 * j)
        add(sub(ch[0] - 12, d, gain), t, 1.0)
        t += chord_len
        i += 1


THEME = [(69, 2), (74, 2), (72, 1), (70, 1), (69, 2), (65, 2), (67, 1), (69, 3)]  # original, in beats at 72


def theme(t0, bpm, inst, transpose=0, gain=1.0):
    b = 60 / bpm * 2  # felt beat (half-time)
    t = t0
    for m, d in THEME:
        add(inst(m + transpose, d * b * 0.98), t, gain, 0.1)
        t += d * b


# ================= 0:00 - 0:07.6  PROMETHEUS =================
add(fire(9.0), 0.0, 1.0)
add(strings(38, 8.0, a=0.8, r=1.0, gain=2.4), 0.0, 1.0)
add(strings(45, 8.0, a=1.2, r=1.0, gain=1.8), 0.0, 1.0)
add(choir(62, 7.8, 1.4), 0.3, 1.0)
add(sub(38, 7.6, 1.5), 0.0)
add(bell(74), 0.4, 1.0, 0.3)
ostinato(1.0, 7.4, 144, [Dm], gain=0.9, octave=12, density=0)
add(riser(2.0, 0.6), 5.5)
# drop 7.6 -> 9.7: only the flame
add(fire(2.2), 7.6, 0.6)

# ================= 0:09.7 - 0:24  GREECE =================
add(braam(50), 9.7, 0.9)
pad_prog(9.7, 24.0, MINOR, 3.33)
ostinato(9.7, 24.0, 144, MINOR, gain=0.8)
theme(13.0, 144, lambda m, d: strings(m, d, a=0.15, r=0.4, voices=3, gain=1.3), gain=1.0)
for t in np.arange(9.7, 24, 1.667):
    add(drum(70, gain=0.5), t)

# ================= 0:24 - 0:39.8  ROME -> MONKS =================
pad_prog(24.0, 36.0, [Dm, Gm, Bb, A], 3.0, gain=1.1)
ostinato(24.0, 36.0, 144, [Dm, Gm, Bb, A], gain=0.9)
for t in np.arange(24.0, 36.0, 0.833):
    add(drum(62, gain=0.9 if int((t - 24) / 0.833) % 2 == 0 else 0.55), t, 1.0, rng.uniform(-.3, .3))
for k, t in enumerate(np.arange(24.0, 36.0, 3.0)):
    for m in (Dm, Gm, Bb, A)[k % 4][1:]:
        add(brass(m - 12, 2.8, 1.2), t)
# monks: thin out, lone choir
add(choir(62, 4.2, 1.2), 36.0, 1.0)
add(choir(57, 4.2, 0.8), 36.0, 1.0)
add(riser(3.6, 1.0), 36.2)

# ================= 0:39.8 - 0:64.9  CATHEDRAL -> UNIVERSITY -> GALILEO =================
add(braam(50, 4.5), 39.8, 1.1)
pad_prog(39.8, 64.6, [Dm, Bb, F, C, Gm, Bb, A, A], 3.33, gain=1.2, with_choir=True)
ostinato(39.8, 64.6, 144, [Dm, Bb, F, C, Gm, Bb, A, A], gain=1.0)
theme(43.1, 144, lambda m, d: brass(m, d, 1.3), gain=1.0)
theme(56.4, 144, lambda m, d: strings(m + 12, d, a=0.1, r=0.4, voices=4, gain=1.2), gain=1.0)
for t in np.arange(39.8, 64.6, 1.667):
    add(drum(66, gain=0.8), t)
    add(drum(90, 0.6, 0.35), t + 0.625, 1.0, 0.4)
for t in pulse_grid(39.8, 64.6, 144)[0]:
    add(tick(), t, 1.0, 0.6)
# drop 64.9 -> 66.9: silence then a single bell
add(bell(81, 3.0), 65.3, 0.6)

# ================= 0:66.9  BACH (Toccata in D minor, BWV 565, public domain) =================
t = 66.9
toccata = [(81, .12), (79, .12), (81, .9), (79, .14), (77, .14), (76, .14), (74, .14), (73, .6), (74, 1.4)]
for m, d in toccata:
    add(organ(m, d + 0.05), t, 2.2)
    add(organ(m - 12, d + 0.05), t, 1.6)
    t += d
for m, g in ((26, 2.0), (38, 2.4), (50, 1.8), (53, 1.4), (57, 1.4), (62, 1.0)):
    add(organ(m, 3.6), 69.1, g)
add(braam(50, 3.0), 66.9, 0.6)
# BEETHOVEN (Fifth Symphony motif, public domain) at ~70.5 s, in D minor: A A A F
t = 70.6
for m, d in ((69, .22), (69, .22), (69, .22), (65, 1.3)):
    for o in (-12, 0):
        add(strings(m + o, d, a=0.01, r=0.15, voices=5, gain=2.0), t, 1.0)
    t += d + 0.03

# ================= 1:12 - 1:34.6  WORK -> RADIO -> PLAGUES -> DNA -> FIRE =================
pad_prog(72.5, 94.4, [Dm, Bb, F, C], 3.33, gain=1.3, with_choir=True)
ostinato(72.5, 94.4, 144, [Dm, Bb, F, C], gain=1.15)
ostinato(79.8, 94.4, 144, [Dm, Bb, F, C], gain=0.6, octave=12)
add(braam(50, 3.0), 79.8, 0.8)
for t in np.arange(72.5, 94.4, 0.833):
    add(drum(58, gain=0.85), t)
for t in pulse_grid(72.5, 94.4, 144)[0]:
    add(tick(), t, 1.2, -0.5)
theme(86.0, 144, lambda m, d: brass(m, d, 1.6), gain=1.0)
add(riser(5.0, 1.4), 89.6)
# countdown: three drum hits before liftoff
for k, t in enumerate((92.6, 93.4, 94.0)):
    add(drum(55, 1.0, 0.9 + 0.2 * k), t)

# ================= 1:34.6 - 2:04.9  SPACE, 152 BPM, major =================
add(braam(50, 5.0), 94.6, 1.3)
add(rng.standard_normal(int(4 * SR)) * np.exp(-np.arange(int(4 * SR)) / SR * 1.0) * 0.05, 94.6)
pad_prog(94.6, 124.6, MAJOR, 60 / 152 * 8, gain=1.4, with_choir=True)
ostinato(94.6, 124.6, 152, MAJOR, gain=1.2)
ostinato(102.0, 124.6, 152, MAJOR, gain=0.7, octave=12)
theme(98.0, 152, lambda m, d: brass(m, d, 1.8), transpose=0, gain=1.0)
theme(108.4, 152, lambda m, d: choir(m + 12, d, 1.4), gain=1.0)
for t in np.arange(94.6, 124.6, 60 / 152 * 2):
    add(drum(60, gain=0.9), t, 1.0, rng.uniform(-.3, .3))
for t in pulse_grid(94.6, 124.6, 152)[0]:
    add(tick(), t, 1.4, 0.5)
# MACHINES: heavy pulses every 1.6 s
for t in (110.2, 111.8, 113.4, 115.0):
    add(braam(50, 1.5), t, 0.8)
    add(drum(45, 1.2, 1.3), t)
theme(116.0, 152, lambda m, d: strings(m + 12, d, a=0.05, r=0.3, voices=6, gain=1.6), gain=1.0)
add(riser(4.0, 1.6), 120.8)

# ================= 2:04.9 - 2:16.5  ONE FIRE -> ACCELERATE =================
add(fire(12.0), 124.9, 1.0)
add(bell(74, 8.0), 125.4, 1.0, 0.2)
for m in (50, 57, 62, 66, 69):
    add(strings(m, 11.0, a=2.0, r=3.0, gain=1.1), 126.0, 1.0)
add(choir(66, 9.0, 1.2), 128.0)
add(choir(69, 9.0, 1.0), 128.0)
add(sub(38, 10.0), 126.0)
add(braam(50, 4.0), 132.6, 0.7)
add(bell(86, 5.0), 132.6, 0.6, -0.3)


# ---------- mix ----------
def lowpass(x, fc):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(X / np.sqrt(1 + (f / fc) ** 4), len(x))


def reverb(x, sec, seed):
    r = np.random.default_rng(seed)
    n = int(sec * SR)
    t = np.arange(n) / SR
    ir = lowpass(r.standard_normal(n) * np.exp(-t * 6.9 / sec), 4500)
    ir[:int(0.03 * SR)] = 0
    ir /= np.sqrt(np.sum(ir ** 2))
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    return np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[:len(x)]


L, R = lowpass(L, 10000), lowpass(R, 10000)
L, R = L * 0.75 + reverb(L, 4.0, 1) * 0.5, R * 0.75 + reverb(R, 4.0, 2) * 0.5
out = np.stack([L, R], 1)[: int(DUR * SR)]
out /= np.max(np.abs(out)) + 1e-9
out = np.tanh(out * 2.0) / np.tanh(2.0)
f = int(2.5 * SR)
out[-f:] *= np.linspace(1, 0, f)[:, None]
out *= 0.9
with wave.open(sys.argv[1] if len(sys.argv) > 1 else "prometheus_v2.wav", "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((out * 32767).astype("<i2").tobytes())
print(f"rendered {DUR}s")
