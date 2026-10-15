"""Measure production techniques section by section, for a reference and a
candidate mix. Prints numbers only: no audio from the reference is written.

    python3 dissect.py ref.wav candidate.wav
"""
import sys

import librosa
import numpy as np

import cues as C

SR = 22050
SECTIONS = [("intro", 0, 7.6), ("greece", 9.9, 22), ("rome", 22.3, 35.4), ("monks", 35.4, 39.3),
            ("cathedral", 39.8, 64.8), ("bach", 65.3, 71), ("industry", 71.2, 91),
            ("space", 94.6, 124.4), ("onefire", 124.9, 130), ("accelerate", 130.5, 136)]
OCT = [(20, 60), (60, 120), (120, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000),
       (4000, 8000), (8000, 11025)]


def load(p):
    y, _ = librosa.load(p, sr=SR, mono=False)
    return y


def measure(y):
    mono = y.mean(0)
    S = np.abs(librosa.stft(mono, n_fft=2048, hop_length=256))
    H, P = librosa.decompose.hpss(S, margin=(1.0, 3.0))
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    t = librosa.times_like(S, sr=SR, hop_length=256)
    Ls, Rs = [np.abs(librosa.stft(c, n_fft=2048, hop_length=256)) for c in y]
    Ms, Ss = np.abs(librosa.stft((y[0] + y[1]) / 2, n_fft=2048, hop_length=256)), \
        np.abs(librosa.stft((y[0] - y[1]) / 2, n_fft=2048, hop_length=256))
    on = librosa.onset.onset_strength(S=librosa.amplitude_to_db(P), sr=SR, hop_length=256)
    flat = librosa.feature.spectral_flatness(S=S)[0]
    chroma = librosa.feature.chroma_stft(S=H ** 2, sr=SR)
    rms = librosa.feature.rms(S=S, frame_length=2048)[0]
    res = {}
    for name, a, b in SECTIONS:
        m = (t >= a) & (t < b)
        if not m.any():
            continue
        r = {}
        tot = (S[:, m] ** 2).sum()
        r["octaves_dB"] = [round(10 * np.log10((S[(f >= lo) & (f < hi)][:, m] ** 2).sum() / tot + 1e-12), 1)
                           for lo, hi in OCT]
        r["perc_share"] = round(float((P[:, m] ** 2).sum() / ((P[:, m] ** 2).sum() + (H[:, m] ** 2).sum())), 3)
        r["width_side_mid_dB"] = round(float(10 * np.log10((Ss[:, m] ** 2).sum() / ((Ms[:, m] ** 2).sum() + 1e-12))), 1)
        r["onset_sharpness"] = round(float(np.percentile(on[m], 95) / (np.median(on[m]) + 1e-9)), 2)
        db = 20 * np.log10(rms[m] + 1e-9)
        r["dyn_range_dB"] = round(float(np.percentile(db, 95) - np.percentile(db, 10)), 1)
        ch = chroma[:, m] / (chroma[:, m].max(0, keepdims=True) + 1e-9)
        r["active_pitch_classes"] = round(float((ch > 0.5).sum(0).mean()), 2)
        r["noisiness"] = round(float(flat[m].mean()), 4)
        # ostinato note length: how fast the percussive envelope decays after onsets
        pe = P[:, m].sum(0)
        pk = librosa.util.peak_pick(pe, pre_max=4, post_max=4, pre_avg=10, post_avg=10, delta=np.median(pe), wait=4)
        decays = []
        for i in pk:
            seg = pe[i:i + 40]
            below = np.where(seg < seg[0] * 0.3)[0]
            if len(below):
                decays.append(below[0] * 256 / SR)
        r["note_decay_ms"] = round(1000 * float(np.median(decays)), 0) if decays else None
        res[name] = r
    # reverb tail: time to fall 20 dB after each drop
    tails = []
    for d in C.DROPS:
        i = np.searchsorted(t, d - 0.3)
        seg = 20 * np.log10(rms[i:i + 200] + 1e-9)
        below = np.where(seg < seg[0] - 20)[0]
        tails.append(round(below[0] * 256 / SR, 2) if len(below) else None)
    res["drop_tail_s"] = tails
    return res


if __name__ == "__main__":
    ref, cand = measure(load(sys.argv[1])), measure(load(sys.argv[2]))
    keys = ["perc_share", "onset_sharpness", "note_decay_ms", "width_side_mid_dB", "dyn_range_dB",
            "active_pitch_classes", "noisiness"]
    for name, _, _ in SECTIONS:
        if name not in ref:
            continue
        print(f"\n== {name}")
        for k in keys:
            print(f"  {k:22s} ref {str(ref[name][k]):>8s}   cand {str(cand[name][k]):>8s}")
        print("  octave balance dB  ref ", ref[name]["octaves_dB"])
        print("                     cand", cand[name]["octaves_dB"])
    print("\ndrop tails (s, -20 dB)  ref", ref["drop_tail_s"], " cand", cand["drop_tail_s"])
