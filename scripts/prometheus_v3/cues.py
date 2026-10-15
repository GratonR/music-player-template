"""Cue sheet measured on the reference video (seconds).

Only timing is taken from the reference: section boundaries, the note rate of
each section, impact/drop points, scene cuts that need a sound effect.
No melody, harmony voicing or sample from the reference is used.
"""

DUR = 136.5

# (start, end, seconds between ostinato notes, grid anchor)
PULSE = {
    "greece":    (9.94, 22.0, 0.2778, 9.94),    # 72 BPM, triplet eighths
    "rome":      (22.3, 35.4, 0.2778, 22.5),    # same grid, march accents
    "monks":     (35.4, 39.3, 0.2778, 35.57),
    "cathedral": (39.8, 64.8, 0.2325, 39.87),   # ~258 notes/min
    "bach":      (64.9, 71.0, None, None),
    "industry":  (71.2, 91.0, 0.2090, 71.22),   # 72 BPM sixteenths
    "countdown": (91.0, 94.5, 0.395, 92.55),
    "space":     (94.57, 124.4, 0.1975, 94.57),  # ~304 notes/min
    "onefire":   (124.4, 130.0, None, None),
    "accelerate": (130.57, 136.5, 0.5, 130.57),
}

DROPS = [7.6, 64.9, 124.9]                 # music sucked out
IMPACTS = [9.7, 39.8, 66.9, 94.57]         # big hits
MACHINE_HITS = [110.2, 111.8, 113.4, 115.0]
SCENE_CUTS = [0.47, 1.07, 2.2, 2.53, 4.37, 8.47, 24.0, 34.93, 37.1, 41.17, 46.8, 65.3,
              68.97, 75.83, 76.27, 76.67, 77.1, 77.53, 77.97, 78.4, 78.83, 79.27, 86.1,
              87.83, 88.27, 88.67, 89.1, 89.53, 94.37, 96.07, 97.67, 104.07, 111.7,
              115.27, 116.47, 118.47, 120.07, 121.67, 122.47, 122.87, 123.97, 124.77,
              126.27, 133.47]
MONTAGE_BURSTS = [(75.8, 79.3), (87.8, 89.6), (120.0, 124.0)]

# Loudness contour of the reference, dB relative to its loudest 4 s window,
# one value per 4 s. Used only as a mix-automation target.
REF_CONTOUR = [-6.6, -7.7, -6.2, -5.8, -5.9, -4.2, -5.0, -5.5, -6.9, -8.9, -2.8, -2.4,
               -3.0, -3.1, -2.6, -3.0, -4.8, -1.1, -2.3, -2.6, -2.4, -2.4, -1.6, -1.8,
               -0.6, -1.2, -0.6, -1.5, -1.2, -0.3, 0.0, -4.1, -5.3, -4.8]
