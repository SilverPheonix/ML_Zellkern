#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# ---- Settings ----
INPUT_IMG   = r"output/test_masking/Li_on_red.png"
AXIS        = 1          # 0 = Zeilensumme (horiz. Linien), 1 = Spaltensumme (vert. Linien)
SMOOTH_WIN  = 5          # 1 = keine Glättung
MIN_DISTANCE= 20         # Mindestabstand zwischen Peaks
OUT_CURVE   = "peaks_curve.png"
OUT_OVERLAY = "peaks_overlay.png"
# -------------------

def moving_average(x, w):
    if w <= 1: return x
    return np.convolve(x, np.ones(w)/w, mode="same")

def simple_peaks(y, min_distance=10):
    idx = [i for i in range(1, len(y)-1) if y[i] > y[i-1] and y[i] > y[i+1]]
    if min_distance and len(idx) > 1:
        keep = [idx[0]]
        for p in idx[1:]:
            if p - keep[-1] >= min_distance:
                keep.append(p)
        idx = keep
    return np.array(idx, dtype=int)

def profile_from_image(img, axis):
    # 2D: (H, W)
    if img.ndim == 2:
        return img.sum(axis=0) if axis == 1 else img.sum(axis=1)
    # 3D: (H, W, C) -> Spaltensumme = Summe über Zeilen UND Kanäle, übrig bleibt W
    #                    Zeilensumme = Summe über Spalten UND Kanäle, übrig bleibt H
    if img.ndim == 3:
        return img.sum(axis=(0, 2)) if axis == 1 else img.sum(axis=(1, 2))
    # fallback
    raise ValueError("Unerwartete Bildform.")

def main():
    img = io.imread(INPUT_IMG)
    print(f"Bildform: {img.shape}")

    prof = profile_from_image(img, AXIS).astype(float)
    smooth = moving_average(prof, SMOOTH_WIN)
    peaks = simple_peaks(smooth, MIN_DISTANCE)

    # Kurvenplot
    x = np.arange(len(prof))
    plt.figure(figsize=(10,5))
    plt.plot(x, prof, label="Profil")
    plt.plot(x, smooth, label=f"Glättung={SMOOTH_WIN}")
    if peaks.size:
        plt.scatter(peaks, smooth[peaks], color="red", marker="x", s=50, label=f"Peaks n={len(peaks)}")
    plt.xlabel("Index"); plt.ylabel("Summe"); plt.legend(); plt.tight_layout()
    plt.savefig(OUT_CURVE, dpi=150); plt.close()
    print(f"Kurvenplot gespeichert: {OUT_CURVE}")

    # Overlay auf dem Originalbild (unverändert)
    plt.figure(figsize=(10,6))
    plt.imshow(img)
    if peaks.size:
        if AXIS == 1:     # Spaltensumme -> vertikale Linien (x in [0..W-1])
            for p in peaks: plt.axvline(p, color="cyan", linestyle="--", linewidth=1.2)
        else:             # Zeilensumme -> horizontale Linien (y in [0..H-1])
            for p in peaks: plt.axhline(p, color="cyan", linestyle="--", linewidth=1.2)
    plt.title("Peaks (Originalbild, unverändert)")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(OUT_OVERLAY, dpi=150); plt.show()
    print(f"Overlay gespeichert: {OUT_OVERLAY}")

if __name__ == "__main__":
    main()
