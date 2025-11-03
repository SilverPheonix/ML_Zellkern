#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# ================== SETTINGS ==================
INPUT_IMG        = r"output/test_masking/Li_on_red.png"  # Bild (wird unverändert angezeigt)
AXIS             = 1     # 1 = Spaltensumme -> vertikale Linien, 0 = Zeilensumme -> horizontale Linien
SMOOTH_WIN       = 0     # 1 = keine Glättung, größere Werte -> weniger Peaks
MIN_DISTANCE     = 1    # Mindestabstand zwischen Peaks (Pixel entlang der Profilachse)

# Mindesthöhe eines Peaks:
MIN_HEIGHT_ABS   = None  # z.B. 2500.0 (fester Wert), oder None für Perzentil
MIN_HEIGHT_PCTL  = 60    # nur verwendet, wenn ABS=None (0..100). Höher -> weniger Peaks

OUT_CURVE        = "peaks_curve.png"
OUT_OVERLAY      = "peaks_overlay.png"
# ==============================================


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def simple_peaks(y: np.ndarray, min_distance: int = 10, min_height: float | None = None) -> np.ndarray:
    """Einfache Peak-Suche: lokale Maxima + optional Mindesthöhe + Mindestabstand."""
    idx = [i for i in range(1, len(y) - 1) if y[i] > y[i - 1] and y[i] > y[i + 1]]
    if min_height is not None:
        idx = [i for i in idx if y[i] >= min_height]
    if min_distance and len(idx) > 1:
        keep = [idx[0]]
        for p in idx[1:]:
            if p - keep[-1] >= min_distance:
                keep.append(p)
        idx = keep
    return np.array(idx, dtype=int)


def profile_from_image(img: np.ndarray, axis: int) -> np.ndarray:
    """
    2D (H,W):   axis=1 -> Summe über Zeilen (ergibt Länge W)
                axis=0 -> Summe über Spalten (ergibt Länge H)
    3D (H,W,C): zusätzlich über Kanäle summieren.
    """
    if img.ndim == 2:
        return img.sum(axis=0) if axis == 1 else img.sum(axis=1)
    if img.ndim == 3:
        return img.sum(axis=(0, 2)) if axis == 1 else img.sum(axis=(1, 2))
    raise ValueError("Unerwartete Bildform.")


def main():
    # --- Bild laden ---
    img = io.imread(INPUT_IMG)
    print(f"Bildform: {img.shape}")

    # --- Profil & Glättung ---
    prof = profile_from_image(img, AXIS).astype(float)
    smooth = moving_average(prof, SMOOTH_WIN)

    # --- Mindesthöhe bestimmen ---
    if MIN_HEIGHT_ABS is not None:
        height_thr = float(MIN_HEIGHT_ABS)
    else:
        height_thr = float(np.percentile(smooth, MIN_HEIGHT_PCTL))
    print(f"Mindesthöhe (Schwelle) = {height_thr:.2f}")

    # --- Peaks finden ---
    peaks = simple_peaks(smooth, min_distance=MIN_DISTANCE, min_height=height_thr)

    # --- Kurvenplot speichern ---
    x = np.arange(len(prof))
    plt.figure(figsize=(10, 5))
    plt.plot(x, prof, label="Profil")
    plt.plot(x, smooth, label=f"Glättung={SMOOTH_WIN}")
    plt.axhline(height_thr, linestyle=":", label=f"Min-Höhe={height_thr:.1f}")
    if peaks.size:
        plt.scatter(peaks, smooth[peaks], color="red", marker="x", s=50,
                    label=f"Peaks n={len(peaks)}")
    plt.xlabel("Index"); plt.ylabel("Summenintensität")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_CURVE, dpi=150)
    print(f"Kurvenplot gespeichert: {OUT_CURVE}")
    plt.close()

    # --- Overlay (auf Originalbild, unverändert) ---
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    if peaks.size:
        if AXIS == 1:
            for p in peaks:
                plt.axvline(p, color="cyan", linestyle="--", linewidth=1.2)
        else:
            for p in peaks:
                plt.axhline(p, color="cyan", linestyle="--", linewidth=1.2)
    plt.title("Peaks (Originalbild, unverändert)")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(OUT_OVERLAY, dpi=150)
    plt.show()
    print(f"Overlay gespeichert: {OUT_OVERLAY}")

    # --- Konsolenausgabe ---
    if peaks.size:
        print("Gefundene Peaks (Index -> Wert):")
        for i in peaks:
            print(f"{i}\t{smooth[i]:.2f}")
    else:
        print("Keine Peaks gefunden – SMOOTH_WIN/MIN_DISTANCE/MIN_HEIGHT_* anpassen.")


if __name__ == "__main__":
    main()
