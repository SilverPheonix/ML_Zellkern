#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os

# ================== SETTINGS ==================
INPUT_IMG        = r"output/test_masking/Li_on_red.png"  # Pfad zum Eingabebild
AXIS             = 1     # 1 = Spaltensumme (vertikale Linien), 0 = Zeilensumme (horizontale Linien)
MIN_DISTANCE     = 1     # Mindestabstand zwischen Peaks in Pixeln
MIN_HEIGHT_ABS   = None  # Fester Schwellenwert für Peak-Höhe (z. B. 3000.0)
MIN_HEIGHT_PCTL  = 60    # Prozentuale Schwelle (0–100), nur aktiv wenn ABS=None
OUT_CURVE        = "peaks_curve.png"
OUT_OVERLAY      = "peaks_overlay.png"
# ==============================================


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

    # --- Profil berechnen (ohne Glättung) ---
    prof = profile_from_image(img, AXIS).astype(float)

    # --- Mindesthöhe bestimmen ---
    if MIN_HEIGHT_ABS is not None:
        height_thr = float(MIN_HEIGHT_ABS)
    else:
        height_thr = float(np.percentile(prof, MIN_HEIGHT_PCTL))
    print(f"Mindesthöhe (Schwelle) = {height_thr:.2f}")

    # --- Peaks finden ---
    peaks = simple_peaks(prof, min_distance=MIN_DISTANCE, min_height=height_thr)

    # --- Kurvenplot speichern UND anzeigen ---
    x = np.arange(len(prof))
    plt.figure(figsize=(10, 5))
    plt.plot(x, prof, label="Profil (unglättet)")
    plt.axhline(height_thr, linestyle=":", label=f"Min-Höhe={height_thr:.1f}")
    if peaks.size:
        plt.scatter(peaks, prof[peaks], color="red", marker="x", s=50,
                    label=f"Peaks n={len(peaks)}")
    plt.xlabel("Index")
    plt.ylabel("Summenintensität")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_CURVE, dpi=150)
    print(f"Kurvenplot gespeichert: {OUT_CURVE}")
    plt.show()  # jetzt wird die Kurve auch angezeigt

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
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_OVERLAY, dpi=150)
    print(f"Overlay gespeichert: {OUT_OVERLAY}")
    plt.show()  #zeigt auch das Overlay an

    # --- Optional: automatisch im Windows-Fotoanzeiger öffnen ---
    if os.name == "nt":  # nur auf Windows
        os.startfile(OUT_CURVE)
        os.startfile(OUT_OVERLAY)

    # --- Konsolenausgabe ---
    if peaks.size:
        print("\nGefundene Peaks (Index -> Wert):")
        for i in peaks:
            print(f"{i}\t{prof[i]:.2f}")
    else:
        print("Keine Peaks gefunden – MIN_DISTANCE oder MIN_HEIGHT_* anpassen.")


if __name__ == "__main__":
    main()
