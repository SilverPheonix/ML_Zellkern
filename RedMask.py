#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Minimal: Li-Threshold -> Maske -> auf Rotbild anwenden + automatisch anzeigen
# Benötigt: numpy, scikit-image, matplotlib

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters


# ---- Pfade anpassen (falls nötig) ----
INPUT_GRAY = r"output/test_masking/8_Blue.tif"     # Graubild für die Maskenerzeugung
INPUT_RED  = r"output/test_masking/8_Red.tif"      # Farbbild, auf das die Maske angewendet wird
OUT_MASK   = r"output/test_masking/Li_mask.png"    # Binärmaske (weiß=1, schwarz=0)
OUT_APPLY  = r"output/test_masking/Li_on_red.png"  # Rotbild mit Maske
# --------------------------------------

def show_image(img, title, cmap=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def load_gray(path: str) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 3:  # RGB oder RGBA → Grauwert
        img = color.rgb2gray(img[:, :, :3])
    return img.astype(np.float32, copy=False)

def apply_mask_to_rgb(mask: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Maske (bool) auf 3-kanaliges Bild anwenden -> andere Pixel = 0."""
    if rgb.ndim == 2:
        rgb = np.stack([rgb]*3, axis=-1)
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    out = np.where(mask3, rgb, 0)
    return out.astype(rgb.dtype)

def main():
    # Eingaben prüfen
    for p in [INPUT_GRAY, INPUT_RED]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {p}")

    # 1️⃣ Graubild laden
    gray = load_gray(INPUT_GRAY)
    show_image(gray, "Graubild", cmap="gray")

    # 2️⃣ Li-Threshold -> Maske
    thr = filters.threshold_li(gray)
    mask = gray > thr
    mask_u8 = (mask.astype(np.uint8) * 255)
    io.imsave(OUT_MASK, mask_u8)
    print(f"Maske gespeichert: {OUT_MASK}")
    show_image(mask, "Li-Maske", cmap="gray")

    # 3️⃣ Rotbild laden und Maske anwenden
    red_img = io.imread(INPUT_RED)
    masked_red = apply_mask_to_rgb(mask, red_img)
    io.imsave(OUT_APPLY, masked_red)
    print(f"Maskiertes Rotbild gespeichert: {OUT_APPLY}")
    show_image(masked_red, "Li-Maske auf Rotbild")

if __name__ == "__main__":
    main()
