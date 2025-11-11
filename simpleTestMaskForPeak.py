#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, exposure, img_as_ubyte, img_as_uint

# ===== Einstellungen =====
INPUT = r"output/test_masking/8_Red.tif"  # DEIN Bild
OUT_DIR = r"output/test_masking"
BLUR = 3                  # 0 = kein Blur, sonst 3/5
METHOD = "otsu"           # "otsu" | "yen" | "li" | "triangle" | "percentile"
PERCENTILE = 90           # nur bei METHOD="percentile"
INVERT_SIGNAL = False     # True, wenn Signal dunkel auf hellem Hintergrund
USE_ALL_PAGES = True      # True: Mehrseitige TIFs mitteln
# =========================

def to_float01(arr):
    arr = arr.astype(np.float64)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr[:] = 0.0
    return arr

def load_gray(path: Path):
    img = io.imread(str(path))
    # Mehrseitige TIFs: (frames, y, x)
    if img.ndim == 3 and img.shape[-1] not in (3,4):
        if USE_ALL_PAGES:
            img = img.mean(axis=0)
        else:
            img = img[0]
    # RGB/RGBA?
    if img.ndim == 3 and img.shape[-1] in (3,4):
        img = color.rgb2gray(img)  # 0..1 float
        return img
    # single-channel int/float
    return img

def auto_threshold(gray01, method="otsu"):
    if method == "otsu":
        t = filters.threshold_otsu(gray01)
    elif method == "yen":
        t = filters.threshold_yen(gray01)
    elif method == "li":
        t = filters.threshold_li(gray01)
    elif method == "triangle":
        t = filters.threshold_triangle(gray01)
    elif method == "percentile":
        t = np.percentile(gray01, PERCENTILE)
    else:
        t = filters.threshold_otsu(gray01)
    return gray01 > t

def main():
    path = Path(INPUT)
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        raise FileNotFoundError(f"Bild nicht gefunden: {path}")

    raw = load_gray(path)
    print(f"Shape: {raw.shape}, dtype: {raw.dtype}, min/max: {raw.min()} / {raw.max()}")

    # Für Anzeige/Threshold auf 0..1 normalisieren
    if raw.dtype.kind in ("u", "i", "f"):
        gray01 = to_float01(raw)
    else:
        gray01 = to_float01(raw.astype(np.float64))

    # optional invertieren (falls Signal dunkel ist)
    if INVERT_SIGNAL:
        gray01 = 1.0 - gray01

    # Blur (nur für Threshold)
    if BLUR and BLUR >= 3 and BLUR % 2 == 1:
        from scipy.ndimage import gaussian_filter
        blur_sigma = (BLUR - 1) / 6.0  # grober Zusammenhang für 3x3/5x5
        gray_for_thr = gaussian_filter(gray01, sigma=blur_sigma)
    else:
        gray_for_thr = gray01

    # Maske
    mask = auto_threshold(gray_for_thr, METHOD)

    # Maskiertes Bild – einmal als Preview (8-bit, autocontrast) und einmal originalgetreu (16-bit wenn möglich)
    masked_float = gray01.copy()
    masked_float[~mask] = 0.0

    # Preview (skaliert für Anzeige, 8-bit)
    preview = exposure.rescale_intensity(masked_float, in_range="image", out_range=(0,1))
    preview_u8 = img_as_ubyte(preview)

    # Originalgetreu speichern (wenn Eingabe 16-bit, behalten wir 16-bit)
    if raw.dtype == np.uint16:
        masked_raw = raw.copy()
        masked_raw[~mask] = 0
        masked_save = masked_raw  # 16-bit
        out_masked = out_dir / f"{path.stem}_masked16.tif"
        io.imsave(out_masked, masked_save)
        print(f"Maskiert (16-bit) gespeichert: {out_masked}")
    else:
        # auf 8-bit runter (Preview genügt oft)
        out_masked = out_dir / f"{path.stem}_masked8.png"
        io.imsave(out_masked, preview_u8)
        print(f"Maskiert (8-bit Preview) gespeichert: {out_masked}")

    # Maske speichern (8-bit)
    mask_u8 = (mask.astype(np.uint8) * 255)
    out_mask = out_dir / f"{path.stem}_mask.png"
    io.imsave(out_mask, mask_u8)
    print(f"Maske gespeichert: {out_mask}")

    # Debug-Plot (mit Autokontrast, damit nichts „schwarz aussieht“)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(exposure.equalize_hist(gray01), cmap="gray")
    ax[0].set_title("Original (Auto-Contrast)")
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title(f"Maske ({METHOD})")
    ax[2].imshow(preview_u8, cmap="gray")
    ax[2].set_title("Maskiert (Preview)")
    for a in ax: a.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
