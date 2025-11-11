from pathlib import Path
import math
import csv

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure

# ----------------------------------------------------------------------
# Einstellungen
# ----------------------------------------------------------------------

# Bild aus RedMask.py:
# vorher: python RedMask.py ausführen → Li_on_red.png wird erzeugt
INPUT_IMG = r"output/test_masking/Li_on_red.png"

# minimale Fläche (in Pixeln), damit eine Region als Peak zählt
# für ganze Zellkerne eher größer wählen (z. B. 500–2000)
MIN_AREA = 1000

# Faktor, um den Kreis etwas größer zu machen als die Region
RADIUS_SCALE = 1.1


# ----------------------------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------------------------


def analyze_peaks(mask: np.ndarray, intensity_image: np.ndarray, min_area: int = 5):
    """
    Nimmt eine Binärmaske (mask) und das Intensitätsbild (intensity_image)
    und berechnet pro Peak (zusammenhängende Region):

      - x, y: Koordinate des Peak-Punkts (hellster Pixel)  <-- X
      - cx, cy: Schwerpunkt (Centroid) der Region          <-- Kreiszentrum
      - area_px: Anzahl Pixel
      - radius_px: äquivalenter Kreisradius (mit RADIUS_SCALE skaliert)
      - intensity_sum: Summe der Intensitäten aller Peak-Pixel
      - intensity_mean: Durchschnittsintensität
    """
    labels = measure.label(mask.astype(bool), connectivity=1)
    regions = measure.regionprops(labels, intensity_image=intensity_image)

    peaks = []
    for region_id, r in enumerate(regions, start=1):
        area = int(r.area)
        if area < min_area:
            continue

        # Schwerpunkt (für Kreiszentrum)
        cy, cx = r.centroid  # (Zeile, Spalte)

        # hellster Pixel in der Region (Peak-Punkt)
        coords = r.coords  # Nx2, (row, col)
        intensities = intensity_image[coords[:, 0], coords[:, 1]]
        max_idx = np.argmax(intensities)
        peak_y, peak_x = coords[max_idx]  # globale Koordinaten (Zeile, Spalte)

        intensity_sum = float(intensities.sum())
        intensity_mean = intensity_sum / area if area > 0 else 0.0

        # Kreisradius aus Fläche + leicht größer skalieren
        radius = math.sqrt(area / math.pi) * RADIUS_SCALE

        peaks.append(
            {
                "id": region_id,
                "x": float(peak_x),   # Peak (für X)
                "y": float(peak_y),
                "cx": float(cx),      # Kreiszentrum
                "cy": float(cy),
                "area_px": area,
                "radius_px": float(radius),
                "intensity_sum": intensity_sum,
                "intensity_mean": intensity_mean,
            }
        )

    return labels, peaks


def save_peak_stats(peaks, out_csv_path: str):
    """
    Speichert die Peak-Infos in eine CSV-Datei.
    Nur die Infos, die du explizit wolltest.
    """
    fieldnames = [
        "index",           # Nummer im Bild (1,2,3,…)
        "x",
        "y",
        "area_px",
        "radius_px",
        "intensity_sum",
        "intensity_mean",
    ]
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, p in enumerate(peaks, start=1):
            writer.writerow(
                {
                    "index": idx,
                    "x": p["x"],
                    "y": p["y"],
                    "area_px": p["area_px"],
                    "radius_px": p["radius_px"],
                    "intensity_sum": p["intensity_sum"],
                    "intensity_mean": p["intensity_mean"],
                }
            )


def build_side_text(peaks):
    """
    Baut den Textblock für die Übersicht am Rand.
    Eine Zeile pro Peak:
      Nr: x=.., y=.., r=.., pix=.., ΣI=.., μ=..
    """
    lines = []
    for idx, p in enumerate(peaks, start=1):
        line = (
            f"{idx}: "
            f"x={p['x']:.0f}, y={p['y']:.0f}, "
            f"r={p['radius_px']:.1f}, "
            f"pix={p['area_px']}, "
            f"ΣI={p['intensity_sum']:.1f}, "
            f"μ={p['intensity_mean']:.3f}"
        )
        lines.append(line)
    return "\n".join(lines)


def draw_peak_overlays(base_image: np.ndarray, peaks, out_dir: Path, prefix: str):
    """
    Erzeugt 3 Overlay-Bilder auf dem roten Bild:

      1) nur X (grün) + Nummern + Übersichtstext am Rand
      2) X + Kreis (Zellkern) + Nummern + Übersichtstext am Rand
      3) nur Kreise + Nummern (ohne Textblock)
    """

    # ----------------- 1) nur X + Nummern + Textblock -----------------
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.imshow(base_image)
    for idx, p in enumerate(peaks, start=1):
        # X auf Peak-Punkt
        ax1.plot(p["x"], p["y"], marker="x", markersize=10,
                 color="lime", alpha=0.9)
        # Nummer neben das X
        ax1.text(
            p["x"] + 5,
            p["y"] + 5,
            str(idx),
            fontsize=10,
            color="white",
            alpha=0.9,
            va="top",
            ha="left",
        )

    side_text = build_side_text(peaks)
    # Textblock oben rechts INS Bild setzen (dort ist meist dunkel)
    ax1.text(
        0.99,
        0.01,
        side_text,
        transform=ax1.transAxes,
        fontsize=8,
        color="yellow",
        alpha=0.9,
        va="bottom",
        ha="right",
        linespacing=1.3,
    )

    ax1.set_title("Peaks – nur X (nummeriert) + Werte")
    ax1.axis("off")
    fig1.tight_layout()
    out1 = out_dir / f"{prefix}_cross.png"
    fig1.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # ----------------- 2) X + Kreis + Textblock -----------------
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.imshow(base_image)
    for idx, p in enumerate(peaks, start=1):
        # X auf Peak
        ax2.plot(p["x"], p["y"], marker="x", markersize=10,
                 color="lime", alpha=0.9)
        ax2.text(
            p["x"] + 5,
            p["y"] + 5,
            str(idx),
            fontsize=10,
            color="white",
            alpha=0.9,
            va="top",
            ha="left",
        )
        # Kreis um gesamten Zellkern (Zentrum = Schwerpunkt)
        circle = plt.Circle(
            (p["cx"], p["cy"]),
            p["radius_px"],
            edgecolor="lime",
            facecolor="none",
            linewidth=1.5,
            alpha=0.9,
        )
        ax2.add_patch(circle)

    side_text = build_side_text(peaks)
    ax2.text(
        0.99,
        0.01,
        side_text,
        transform=ax2.transAxes,
        fontsize=8,
        color="yellow",
        alpha=0.9,
        va="bottom",
        ha="right",
        linespacing=1.3,
    )

    ax2.set_title("Peaks – X + Foci-Kreis (nummeriert) + Werte")
    ax2.axis("off")
    fig2.tight_layout()
    out2 = out_dir / f"{prefix}_circle_cross.png"
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # ----------------- 3) nur Kreise + Nummern -----------------
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.imshow(base_image)
    for idx, p in enumerate(peaks, start=1):
        circle = plt.Circle(
            (p["cx"], p["cy"]),
            p["radius_px"],
            edgecolor="lime",
            facecolor="none",
            linewidth=1.5,
            alpha=0.9,
        )
        ax3.add_patch(circle)
        ax3.text(
            p["cx"] + 5,
            p["cy"] + 5,
            str(idx),
            fontsize=10,
            color="yellow",
            alpha=0.9,
            va="top",
            ha="left",
        )

    ax3.set_title("Peaks – nur Foci-Kreise (nummeriert)")
    ax3.axis("off")
    fig3.tight_layout()
    out3 = out_dir / f"{prefix}_circle.png"
    fig3.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close(fig3)

    return out1, out2, out3


# ----------------------------------------------------------------------
# Hauptlogik
# ----------------------------------------------------------------------


def main():
    path = Path(INPUT_IMG)
    if not path.exists():
        raise FileNotFoundError(f"Eingangsbild nicht gefunden: {path}")

    out_dir = path.parent
    print(f"Verwende Bild: {path}")
    print(f"Ausgabeordner: {out_dir}")

    # Bild einlesen
    img = io.imread(path)

    # Rotkanal / Intensitätsbild bestimmen
    if img.ndim == 2:
        red_raw = img.astype(float)
        h, w = img.shape
        # Overlay-Bild: Graubild als roten Kanal anzeigen
        overlay_img = np.zeros((h, w, 3), dtype=float)
        max_val = red_raw.max() if red_raw.max() > 0 else 1.0
        overlay_img[..., 0] = red_raw / max_val
    else:
        # RGB-Bild: Rotkanal verwenden
        red_raw = img[..., 0].astype(float)
        overlay_img = img.astype(float)
        max_val = overlay_img.max() if overlay_img.max() > 0 else 1.0
        if max_val > 1.0:
            overlay_img /= max_val  # auf 0..1 skalieren

    # Intensitätsbild für Statistik = originaler Rotkanal
    intensity_image = red_raw

    # Maske: alles, was überhaupt rot ist (>0) → ganze Zellkerne
    mask = red_raw > 0

    # Peak-Analyse
    labels, peaks = analyze_peaks(mask, intensity_image, min_area=MIN_AREA)

    print("\n--- Peak-Statistik ---")
    print(f"Gefundene Peaks (nach Flächenfilter, MIN_AREA={MIN_AREA}): {len(peaks)}")
    for idx, p in enumerate(peaks, start=1):
        print(
            f"{idx}: "
            f"x={p['x']:.1f}, y={p['y']:.1f}, "
            f"Pix={p['area_px']}, r={p['radius_px']:.2f}, "
            f"SumI={p['intensity_sum']:.3f}, "
            f"MeanI={p['intensity_mean']:.4f}"
        )

    # CSV mit allen Kennzahlen
    out_csv = out_dir / f"{path.stem}_peaks2d.csv"
    save_peak_stats(peaks, str(out_csv))
    print(f"\nPeak-Statistik gespeichert in: {out_csv}")

    # Overlays auf dem roten Bild erzeugen
    cross_img, circle_cross_img, circle_img = draw_peak_overlays(
        overlay_img,
        peaks,
        out_dir,
        prefix=f"{path.stem}_peaks2d",
    )

    print("\nOverlay-Bilder gespeichert:")
    print("  nur X (+Werte):     ", cross_img)
    print("  Kreis + X (+Werte): ", circle_cross_img)
    print("  nur Kreise (+Num.): ", circle_img)


if __name__ == "__main__":
    main()
