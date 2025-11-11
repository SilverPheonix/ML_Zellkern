from pathlib import Path
import math
import csv

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, measure, feature

# ----------------------------------------------------------------------
# Einstellungen
# ----------------------------------------------------------------------

# Bild aus RedMask.py:
INPUT_IMG = r"output/test_masking/Li_on_red.png"

# Zellkern-Filter: minimale Fläche pro Kern (in Pixeln)
MIN_AREA_NUCLEUS = 1000     # ggf. anpassen, z.B. 800–2000

# Foci-Detection innerhalb jedes Kerns
FOCI_THRESHOLD_REL = 0.6    # relative Schwelle pro Kern (0..1)
FOCI_MIN_DISTANCE = 3       # minimaler Abstand zwischen Foci (Pixel)

# Radius-Skalierung für Kern-Kreis
RADIUS_SCALE = 1.1


# ----------------------------------------------------------------------
# Zellkern-Analyse
# ----------------------------------------------------------------------


def analyze_nuclei(mask: np.ndarray, intensity_image: np.ndarray, min_area: int = 5):
    """
    Mask = Binärbild für Zellkerne (alles was rot ist).
    intensity_image = Intensitäten (Rotkanal).

    Liefert:
      nuclei: Liste von Dicts mit
        - id: Label-ID
        - cx, cy: Schwerpunkt (Kreiszentrum)
        - area_px: Fläche
        - radius_px: Kreisradius (skaliert)
        - intensity_sum: Summe Intensität in Kern
        - intensity_mean: mittlere Intensität
        - bbox: Bounding-Box (minr, minc, maxr, maxc)
        - num_foci: wird später mit Anzahl Foci gefüllt
    """
    labels = measure.label(mask.astype(bool), connectivity=1)
    regions = measure.regionprops(labels, intensity_image=intensity_image)

    nuclei = []
    for region_id, r in enumerate(regions, start=1):
        area = int(r.area)
        if area < min_area:
            continue

        cy, cx = r.centroid
        intensities = r.intensity_image[r.image]  # nur innerhalb Region
        intensity_sum = float(intensities.sum())
        intensity_mean = intensity_sum / area if area > 0 else 0.0

        radius = math.sqrt(area / math.pi) * RADIUS_SCALE

        nuclei.append(
            {
                "id": region_id,
                "cx": float(cx),
                "cy": float(cy),
                "area_px": area,
                "radius_px": float(radius),
                "intensity_sum": intensity_sum,
                "intensity_mean": intensity_mean,
                "bbox": r.bbox,      # (minr, minc, maxr, maxc)
                "label": region_id,  # für Foci-Zuordnung
                "num_foci": 0,       # wird später gesetzt
            }
        )

    return labels, nuclei


# ----------------------------------------------------------------------
# Foci-Analyse innerhalb der Kerne
# ----------------------------------------------------------------------


def detect_foci_in_nucleus(
    intensity_image: np.ndarray,
    labels: np.ndarray,
    nucleus_label: int,
    threshold_rel: float,
    min_distance: int,
):
    """
    Sucht lokale Maxima (Foci) innerhalb eines Zellkerns (labels == nucleus_label).
    Verwendet skimage.feature.peak_local_max auf dem Ausschnitt.

    Gibt eine Liste von (y, x) in globalen Bildkoordinaten zurück.
    """
    mask_nucleus = labels == nucleus_label
    if not np.any(mask_nucleus):
        return []

    # Bounding Box des Kerns
    coords = np.column_stack(np.nonzero(mask_nucleus))
    minr, minc = coords.min(axis=0)
    maxr, maxc = coords.max(axis=0) + 1  # slicing-ende

    subimg = intensity_image[minr:maxr, minc:maxc]
    submask = mask_nucleus[minr:maxr, minc:maxc]

    # Alles außerhalb des Kerns auf 0 setzen
    subimg_masked = subimg.copy()
    subimg_masked[~submask] = 0

    if subimg_masked.max() <= 0:
        return []

    # relative Schwelle pro Kern
    thr_abs = threshold_rel * subimg_masked.max()

    coords_local = feature.peak_local_max(
        subimg_masked,
        min_distance=min_distance,
        threshold_abs=thr_abs,
        exclude_border=False,
    )

    # in globale Koordinaten umrechnen
    foci_global = [(int(minr + y), int(minc + x)) for (y, x) in coords_local]
    return foci_global


# ----------------------------------------------------------------------
# CSV + Textblock
# ----------------------------------------------------------------------


def save_nucleus_stats(nuclei, out_csv_path: str):
    """
    Speichert pro Zellkern die Kern-Infos in eine CSV-Datei.
    """
    fieldnames = [
        "index",           # Nummer im Bild (1..N)
        "cx",
        "cy",
        "area_px",
        "radius_px",
        "intensity_sum",
        "intensity_mean",
        "num_foci",
    ]
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, n in enumerate(nuclei, start=1):
            writer.writerow(
                {
                    "index": idx,
                    "cx": n["cx"],
                    "cy": n["cy"],
                    "area_px": n["area_px"],
                    "radius_px": n["radius_px"],
                    "intensity_sum": n["intensity_sum"],
                    "intensity_mean": n["intensity_mean"],
                    "num_foci": n["num_foci"],
                }
            )


def build_side_text(nuclei):
    """
    Textblock am Rand: eine Zeile pro Zellkern.
    Enthält jetzt zusätzlich die Anzahl der Foci.
    """
    lines = []
    for idx, n in enumerate(nuclei, start=1):
        line = (
            f"{idx}: "
            f"cx={n['cx']:.0f}, cy={n['cy']:.0f}, "
            f"r={n['radius_px']:.1f}, "
            f"pix={n['area_px']}, "
            f"ΣI={n['intensity_sum']:.1f}, "
            f"μ={n['intensity_mean']:.3f}, "
            f"foci={n['num_foci']}"
        )
        lines.append(line)
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Overlay-Zeichnung
# ----------------------------------------------------------------------


def draw_overlay(base_image, nuclei, nuclei_foci, out_dir: Path, prefix: str):
    """
    Erzeugt das Bild:
      - grüne Kreise um Zellkerne
      - Zellkern-Nummer am Kreis
      - Foci innerhalb der Kerne als transparente X
      - Textblock mit Kern-Infos (inkl. Foci-Anzahl) am Rand
    nuclei_foci: Liste gleicher Länge wie nuclei, pro Kern Liste von (y, x).
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(base_image)

    # 1) Foci: kleine, transparente X
    for foci in nuclei_foci:
        for (y, x) in foci:
            ax.plot(
                x,
                y,
                marker="x",
                markersize=6,
                color="lime",
                alpha=0.5,  # durchsichtig
            )

    # 2) Zellkern-Kreise + Nummern
    for idx, n in enumerate(nuclei, start=1):
        circle = plt.Circle(
            (n["cx"], n["cy"]),
            n["radius_px"],
            edgecolor="lime",
            facecolor="none",
            linewidth=1.5,
            alpha=0.9,
        )
        ax.add_patch(circle)
        ax.text(
            n["cx"] + 5,
            n["cy"] + 5,
            str(idx),
            fontsize=10,
            color="yellow",
            alpha=0.9,
            va="top",
            ha="left",
        )

    # 3) Textblock am Rand mit Kern-Infos + Foci-Anzahl
    side_text = build_side_text(nuclei)
    ax.text(
        0.99,
        0.01,
        side_text,
        transform=ax.transAxes,
        fontsize=8,
        color="yellow",
        alpha=0.9,
        va="bottom",
        ha="right",
        linespacing=1.3,
    )

    ax.set_title("Foci (lokale Maxima)")
    ax.axis("off")
    fig.tight_layout()
    out_path = out_dir / f"{prefix}_nuclei_foci.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path


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

    # Rotkanal / Overlay-Bild
    if img.ndim == 2:
        red_raw = img.astype(float)
        h, w = img.shape
        overlay_img = np.zeros((h, w, 3), dtype=float)
        max_val = red_raw.max() if red_raw.max() > 0 else 1.0
        overlay_img[..., 0] = red_raw / max_val
    else:
        red_raw = img[..., 0].astype(float)
        overlay_img = img.astype(float)
        max_val = overlay_img.max() if overlay_img.max() > 0 else 1.0
        if max_val > 1.0:
            overlay_img /= max_val

    intensity_image = red_raw

    # Maske für Zellkerne: alles was rot ist
    mask_nuclei = red_raw > 0

    # 1) Kerne analysieren
    labels, nuclei = analyze_nuclei(mask_nuclei, intensity_image, min_area=MIN_AREA_NUCLEUS)
    print(f"Gefundene Zellkerne (nach Flächenfilter): {len(nuclei)}")

    # 2) Foci pro Kern suchen
    nuclei_foci = []
    for n in nuclei:
        foci = detect_foci_in_nucleus(
            intensity_image,
            labels,
            n["label"],
            threshold_rel=FOCI_THRESHOLD_REL,
            min_distance=FOCI_MIN_DISTANCE,
        )
        nuclei_foci.append(foci)
        n["num_foci"] = len(foci)   # hier Anzahl Foci speichern

    total_foci = sum(len(f) for f in nuclei_foci)

    # Ausgabe Konsole
    for idx, n in enumerate(nuclei, start=1):
        print(
            f"Kern {idx}: "
            f"cx={n['cx']:.1f}, cy={n['cy']:.1f}, "
            f"Pix={n['area_px']}, r={n['radius_px']:.2f}, "
            f"SumI={n['intensity_sum']:.1f}, "
            f"MeanI={n['intensity_mean']:.3f}, "
            f"Foci={n['num_foci']}"
        )
    print(f"Gesamtzahl Foci (alle Kerne): {total_foci}")

    # 3) CSV mit Kern-Infos (inkl. num_foci)
    out_csv = out_dir / f"{path.stem}_nuclei_stats.csv"
    save_nucleus_stats(nuclei, str(out_csv))
    print(f"Kern-Statistik gespeichert in: {out_csv}")

    # 4) Overlay-Bild erzeugen
    overlay_path = draw_overlay(
        overlay_img,
        nuclei,
        nuclei_foci,
        out_dir,
        prefix=f"{path.stem}",
    )

    print("\nOverlay-Bild gespeichert:")
    print("  ", overlay_path)


if __name__ == "__main__":
    main()
