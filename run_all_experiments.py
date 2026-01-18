"""
run_all_experiments.py

Batch über alle Experimente in ./data/<experiment_folder>/

Speichert pro Experiment in:
data/<experiment>/results/
- foci_analysis.csv
- mask.png
- original_red_overlay.png               (wie im Screenshot: roter Hintergrund + Boxen/Foci)
- original_red_overlay_redchannel.png    (gleiche Basis, Foci im Rotkanal zusätzlich geboostet)

Erwartete Dateien:
- *Blue.tif
- *Red.tif
"""

from __future__ import annotations

from pathlib import Path
import traceback
import os, sys, subprocess  # <-- für Explorer-Open

import numpy as np
import pandas as pd
import cv2
# from PIL import Image  # nicht mehr nötig
from skimage import io, filters, color, measure, morphology, feature, segmentation, exposure
from skimage.segmentation import find_boundaries
from scipy import ndimage as nd
from skimage.draw import disk


# -------------------- Settings (Notebook) --------------------
DATA_DIR = Path("data")
OPEN_EXPLORER_AFTER_RUN = True  # <-- Explorer am Ende öffnen

BLUE_GAUSS_SIGMA = 1
WATERSHED_MIN_DISTANCE = 20
MIN_SIZE_FACTOR = 0.65

WINDOW_SIZE = 21
EPSILON = 1e-6
RED_SMOOTH_SIGMA = 1

BLOB_MIN_SIGMA = 2
BLOB_MAX_SIGMA = 5
BLOB_NUM_SIGMA = 10
BLOB_THRESHOLD = 1

MIN_CELL_AREA = 50

# Wie stark der Rotkanal bei FOCI-Pixeln angehoben werden soll (0..255)
REDCHANNEL_FOCI_BOOST = 140

# Darstellung des Rotkanals wie im Screenshot
VIS_P_LOW = 2.0      # unteres Perzentil fürs Stretching
VIS_P_HIGH = 99.8    # oberes Perzentil
CLAHE_CLIP = 2.0     # 0 = aus; 2.0–4.0 macht das Bild „griffiger“
CLAHE_TILE = (8, 8)


# -------------------- Helper --------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def open_in_explorer(path: Path) -> None:
    """Öffnet einen Ordner im Datei-Explorer (Windows/macOS/Linux)."""
    try:
        p = str(path)
        if sys.platform.startswith("win"):
            subprocess.run(["explorer", p], check=False)
        elif sys.platform == "darwin":
            subprocess.run(["open", p], check=False)
        else:
            subprocess.run(["xdg-open", p], check=False)
    except Exception as e:
        print(f"[WARN] Explorer konnte nicht geöffnet werden: {e}", flush=True)


def analyze_cell_sizes(li_mask):
    labeled_li = measure.label(li_mask)
    li_props = measure.regionprops(labeled_li)
    li_areas = [r.area for r in li_props]
    if not li_areas:
        return [], 0.0
    return li_props, float(np.mean(li_areas))


def get_separated_binary_mask(binary_mask, min_distance=WATERSHED_MIN_DISTANCE):
    binary_mask = binary_mask.astype(bool)
    distance = nd.distance_transform_edt(binary_mask)

    coords = feature.peak_local_max(distance, min_distance=min_distance, labels=binary_mask)
    if len(coords) == 0:
        return binary_mask, np.zeros_like(binary_mask, dtype=int), []

    seed_mask = np.zeros(distance.shape, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    markers, _ = nd.label(seed_mask)

    labeled_cells = segmentation.watershed(-distance, markers, mask=binary_mask)
    props = measure.regionprops(labeled_cells)
    areas = [r.area for r in props]

    boundaries = find_boundaries(labeled_cells, mode="thick")
    final_binary_mask = binary_mask.copy()
    final_binary_mask[boundaries] = 0

    return final_binary_mask, labeled_cells, areas


def load_red_channel_uint8(red_path: Path) -> np.ndarray:
    """
    Für die Analyse (Z-Score) nutzen wir weiterhin 8-Bit.
    """
    img = io.imread(str(red_path))
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.uint8)


def load_red_channel_raw(red_path: Path) -> np.ndarray:
    """
    Für die VISUALISIERUNG holen wir die Rohwerte (8/16 Bit) und
    machen dann Percentile-Stretch + optional CLAHE.
    """
    img = io.imread(str(red_path))
    if img.ndim == 3:
        img = img[:, :, 0]
    return img  # dtype kann 8/16 Bit sein


def make_red_visual_canvas(gray_img: np.ndarray) -> np.ndarray:
    """
    Erzeugt ein rotes BGR-Bild wie im Screenshot:
    - Percentile-Stretch (2..99.8 %)
    - optional CLAHE
    - in den roten Kanal legen
    """
    g = gray_img.astype(np.float32)
    lo, hi = np.percentile(g, (VIS_P_LOW, VIS_P_HIGH))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(g)), float(np.max(g) + 1e-6)
    red8 = exposure.rescale_intensity(g, in_range=(lo, hi), out_range=(0, 255)).astype(np.uint8)

    if CLAHE_CLIP and CLAHE_CLIP > 0:
        clahe = cv2.createCLAHE(clipLimit=float(CLAHE_CLIP), tileGridSize=CLAHE_TILE)
        red8 = clahe.apply(red8)

    canvas = np.zeros((red8.shape[0], red8.shape[1], 3), dtype=np.uint8)
    canvas[:, :, 2] = red8  # nur Rotkanal
    return canvas


def preprocess_and_normalize(img_red_raw: np.ndarray, window_size: int, epsilon: float):
    img_red_smoothed = filters.gaussian(img_red_raw, sigma=RED_SMOOTH_SIGMA)

    local_mean = nd.uniform_filter(img_red_smoothed, size=window_size)
    ex2 = nd.uniform_filter(img_red_smoothed ** 2, size=window_size)
    variance = ex2 - local_mean ** 2
    variance = np.clip(variance, 0, None)  # stabil
    local_std = np.sqrt(variance)

    img_red_norm = (img_red_smoothed - local_mean) / (local_std + epsilon)
    return img_red_norm.astype(np.float32), img_red_smoothed.astype(np.float32)


# -------------------- Pipeline pro Experiment --------------------

def analyze_experiment(folder: Path) -> pd.DataFrame:
    exp_name = folder.name
    results_dir = folder / "results"
    ensure_dir(results_dir)

    blue_path = next(folder.glob("*Blue.tif"))
    red_path = next(folder.glob("*Red.tif"))

    # ---- Blue: grayscale + blur + Li threshold ----
    original_blue = io.imread(str(blue_path))
    if original_blue.ndim == 3:
        grayscale_blue = color.rgb2gray(original_blue)
    else:
        grayscale_blue = original_blue.astype(np.float32)

    blurred_blue = filters.gaussian(grayscale_blue, sigma=BLUE_GAUSS_SIGMA)
    li_threshold_value = filters.threshold_li(blurred_blue)
    li_mask_initial = blurred_blue > li_threshold_value

    _, average_area = analyze_cell_sizes(li_mask_initial)
    if average_area <= 0:
        empty = pd.DataFrame()
        # ---- CSV für DE-Excel: Semikolon + Komma-Decimal ----
        empty.to_csv(results_dir / "foci_analysis.csv",
                     index=False, float_format="%.4f", sep=";", decimal=",")
        return empty

    min_size_threshold = int(average_area) * MIN_SIZE_FACTOR
    li_mask_filtered = morphology.remove_small_objects(
        li_mask_initial.astype(bool),
        min_size=int(min_size_threshold),
    )

    # ---- Watershed separation ----
    separated_mask, _, _ = get_separated_binary_mask(li_mask_filtered, min_distance=WATERSHED_MIN_DISTANCE)

    # Output: Maske
    cv2.imwrite(str(results_dir / "mask.png"), (separated_mask.astype(np.uint8) * 255))

    # ---- Red: VISUAL (für Overlay) + ANALYSE (Z-Score) ----
    red_raw_for_visual = load_red_channel_raw(red_path)           # 8/16 Bit möglich
    overlay = make_red_visual_canvas(red_raw_for_visual)          # rotes BGR-Bild
    overlay_redchannel = overlay.copy()

    img_red_orig_u8 = load_red_channel_uint8(red_path)
    img_red_norm, img_red_smoothed = preprocess_and_normalize(img_red_orig_u8, WINDOW_SIZE, EPSILON)

    # ---- Zell-Labeling (wie Notebook): cell_mask aus SMOOTHED * separated_mask ----
    masked_smooth = img_red_smoothed * separated_mask
    cell_mask = masked_smooth > 0
    labeled_cells = measure.label(cell_mask)
    regions = measure.regionprops(labeled_cells, intensity_image=img_red_norm)

    # Maske, in die wir NUR die Foci zeichnen (für Rot-Boost)
    foci_draw_mask = np.zeros(separated_mask.shape, dtype=np.uint8)

    # ---- Foci detection + Zeichnen ----
    foci_records = []
    peak_id_counter = 1

    for region in regions:
        if region.area < MIN_CELL_AREA:
            continue

        cell_avg_intensity = float(region.mean_intensity)
        minr, minc, maxr, maxc = region.bbox
        cropped_intensity = img_red_norm[minr:maxr, minc:maxc]

        blobs_log = feature.blob_log(
            cropped_intensity,
            min_sigma=BLOB_MIN_SIGMA,
            max_sigma=BLOB_MAX_SIGMA,
            num_sigma=BLOB_NUM_SIGMA,
            threshold=BLOB_THRESHOLD,
        )

        if len(blobs_log) == 0:
            # Box/ID trotzdem zeichnen, damit die Zellen sichtbar sind
            for img in (overlay, overlay_redchannel):
                cv2.rectangle(img, (minc, minr), (maxc, maxr), (0, 255, 0), 1)
                cv2.putText(img, f"ID:{region.label}", (minc, max(0, minr - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            continue

        # Boxen/IDs: auf beide Overlays (grün)
        for img in (overlay, overlay_redchannel):
            cv2.rectangle(img, (minc, minr), (maxc, maxr), (0, 255, 0), 1)
            cv2.putText(
                img,
                f"ID:{region.label}",
                (minc, max(0, minr - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

        for local_r, local_c, sigma in blobs_log:
            local_r, local_c = int(local_r), int(local_c)
            global_r, global_c = minr + local_r, minc + local_c

            if labeled_cells[global_r, global_c] != region.label:
                continue

            intensity_at_peak = float(img_red_norm[global_r, global_c])
            if intensity_at_peak <= cell_avg_intensity:
                continue

            radius = float(sigma * np.sqrt(2))
            r_px = int(max(1, radius))

            # Foci zeichnen: auf beide Overlays
            for img in (overlay, overlay_redchannel):
                cv2.circle(img, (global_c, global_r), r_px, (0, 255, 255), 1)  # gelber Ring
                cv2.circle(img, (global_c, global_r), 1, (0, 0, 255), -1)      # roter Punkt

            # NUR Foci in Maske einzeichnen (damit Rot-Boost nur dort passiert)
            cv2.circle(foci_draw_mask, (global_c, global_r), r_px, 255, -1)

            rr, cc = disk((global_r, global_c), max(1, int(radius)), shape=img_red_norm.shape)
            focus_mean_intensity = float(np.mean(img_red_norm[rr, cc]))

            foci_records.append({
                "experiment": exp_name,
                "cell_id": int(region.label),
                "peak_id": int(peak_id_counter),
                "cell_intensity_mean": round(cell_avg_intensity, 4),
                "foci_intensity_peak": round(intensity_at_peak, 4),
                "foci_intensity_mean_disk": round(focus_mean_intensity, 4),
                "foci_radius": round(radius, 2),
                "signal_to_cell_ratio": round(intensity_at_peak / cell_avg_intensity, 2),
                "foci_x": int(global_c),
                "foci_y": int(global_r),
                "cell_area": int(region.area),
            })
            peak_id_counter += 1

    df = pd.DataFrame(foci_records)

    # ---- Rotkanal-Boost NUR dort, wo Foci-Maske gesetzt ist ----
    if np.any(foci_draw_mask):
        red = overlay_redchannel[:, :, 2].astype(np.int16)
        mask_foci = foci_draw_mask > 0
        red[mask_foci] = np.clip(red[mask_foci] + REDCHANNEL_FOCI_BOOST, 0, 255)
        overlay_redchannel[:, :, 2] = red.astype(np.uint8)

    # Output: CSV + Overlays
    df.to_csv(results_dir / "foci_analysis.csv",
              index=False, float_format="%.4f", sep=";", decimal=",")
    cv2.imwrite(str(results_dir / "original_red_overlay.png"), overlay)
    cv2.imwrite(str(results_dir / "original_red_overlay_redchannel.png"), overlay_redchannel)

    return df


def find_experiment_folders(data_dir: Path) -> list[Path]:
    folders = []
    if not data_dir.exists():
        return folders
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and any(p.glob("*Blue.tif")) and any(p.glob("*Red.tif")):
            folders.append(p)
    return folders


def main() -> None:
    folders = find_experiment_folders(DATA_DIR)
    print(f"Gefundene Experimente: {len(folders)}")
    if not folders:
        print("Keine Experiment-Ordner mit *Blue.tif und *Red.tif gefunden.")
        return

    all_rows = []
    failed = []
    last_results_dir: Path | None = None

    for folder in folders:
        try:
            print(f"-> {folder.name}")
            df = analyze_experiment(folder)
            last_results_dir = folder / "results"  # <-- damit wir am Ende den Ordner öffnen können
            if not df.empty:
                all_rows.append(df)
        except Exception as e:
            failed.append((folder.name, repr(e)))
            print(f"!! Fehler in {folder.name}: {e}")
            traceback.print_exc()

    # Optional: Gesamt-CSV im Projektroot
    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        df_all.to_csv("all_foci_analysis.csv",
                      index=False, float_format="%.4f", sep=";", decimal=",")
        print("Gesamt-CSV: all_foci_analysis.csv")

    if failed:
        pd.DataFrame(failed, columns=["experiment", "error"]).to_csv("failed_experiments.csv", index=False)
        print("Fehlerliste: failed_experiments.csv")

    # Explorer am Ende öffnen
    if OPEN_EXPLORER_AFTER_RUN and last_results_dir is not None:
        open_in_explorer(last_results_dir)


if __name__ == "__main__":
    main()
