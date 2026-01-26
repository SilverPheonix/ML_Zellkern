"""
AUTOMATED FOCI ANALYSIS PIPELINE

PURPOSE:
Batch processes multi-channel microscopy images to segment cell nuclei and
detect sub-nuclear signals (foci).

WORKFLOW:
1. Nuclei Segmentation (Blue Channel):
   - Applies Li-thresholding and morphological filtering.
   - Uses Watershed transformation to separate touching nuclei.
   - Generates a binary 'mask.png'.

2. Signal Preprocessing (Red Channel):
   - Performs local Z-score normalization (Background subtraction) using
     a sliding window to highlight peaks relative to local intensity.

3. Foci Detection (Red Channel):
   - Detects blobs using the Laplacian of Gaussian (LoG) method.
   - Filters candidates based on cell-specific mean intensity and mask boundaries.

4. Outputs (per experiment folder):
   - 'foci_analysis.csv': Granular data for every detected foci.
   - 'cell_summary.csv': Total foci count per cell.
   - 'original_red_overlay.png': Diagnostic image with bounding boxes and markers.
   - 'original_red_overlay_redchannel.png': Visualization with boosted foci intensity.

CONFIGURABILITY:
All detection thresholds, sigma values, and scaling factors are controlled
via an external 'config.yaml' file.

RUN SCRIPT SUCCESSFULLY:

1. PREREQUISITES:
   Ensure you have the required libraries installed:
   $ pip install pyyaml numpy pandas opencv-python pillow scikit-image scipy

2. CONFIGURATION:
   - Edit 'config.yaml' in the same folder as this script.
   - Set 'data_dir' to the absolute path of your experiment data.
   - Adjust 'foci_detection' parameters to tune sensitivity.

3. DATA STRUCTURE:
   The 'data_dir' must contain subfolders. Each subfolder needs:
   - One file ending in '*Blue.tif' (Nuclei)
   - One file ending in '*Red.tif'  (Foci)

4. EXECUTION (Terminal/Command Line):
   Navigate to the script's directory:
   $ cd C:/Users/thebl/PycharmProjects/Foci_Count_Project
   Run the script:
   $ python run_all_experiments.py

5. RESULTS:
   Check each experiment subfolder for a 'results' directory containing:
   - cell_summary.csv: Count per cell + config parameters used.
   - foci_analysis.csv: Detailed coordinates and intensity for every focus.
   - Visualizations (mask.png and overlays).
"""

from __future__ import annotations

import os
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import cv2
import yaml
from PIL import Image

from skimage import io, filters, color, measure, morphology, feature, segmentation
from skimage.segmentation import find_boundaries
from scipy import ndimage as nd
from skimage.draw import disk


# -------------------- Settings --------------------
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

SCRIPT_DIR = Path(__file__).parent.absolute()
cfg = load_config(SCRIPT_DIR / "config.yaml")

DATA_DIR = Path(cfg.get('paths', {}).get('data_dir', 'data'))

BLUE_GAUSS_SIGMA = cfg['segmentation']['blue_gauss_sigma']
WATERSHED_MIN_DISTANCE = cfg['segmentation']['watershed_min_distance']
MIN_SIZE_FACTOR = cfg['segmentation']['min_size_factor']
MIN_CELL_AREA = cfg['segmentation']['min_cell_area']

WINDOW_SIZE = cfg['preprocessing']['window_size']
EPSILON = float(cfg['preprocessing']['epsilon'])
RED_SMOOTH_SIGMA = cfg['preprocessing']['red_smooth_sigma']

BLOB_MIN_SIGMA = cfg['foci_detection']['blob_min_sigma']
BLOB_MAX_SIGMA = cfg['foci_detection']['blob_max_sigma']
BLOB_NUM_SIGMA = cfg['foci_detection']['blob_num_sigma']
BLOB_THRESHOLD = cfg['foci_detection']['blob_threshold']
REDCHANNEL_FOCI_BOOST = cfg['foci_detection']['redchannel_foci_boost']


# -------------------- Notebook helper --------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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
    img = io.imread(str(red_path))
    if img.ndim == 3:
        img = img[:, :, 0]
    # Wenn ihr 16-bit TIFF habt, wird das hier abgeschnitten. Falls das relevant ist, sag Bescheid.
    return img.astype(np.uint8)


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
        empty.to_csv(results_dir / "foci_analysis.csv", index=False)
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

    # ---- Red: preprocess + z-score ----
    img_red_orig_u8 = load_red_channel_uint8(red_path)
    img_red_norm, img_red_smoothed = preprocess_and_normalize(img_red_orig_u8, WINDOW_SIZE, EPSILON)

    # ---- Zell-Labeling (wie Notebook): cell_mask aus SMOOTHED * separated_mask ----
    masked_smooth = img_red_smoothed * separated_mask
    cell_mask = masked_smooth > 0
    labeled_cells = measure.label(cell_mask)
    regions = measure.regionprops(labeled_cells, intensity_image=img_red_norm)

    # ---- Overlay: GENAU wie vorher (dunkles Original) ----
    img_red_orig = np.array(Image.open(red_path).convert("L"))  # 1:1 Notebook style
    overlay = cv2.cvtColor(img_red_orig, cv2.COLOR_GRAY2BGR)

    # ---- Zusatzausgabe: gleiche Basis, aber nur FOCI im Rotkanal hervorheben ----
    overlay_redchannel = overlay.copy()

    # Maske, in die wir NUR die Foci zeichnen (ohne Boxen/IDs)
    foci_draw_mask = np.zeros(img_red_orig.shape, dtype=np.uint8)

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
            continue

        # Boxen/IDs: auf beide Overlays
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
    df.to_csv(results_dir / "foci_analysis.csv", index=False)
    # Summary over nucleai
    if not df.empty:
        # Group by cell_id and cound Foci
        summary_df = df.groupby("cell_id")["peak_id"].count().reset_index()
        summary_df.columns = ["cell_id", "foci_count"]

        summary_df.to_csv(results_dir / "cell_summary.csv", index=False)
    else:
        # If no Foci detected; create empty csv
        summary_df = pd.DataFrame(columns=["cell_id", "foci_count"])
        summary_df.to_csv(results_dir / "cell_summary.csv", index=False)

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

    for folder in folders:
        try:
            print(f"-> {folder.name}")
            df = analyze_experiment(folder)
            if not df.empty:
                all_rows.append(df)
        except Exception as e:
            failed.append((folder.name, repr(e)))
            print(f"!! Fehler in {folder.name}: {e}")
            traceback.print_exc()

    # Optional: Gesamt-CSV im Projektroot
    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        df_all.to_csv("all_foci_analysis.csv", index=False)
        print("Gesamt-CSV: all_foci_analysis.csv")

    if failed:
        pd.DataFrame(failed, columns=["experiment", "error"]).to_csv("failed_experiments.csv", index=False)
        print("Fehlerliste: failed_experiments.csv")


if __name__ == "__main__":
    main()
