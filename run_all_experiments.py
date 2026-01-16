from __future__ import annotations

from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import cv2
from PIL import Image

from skimage import io, filters, color, measure, morphology, feature, segmentation
from skimage.segmentation import find_boundaries
from scipy import ndimage as nd
from skimage.draw import disk


# -------------------- Pfade --------------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


# -------------------- Notebook-Funktionen (gleiche Logik) --------------------

def analyze_cell_sizes(li_mask):
    labeled_li = measure.label(li_mask)
    li_props = measure.regionprops(labeled_li)
    li_areas = [r.area for r in li_props]
    if not li_areas:
        return [], 0
    return li_props, float(np.mean(li_areas))


def get_separated_binary_mask(binary_mask, min_distance=20):
    """
    Splits clumped cells from an EXISTING binary mask and returns:
    final_binary_mask, labeled_cells, areas
    """
    binary_mask = binary_mask.astype(bool)
    distance = nd.distance_transform_edt(binary_mask)

    coords = feature.peak_local_max(distance, min_distance=min_distance, labels=binary_mask)
    if len(coords) == 0:
        return binary_mask, np.zeros_like(binary_mask, dtype=int), []

    mask_seeds = np.zeros(distance.shape, dtype=bool)
    mask_seeds[tuple(coords.T)] = True
    markers, _ = nd.label(mask_seeds)

    labeled_cells = segmentation.watershed(-distance, markers, mask=binary_mask)

    props = measure.regionprops(labeled_cells)
    areas = [r.area for r in props]

    boundaries = find_boundaries(labeled_cells, mode="thick")
    final_binary_mask = binary_mask.copy()
    final_binary_mask[boundaries] = 0

    return final_binary_mask, labeled_cells, areas


def load_red_channel(path: Path) -> np.ndarray:
    image = io.imread(str(path))
    if image.ndim == 3:
        return (image[:, :, 0]).astype(np.uint8)
    elif image.ndim == 2:
        return image.astype(np.uint8)
    else:
        raise ValueError("Image must be 2D (grayscale) or 3D (color).")


def preprocess_and_normalize(img_red_raw: np.ndarray, window_size: int, epsilon: float):
    img_red_smoothed = filters.gaussian(img_red_raw, sigma=1)
    local_mean = nd.uniform_filter(img_red_smoothed, size=window_size)
    local_std = np.sqrt(nd.uniform_filter(img_red_smoothed ** 2, size=window_size) - local_mean ** 2)
    img_red_norm = (img_red_smoothed - local_mean) / (local_std + epsilon)
    return img_red_norm.astype(np.float32), img_red_smoothed.astype(np.float32)


# -------------------- Helpers --------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_to_u8(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)


# -------------------- Pipeline pro Experiment --------------------

def analyze_experiment(folder: Path) -> pd.DataFrame:
    exp_name = folder.name
    exp_out = OUTPUT_DIR / exp_name
    ensure_dir(exp_out)

    blue_path = next(folder.glob("*Blue.tif"))
    red_path = next(folder.glob("*Red.tif"))

    # ---- Blue: grayscale + gaussian + Li threshold ----
    original_blue = io.imread(str(blue_path))
    if original_blue.ndim == 3:
        grayscale_blue = color.rgb2gray(original_blue)
    else:
        grayscale_blue = original_blue.astype(np.float32)

    blurred_blue = filters.gaussian(grayscale_blue, sigma=1)

    li_threshold_value = filters.threshold_li(blurred_blue)
    li_mask_initial = blurred_blue > li_threshold_value

    _, average_area = analyze_cell_sizes(li_mask_initial)
    if average_area <= 0:
        # nichts erkennbar
        pd.DataFrame().to_csv(exp_out / "foci_analysis.csv", index=False)
        return pd.DataFrame()

    min_size_threshold = int(average_area) * 0.65
    li_mask_filtered = morphology.remove_small_objects(li_mask_initial.astype(bool), min_size=int(min_size_threshold))

    # ---- Watershed separation (wie Notebook) ----
    separated_mask, _, _ = get_separated_binary_mask(li_mask_filtered, min_distance=20)

    # Maske speichern
    cv2.imwrite(str(exp_out / "mask.png"), (separated_mask.astype(np.uint8) * 255))

    # ---- Red: preprocess + z-score ----
    WINDOW_SIZE = 21
    EPSILON = 1e-6
    img_red_raw = load_red_channel(red_path)
    img_red_norm, img_red_smoothed = preprocess_and_normalize(img_red_raw, WINDOW_SIZE, EPSILON)

    # ---- Zell-Labeling WIE im Notebook: cell_mask aus *smoothed* (nicht zscore!) ----
    masked_smooth = img_red_smoothed * separated_mask  # <- entscheidend
    cell_mask = masked_smooth > 0
    labeled_cells = measure.label(cell_mask)

    # regionprops: intensity_image = z-score (wie Notebook cell 20)
    regions = measure.regionprops(labeled_cells, intensity_image=img_red_norm)

    # ---- Overlays vorbereiten (wie Notebook) ----
    smoothed_intensity = img_red_norm.astype(np.float32)

    disp_z = normalize_to_u8(smoothed_intensity)
    out_z = cv2.cvtColor(disp_z, cv2.COLOR_GRAY2BGR)

    img_red_orig = np.array(Image.open(red_path).convert("L"))
    disp_orig = normalize_to_u8(img_red_orig)
    out_orig = cv2.cvtColor(disp_orig, cv2.COLOR_GRAY2BGR)

    # ---- Foci Detection + Annotation (Notebook cell 20) ----
    foci_records = []
    peak_id_counter = 1

    for region in regions:
        if region.area < 50:
            continue

        cell_avg_intensity = float(region.mean_intensity)
        minr, minc, maxr, maxc = region.bbox
        cropped_intensity = smoothed_intensity[minr:maxr, minc:maxc]

        blobs_log = feature.blob_log(
            cropped_intensity,
            min_sigma=2,
            max_sigma=5,
            num_sigma=10,
            threshold=1,  # wie Notebook cell 20
        )

        if len(blobs_log) == 0:
            continue

        # Zellbox+ID (nur wenn es Foci gibt, wie Notebook)
        for img in (out_z, out_orig):
            cv2.rectangle(img, (minc, minr), (maxc, maxr), (0, 255, 0), 1)
            cv2.putText(img, f"ID:{region.label}", (minc, max(0, minr - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        for local_r, local_c, sigma in blobs_log:
            local_r, local_c = int(local_r), int(local_c)
            global_r, global_c = minr + local_r, minc + local_c

            if labeled_cells[global_r, global_c] != region.label:
                continue

            intensity_at_peak = float(smoothed_intensity[global_r, global_c])
            if intensity_at_peak <= cell_avg_intensity:
                continue

            radius = float(sigma * np.sqrt(2))

            for img in (out_z, out_orig):
                cv2.circle(img, (global_c, global_r), int(max(1, radius)), (0, 255, 255), 1)
                cv2.circle(img, (global_c, global_r), 1, (0, 0, 255), -1)

            rr, cc = disk((global_r, global_c), max(1, int(radius)), shape=smoothed_intensity.shape)
            focus_mean_intensity = float(np.mean(smoothed_intensity[rr, cc]))

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
            })
            peak_id_counter += 1

    df = pd.DataFrame(foci_records)

    # ---- Speichern (wie Notebook, nur in output/<exp>/) ----
    df.to_csv(exp_out / "foci_analysis.csv", index=False)

    cv2.imwrite(str(exp_out / "zscore_overlay.png"), out_z)
    cv2.imwrite(str(exp_out / "original_red_overlay.png"), out_orig)

    # Vergleichsbild nebeneinander
    h = min(out_z.shape[0], out_orig.shape[0])
    combined = np.concatenate([out_z[:h], out_orig[:h]], axis=1)
    cv2.imwrite(str(exp_out / "foci_comparison.png"), combined)

    total_foci = len(df)

    with open(exp_out / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Total Cells: {len(regions)}\n")
        f.write(f"Total Foci: {total_foci}\n\n")
        f.write("Head (first 20 rows):\n")
        f.write(df.head(20).to_string(index=False))

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
    ensure_dir(OUTPUT_DIR)

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

    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
        df_all.to_csv(OUTPUT_DIR / "all_foci_analysis.csv", index=False)
        print(f"Gesamt-CSV: {OUTPUT_DIR / 'all_foci_analysis.csv'}")

    if failed:
        pd.DataFrame(failed, columns=["experiment", "error"]).to_csv(OUTPUT_DIR / "failed_experiments.csv", index=False)
        print(f"Fehlerliste: {OUTPUT_DIR / 'failed_experiments.csv'}")


if __name__ == "__main__":
    main()
