# tools/ConfusionMatrix_ROCKurve.py

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from sklearn.metrics import confusion_matrix, roc_curve, auc
from skimage.measure import label, regionprops
from skimage.io import imsave
import os

def create_comparison_overlay(predicted_mask, manual_mask, output_path="output/comparison_overlay.png"):
    tp = np.logical_and(predicted_mask, manual_mask)
    fn = np.logical_and(np.logical_not(predicted_mask), manual_mask)
    fp = np.logical_and(predicted_mask, np.logical_not(manual_mask))
    tn = np.logical_and(np.logical_not(predicted_mask), np.logical_not(manual_mask))

    height, width = predicted_mask.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    color_image[tp] = [0, 255, 0]     # Grün
    color_image[fn] = [255, 0, 0]     # Rot
    color_image[fp] = [255, 255, 0]   # Gelb
    color_image[tn] = [0, 0, 0]       # Schwarz

    imsave(output_path, color_image)

    return color_image

def evaluate_segmentation(pred_mask, true_mask):
    # Masken in 1D-Vektoren umwandeln (flatten), damit sklearn sie verarbeiten kann
    pred = pred_mask.flatten()
    true = true_mask.flatten()

    # Konfusionsmatrix berechnen: tn = True Negative, fp = False Positive, fn = False Negative, tp = True Positive
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0 # Trefferquote (Recall)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 # Spezifität (Wie gut wird Hintergrund erkannt?)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0 # Genauigkeit der Positiv-Erkennung

    # Ergebnisse ausgeben
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Sensitivity (Recall): {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Precision: {precision:.2f}")

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision
    }

def analyze_cell_sizes(auto_mask, manual_mask):
    # Labels erzeugen
    labeled_auto = label(auto_mask)
    labeled_manual = label(manual_mask)

    auto_props = regionprops(labeled_auto)
    manual_props = regionprops(labeled_manual)

    print(f"Erkannte Zellen (automatisch): {len(auto_props)}")
    print(f"Zellen in der Ground Truth (manuell): {len(manual_props)}")

    # Zellgrößen extrahieren
    auto_areas = [r.area for r in auto_props]
    manual_areas = [r.area for r in manual_props]

    # Statistische Übersicht
    print("\n--- Zellgrößen ---")
    print(f"Automatisch: min={np.min(auto_areas)}, max={np.max(auto_areas)}, Ø={np.mean(auto_areas):.2f}")
    print(f"Manuell:     min={np.min(manual_areas)}, max={np.max(manual_areas)}, Ø={np.mean(manual_areas):.2f}")

    return auto_props, manual_props

def find_fp_fn_cells(auto_props, manual_props, auto_mask, manual_mask):
    false_positives = []
    false_negatives = []

    for region in auto_props:
        region_mask = (label(auto_mask) == region.label)
        overlap = region_mask & manual_mask
        if np.sum(overlap) < 0.1 * region.area:
            false_positives.append(region.area)

    for region in manual_props:
        region_mask = (label(manual_mask) == region.label)
        overlap = region_mask & auto_mask
        if np.sum(overlap) < 0.1 * region.area:
            false_negatives.append(region.area)

    print("\n--- Fehleranalyse ---")
    print(f"Falsch Positive (n={len(false_positives)}): Ø Größe = {np.mean(false_positives) if false_positives else 0:.2f}")
    print(f"Falsch Negative (n={len(false_negatives)}): Ø Größe = {np.mean(false_negatives) if false_negatives else 0:.2f}")
    return false_positives, false_negatives

def load_mask(path):
    mask = io.imread(path)
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = mask[:, :, :3]  # Entferne Alpha-Kanal (RGBA → RGB)
        mask = color.rgb2gray(mask)
    return (mask > 0.5).astype(np.uint8)

def analyze_cells_detailed(auto_props, manual_props, auto_mask, manual_mask):
    results = []

    for m_region in manual_props:
        m_mask = (label(manual_mask) == m_region.label)
        m_area = m_region.area
        overlap_mask = m_mask & auto_mask
        overlap_area = np.sum(overlap_mask)

        false_negative_area = np.sum(m_mask) - overlap_area
        false_positive_area = np.sum((label(auto_mask) > 0) & (~m_mask)) - (np.sum(auto_mask) - overlap_area)

        results.append({
            "ManualArea": m_area,
            "TP_pixels": overlap_area,
            "TP_%": overlap_area / m_area * 100 if m_area > 0 else 0,
            "FN_pixels": false_negative_area,
            "FN_%": false_negative_area / m_area * 100 if m_area > 0 else 0,
            "FP_pixels": max(false_positive_area, 0),  # neg. clipping vermeiden
            "FP_%": max(false_positive_area, 0) / m_area * 100 if m_area > 0 else 0,
        })

    return results


def main():
    auto_mask_path = "output/test_masking/2_predicted_bitmask.png"
    manual_mask_path = "output/test_masking/2_manual_bitmask.png"

    if not os.path.exists(auto_mask_path) or not os.path.exists(manual_mask_path):
        print("Fehlende Masken-Dateien.")
        return

    auto_mask = load_mask(auto_mask_path)
    manual_mask = load_mask(manual_mask_path)

    evaluate_segmentation(auto_mask, manual_mask)
    auto_props, manual_props = analyze_cell_sizes(auto_mask, manual_mask)
    find_fp_fn_cells(auto_props, manual_props, auto_mask, manual_mask)
    color_image = create_comparison_overlay(auto_mask, manual_mask)
    
    # Als Plot anzeigen
    plt.figure(figsize=(8, 8))
    plt.imshow(color_image)
    plt.title("Vergleich: Automatisch vs. Manuell")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
