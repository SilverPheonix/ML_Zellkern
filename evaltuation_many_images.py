import os
from skimage import io
from tools import ConfusionMatrix_ROCKurve as CMROC
from region_growing import (
    load_image, preprocess_image, compare_threshold_methods,
    load_manual_bitmask, threshold_image, region_growing,
    analyze_regions, create_bitmask, save_bitmask,
    apply_bitmask, save_results
)
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np


def write_csv_results(csv_path, results):
    fieldnames = [
        "Bild", "Methode", "TP", "FP", "FN", "TN",
        "Sensitivity", "Specificity", "Precision",
        "Ø FP-Fläche", "Ø FN-Fläche",
        "Zellen (GT)", "Zellen (Auto)"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


# === Threshold-Vergleich ===

def run_threshold_comparison(image, manual_mask, rgb_path, output_dir, global_results):
    results = []

    method_results = compare_threshold_methods(image, output_dir, rgb_path)

    for bitmask, _, name in method_results:
        print(f"\n=== Evaluation für Methode: {name} ===")
        overlay = CMROC.create_comparison_overlay(
            predicted_mask=bitmask,
            manual_mask=manual_mask,
            output_path=os.path.join(output_dir, f"{name}_overlay.png")
        )
        auto_props, manual_props = CMROC.analyze_cell_sizes(bitmask, manual_mask)

        # Zell-Analyse
        zellstats = CMROC.analyze_cells_detailed(auto_props, manual_props, bitmask, manual_mask)
        for z in zellstats:
            z['ThresholdMethod'] = name
            results.append(z)

        # Metriken berechnen
        eval_result = CMROC.evaluate_segmentation(bitmask, manual_mask)
        fp_sizes, fn_sizes = CMROC.find_fp_fn_cells(
            auto_props, manual_props, bitmask, manual_mask
        )

        global_results.append({
            "Bild": os.path.basename(rgb_path),
            "Methode": name,
            "TP": eval_result["TP"],
            "FP": eval_result["FP"],
            "FN": eval_result["FN"],
            "TN": eval_result["TN"],
            "Sensitivity": round(eval_result["Sensitivity"], 3),
            "Specificity": round(eval_result["Specificity"], 3),
            "Precision": round(eval_result["Precision"], 3),
            "Ø FP-Fläche": round(np.mean(fp_sizes) if fp_sizes else 0, 1),
            "Ø FN-Fläche": round(np.mean(fn_sizes) if fn_sizes else 0, 1),
            "Zellen (GT)": len(manual_props),
            "Zellen (Auto)": len(auto_props),
        })

        # Overlay anzeigen
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.title(f"Overlay {name} – Bild {os.path.basename(rgb_path)}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return results


# === Hauptausführung für alle Bilder ===

def main():
    image_indices = range(1, 7)
    summary_results = []
    global_eval_results = []  # CSV für TP, FP, FN, etc.

    for i in image_indices:
        gray_path = f"output/test_masking/{i}_Blue.tif"
        rgb_path = f"output/test_masking/{i}_Blue.tif"
        manual_mask_path = f"output/test_masking/{i}_manual_bitmask_elaine.png"
        output_dir = f"output/test_masking/{i}_results"

        print(f"\n\n===== Verarbeitung Bild {i} =====")
        os.makedirs(output_dir, exist_ok=True)

        gray_img = load_image(gray_path)
        preprocessed = preprocess_image(gray_img)
        manual_mask = load_manual_bitmask(manual_mask_path)

        results = run_threshold_comparison(preprocessed, manual_mask, rgb_path, output_dir, global_eval_results)

        for res in results:
            res['ImageIndex'] = i
            summary_results.append(res)

    # Zellbasierte Statistik
    df = pd.DataFrame(summary_results)
    df.to_csv("output/summary_segmentation_evaluation.csv", index=False)
    print("\nZellstatistik gespeichert unter: output/summary_segmentation_evaluation.csv")

    # Bildweise Evaluation (TP, FP, etc.)
    write_csv_results("output/summary_global_metrics.csv", global_eval_results)
    print("Globale Metriken gespeichert unter: output/summary_global_metrics.csv")


if __name__ == "__main__":
    main()
