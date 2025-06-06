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


# === Threshold-Vergleich ausgelagert ===

def run_threshold_comparison(image, manual_mask, rgb_path, output_dir):
    results = []
    comparison_data = []

    method_results = compare_threshold_methods(image, output_dir, rgb_path)

    for bitmask, _, name in method_results:
        print(f"\n=== Evaluation für Methode: {name} ===")
        overlay = CMROC.create_comparison_overlay(
            predicted_mask=bitmask,
            manual_mask=manual_mask,
            output_path=os.path.join(output_dir, f"{name}_overlay.png")
        )
        auto_props, manual_props = CMROC.analyze_cell_sizes(bitmask, manual_mask)

        # Analyse pro Zelle
        zellstats = CMROC.analyze_cells_detailed(auto_props, manual_props, bitmask, manual_mask)
        for z in zellstats:
            z['ThresholdMethod'] = name
            results.append(z)

        overlay = CMROC.create_comparison_overlay(
            predicted_mask=bitmask,
            manual_mask=manual_mask,
            output_path=os.path.join(output_dir, f"{name}_overlay.png")
        )

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

        results = run_threshold_comparison(preprocessed, manual_mask, rgb_path, output_dir)

        for res in results:
            res['ImageIndex'] = i
            summary_results.append(res)

    df = pd.DataFrame(summary_results)
    df.to_csv("output/summary_segmentation_evaluation.csv", index=False)
    print("\nAlle Ergebnisse gespeichert unter: output/summary_segmentation_evaluation.csv")


if __name__ == "__main__":
    main()
