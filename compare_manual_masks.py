# manual_mask_comparison.py

import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from tools import ConfusionMatrix_ROCKurve as CMROC
import pandas as pd
import csv

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_comparison_csv(csv_path, results):
    fieldnames = [
        "Bild", "Referenz", "Vorhersage", "TP", "FP", "FN", "TN",
        "Sensitivity", "Specificity", "Precision",
        "Ø FP-Fläche", "Ø FN-Fläche",
        "Zellen (GT)", "Zellen (Auto)"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

def main():
    image_names = ["1", "2", "3", "4", "5", "6"]
    output_dir = 'output/compare_manual_masks/'
    ensure_dir(output_dir)

    all_results = []

    for name in image_names:
        # Alle manuellen Masken zum selben Bild
        manual_masks = {
            "Elaine": f"data/test_masking/{name}_manual_bitmask_elaine.png",
            "Isabella": f"data/test_masking/{name}_manual_bitmask_Isabella.png",
            "Lisa": f"data/test_masking/{name}_manual_bitmask_mayrhofer.png",
            "Marcin": f"data/test_masking/{name}_manual_bitmask_zboroch.png",
        }

        masks = {label: CMROC.load_mask(path) for label, path in manual_masks.items()}

        for ref_name, ref_mask in masks.items():
            for pred_name, pred_mask in masks.items():
                if ref_name == pred_name:
                    continue

                print(f"\n→ {pred_name} als Vorhersage gegen {ref_name} als GT")

                # Overlay erzeugen und speichern
                overlay = CMROC.create_comparison_overlay(pred_mask, ref_mask)
                overlay_path = os.path.join(output_dir, f"{name}_{pred_name}_vs_{ref_name}_overlay.png")
                io.imsave(overlay_path, overlay.astype(np.uint8))

                # Auswertung
                eval_result = CMROC.evaluate_segmentation(pred_mask, ref_mask)
                auto_props, manual_props = CMROC.analyze_cell_sizes(pred_mask, ref_mask)
                fp_sizes, fn_sizes = CMROC.find_fp_fn_cells(auto_props, manual_props, pred_mask, ref_mask)

                # Ergebnisse sammeln
                all_results.append({
                    "Bild": name,
                    "Referenz": ref_name,
                    "Vorhersage": pred_name,
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

                # Optional anzeigen
                #plt.figure(figsize=(6, 6))
                #plt.imshow(overlay)
                #plt.title(f"{name}: {pred_name} vs. {ref_name}")
                #plt.axis("off")
                #plt.tight_layout()
                #plt.show()

    # CSV schreiben
    csv_output_path = os.path.join(output_dir, "mask_comparison_summary.csv")
    write_comparison_csv(csv_output_path, all_results)
    print(f"Vergleich abgeschlossen. Ergebnisse gespeichert in {csv_output_path}")

if __name__ == "__main__":
    main()
