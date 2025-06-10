# main_analysis.py
import os
from skimage import io
from tools import ConfusionMatrix_ROCKurve as CMROC
from region_growing import *
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

def run_threshold_comparison(image, manual_mask, rgb_path, output_dir, global_results):
    results = []
    method_results = compare_threshold_methods(image, output_dir, rgb_path)

    for bitmask, _, name in method_results:
        overlay = CMROC.create_comparison_overlay(
            predicted_mask=bitmask,
            manual_mask=manual_mask,
            output_path=os.path.join(output_dir, f"{name}_overlay.png")
        )

        auto_props, manual_props = CMROC.analyze_cell_sizes(bitmask, manual_mask)
        zellstats = CMROC.analyze_cells_detailed(auto_props, manual_props, bitmask, manual_mask)
        for z in zellstats:
            z['ThresholdMethod'] = name
            results.append(z)

        eval_result = CMROC.evaluate_segmentation(bitmask, manual_mask)
        fp_sizes, fn_sizes = CMROC.find_fp_fn_cells(auto_props, manual_props, bitmask, manual_mask)

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

    return results

def main():
    image_names = ["1", "2", "3", "4", "5"]
    zellstats_all = []
    global_results = []

    for name in image_names:
        image_path = f"data/test_masking/{name}_Blue.tif"
        manual_path = f"data/test_masking/{name}_manual_bitmask_zboroch.png"
        output_dir = f"output/test_masking/{name}_results/"

        gray_img = load_image(image_path)
        manual_mask = load_manual_bitmask(manual_path)
        preprocessed = preprocess_image(gray_img)

        zellstats = run_threshold_comparison(preprocessed, manual_mask, image_path, output_dir, global_results)
        for entry in zellstats:
            entry["Bild"] = name
        zellstats_all.extend(zellstats)

    write_csv_results("output/global_summary.csv", global_results)
    pd.DataFrame(zellstats_all).to_csv("output/zellstats.csv", index=False)
    print("Alle Auswertungen abgeschlossen.")

if __name__ == "__main__":
    main()
