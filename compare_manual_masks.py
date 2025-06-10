import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from tools import ConfusionMatrix_ROCKurve as CMROC

def main():
    # Alle manuellen Masken zum selben Bild
    manual_masks = {
        "Elaine": "data/test_masking/2_manual_bitmask_elaine.png",
        "Isabella": "data/test_masking/2_manual_bitmask_Isabella.png",
        "Lisa": "data/test_masking/2_manual_bitmask_mayrhofer.png",
        "Marcin": "data/test_masking/2_manual_bitmask_zboroch.png",
    }

    masks = {name: CMROC.load_mask(path) for name, path in manual_masks.items()}

    for ref_name, ref_mask in masks.items():
        print(f"\n=== Vergleiche mit Referenz: {ref_name} ===")
        for pred_name, pred_mask in masks.items():
            if ref_name == pred_name:
                continue

            print(f"\n→ {pred_name} als Vorhersage gegen {ref_name} als GT")
            overlay = CMROC.create_comparison_overlay(pred_mask, ref_mask)
            overlay_path = os.path.join('output/compare_manual_masks/', f"{pred_name}_vs_{ref_name}_overlay.png")
            io.imsave(overlay_path, overlay.astype(np.uint8))
            CMROC.evaluate_segmentation(pred_mask, ref_mask)
            auto_props, manual_props = CMROC.analyze_cell_sizes(pred_mask, ref_mask)
            CMROC.find_fp_fn_cells(auto_props, manual_props, pred_mask, ref_mask)

            plt.figure(figsize=(6, 6))
            plt.imshow(overlay)
            plt.title(f"{pred_name} vs. {ref_name}")
            plt.axis("off")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
