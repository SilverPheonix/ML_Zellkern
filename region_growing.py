# Region Growing Pipeline – Elaine Fink
# Hinweis: Benötigte Bibliotheken: opencv-python-headless, scikit-image, matplotlib, numpy, scikit-learn

import os
import cv2
import numpy as np
from skimage import io, filters, measure, color
import matplotlib.pyplot as plt
from tools import ConfusionMatrix_ROCKurve as CMROC


# === Hilfsfunktionen ===

def show_image(image, title, cmap="gray"):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()

def show_results_grid(images, titles, cols=3, figsize=(15, 10), cmap="gray"):
    rows = int(np.ceil(len(images) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.ravel()

    for i in range(len(images)):
        axs[i].imshow(images[i], cmap=cmap if images[i].ndim == 2 else None)
        axs[i].set_title(titles[i])
        axs[i].axis("off")

    for i in range(len(images), len(axs)):
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()


# === Bildverarbeitung ===

def load_image(path):
    image = io.imread(path)
    return color.rgb2gray(image) if image.ndim == 3 else image

def preprocess_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def threshold_image(image):
    threshold = filters.threshold_otsu(image)
    return image > threshold

def region_growing(binary_image):
    _, labels = cv2.connectedComponents(binary_image.astype("uint8"))
    return labels

def analyze_regions(labels):
    regions = measure.regionprops(labels)
    areas = [r.area for r in regions]
    avg_area = np.mean(areas) if areas else 0
    min_area = 0.1 * avg_area
    filtered = np.zeros_like(labels)

    for region in regions:
        if region.area >= min_area:
            filtered[labels == region.label] = region.label
    return filtered

def create_bitmask(filtered_labels):
    return (filtered_labels > 0).astype(np.uint8)

def apply_bitmask(bitmask, image_path):
    rgb_image = io.imread(image_path)
    mask_3d = np.repeat(bitmask[:, :, np.newaxis], 3, axis=2)
    masked_image = np.ma.array(rgb_image, mask=mask_3d == 0)
    return masked_image.filled(0).astype(np.uint8)

def save_bitmask(bitmask, path):
    io.imsave(path, bitmask * 255)
    print(f"Bitmaske gespeichert: {path}")

def save_results(labels, path):
    io.imsave(path, labels.astype("uint16"))
    print(f"Segmentiertes Bild gespeichert: {path}")


# === Threshold-Vergleich ===

def compare_threshold_methods(image, base_output_path, color_image_path):
    methods = {
        "Otsu": filters.threshold_otsu,
        "Li": filters.threshold_li,
        "Triangle": filters.threshold_triangle,
    }

    results = []
    os.makedirs(base_output_path, exist_ok=True)

    for name, method in methods.items():
        thresholded = image > method(image)
        labels = region_growing(thresholded)
        filtered = analyze_regions(labels)
        bitmask = create_bitmask(filtered)
        masked_img = apply_bitmask(bitmask, color_image_path)

        save_bitmask(bitmask, os.path.join(base_output_path, f"{name}_bitmask.png"))
        io.imsave(os.path.join(base_output_path, f"{name}_masked.png"), masked_img)
        print(f"{name}: Verarbeitung abgeschlossen.")

        results.append((bitmask, masked_img, name))
    return results


# === Manuelle Maske laden ===

def load_manual_bitmask(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Maske nicht gefunden: {path}")
    mask = io.imread(path)
    if mask.ndim == 3:
        mask = color.rgb2gray(mask[:, :, :3])
    return (mask > 0.5).astype(np.uint8)


# === Hauptprogramm ===

def main():
    input_gray = "output/test_masking/2_Blue.tif"
    input_rgb = "output/test_masking/2_Red.tif"
    manual_mask_path = "output/test_masking/2_manual_bitmask.png"
    threshold_output_path = "output/threshold_results"
    final_output_path = "output/segmented_image.png"

    images, titles = [], []

    image = load_image(input_gray)
    images.append(image)
    titles.append("Original Graustufenbild")

    preprocessed = preprocess_image(image)
    results = compare_threshold_methods(preprocessed, threshold_output_path, input_rgb)

    for _, masked_img, name in results:
        images.append(masked_img)
        titles.append(f"{name} Maskiert")

    manual_mask = load_manual_bitmask(manual_mask_path)

    comparison_imgs, comparison_titles = [], []
    for bitmask, _, name in results:
        print(f"\n=== Evaluation: {name} ===")
        CMROC.evaluate_segmentation(bitmask, manual_mask)
        auto_props, manual_props = CMROC.analyze_cell_sizes(bitmask, manual_mask)
        CMROC.find_fp_fn_cells(auto_props, manual_props, bitmask, manual_mask)

        overlay = CMROC.create_comparison_overlay(
            predicted_mask=bitmask,
            manual_mask=manual_mask,
            output_path=f"{threshold_output_path}/{name}_comparison_overlay.png"
        )

        comparison_imgs.append(overlay)
        comparison_titles.append(f"{name} Overlay")

    show_results_grid(comparison_imgs, comparison_titles, cols=3, cmap=None)

    # Einzelne Schwellenwertmethode (z. B. Otsu) als Standard verwenden
    binary = threshold_image(preprocessed)
    labels = region_growing(binary)
    filtered = analyze_regions(labels)
    bitmask = create_bitmask(filtered)
    save_bitmask(bitmask, "output/test_masking/2_predicted_bitmask.png")

    final_img = apply_bitmask(bitmask, input_rgb)
    images.append(final_img)
    titles.append("Finale Maske")

    show_results_grid(images, titles, cols=2, cmap="gray")
    save_results(filtered, final_output_path)


if __name__ == "__main__":
    main()
