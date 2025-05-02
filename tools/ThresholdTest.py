import cv2
import numpy as np
from skimage import io, filters, measure, morphology, color
import matplotlib.pyplot as plt
from skimage.filters import threshold_local

# Funktion zur Anzeige von Bildern
def show_image(image, title, cmap="gray"):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Bild laden und in Graustufen umwandeln
def load_image(path):
    image = io.imread(path)
    if len(image.shape) == 3:  # Falls RGB
        image = color.rgb2gray(image)
    show_image(image, "Original Image")
    return image

# Bild weichzeichnen zur Rauschreduzierung
def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    show_image(blurred, "Blurred Image")
    return blurred

# Vergleich verschiedener Threshold-Methoden
def compare_threshold_methods(image):
    methods = {
        "Otsu": filters.threshold_otsu,
        "Yen": filters.threshold_yen,
        "Li": filters.threshold_li,
        "Triangle": filters.threshold_triangle,
        "Adaptive (local)": lambda img: threshold_local(img, block_size=35)
    }

    plt.figure(figsize=(15, 6))
    binary_images = {}

    for i, (name, method) in enumerate(methods.items()):
        if name == "Adaptive (local)":
            thresholded = image > method(image)
        else:
            threshold_value = method(image)
            thresholded = image > threshold_value

        binary_images[name] = thresholded

        plt.subplot(2, 3, i+1)
        plt.imshow(thresholded, cmap="gray")
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Analyse der Flächen
    print("\nRegion-Flächen (in Pixel):")
    for name, binary in binary_images.items():
        labeled = measure.label(binary)
        props = measure.regionprops(labeled)
        total_area = sum([r.area for r in props])
        print(f"{name}: {total_area}")

# Hauptprogramm
def main():
    input_path = "output/test_masking/2_Blue.tif"  # Passe Pfad ggf. an
    image = load_image(input_path)
    preprocessed = preprocess_image(image)
    compare_threshold_methods(preprocessed)

main()