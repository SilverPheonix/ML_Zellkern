# Region Growing Elaine Fink

# Anleitung zur Installation für Teammitglieder
# Stelle sicher, dass Python installiert ist (empfohlen: Python 3.8 oder neuer).
# Installiere die benötigten Bibliotheken mit folgendem Befehl:
# pip install opencv-python-headless scikit-image matplotlib numpy

# Import der notwendigen Bibliotheken
import os
import cv2
import numpy as np
import numpy.ma as ma
from skimage import io, filters, measure, morphology, color
import matplotlib.pyplot as plt
from skimage.filters import threshold_local

# Funktion zur Anzeige von Bildern
def show_image(image, title, cmap="gray"):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()

# 1. Bild laden
def load_image(path):
    image = io.imread(path)
    if len(image.shape) == 3:  # Falls das Bild RGB ist, in Graustufen umwandeln
        image = color.rgb2gray(image)
    #show_image(image, "Original Image")
    return image

# 2. Vorverarbeitung
def preprocess_image(image):
    # Gaussian Blur zur Rauschreduzierung
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    #show_image(blurred, "Blurred Image")
    return blurred

# 3. Schwellenwertsetzung
def threshold_image(image):
    # Otsu-Schwellenwert bestimmen
    threshold = filters.threshold_otsu(image)
    binary_image = image > threshold
    #show_image(binary_image, "Thresholded Image")
    return binary_image

# 4. Region Growing (Flood Fill mit connectedComponents)
def region_growing(binary_image):
    # Connected Components finden
    num_labels, labels = cv2.connectedComponents(binary_image.astype("uint8"))
    #show_image(labels, "Connected Components", cmap="nipy_spectral")
    return labels

# 5. Analyse der Regionen
def analyze_regions(labels):
    # Regionen finden und analysieren
    regions = measure.regionprops(labels)
    print("Region Properties:")
    for region in regions:
        print(f"Region center: {region.centroid}, Area: {region.area}")
    
    # Berechne den Durchschnittswert der Flächen
    areas = [region.area for region in regions]
    avg_area = np.mean(areas) if areas else 0  # Falls keine Regionen existieren
    
    # Setze das Minimum als 10 % des Durchschnitts
    min_area = 0.1 * avg_area
    
    print(f"Durchschnittliche Regionengröße: {avg_area:.2f}, Minimum für Filterung: {min_area:.2f}")

    # Erstelle ein neues Label-Bild, das nur große Regionen enthält
    filtered_labels = np.zeros_like(labels)

    print("Filtered Region Properties:")
    for region in regions:
        if region.area >= min_area:
            filtered_labels[labels == region.label] = region.label
            print(f"Region center: {region.centroid}, Area: {region.area}")

    #show_image(filtered_labels, "Filtered Regions", cmap="nipy_spectral")
    return filtered_labels

# Erstellen der Maske
def create_bitmask(filtered_labels):# <class 'numpy.ndarray'>
    #Create boolean mask where region labels > 0; mask with True for region pixels, False for background
    boolean_mask = filtered_labels > 0
    #Convert boolean mask to integer mask 
    bitmask = boolean_mask.astype(np.uint8)
    ##show_image(bitmask, "Binary Bitmask", cmap="gray")
    return bitmask

# Maske anwenden
def apply_bitmask(bitmask):
    #bitmask auf 3Dimensionen erweitern damit auf Farbkanal gelegt werden kann
    mask_3d = np.repeat(bitmask[:, :, np.newaxis], 3, axis=2)

    rgb_image = io.imread("output/test_masking/2_Red.tif")
    # Bild maskieren
    masked_image = np.ma.array(rgb_image, mask=mask_3d==0)
    # Unsichtbare Bereiche auf Schwarz setzen
    final_image = masked_image.filled(0).astype(np.uint8)
    ##show_image(final_image, "Final Masked Image")
    return final_image

# 6. Ergebnisse speichern
# Funktioniert noch nicht ganz
def save_results(labels, output_path):
    io.imsave(output_path, labels.astype("uint16"))
    print(f"Segmented image saved to {output_path}")

    # Geladenes Label-Bild anzeigen
    labels = io.imread(output_path)
    #plt.imshow(labels, cmap="nipy_spectral")

    #plt.title("Loaded Label Image")
    #plt.colorbar()
    #plt.show()

    # Pixelwerte im Bild prüfen
    print(f"Min pixel value: {labels.min()}, Max pixel value: {labels.max()}")

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

    
def show_results_grid(images, titles, cols=3, figsize=(15, 10), cmap="gray"):
    rows = int(np.ceil(len(images) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.ravel()

    for i in range(len(images)):
        img = images[i]
        if img.ndim == 2:  # Graustufen oder Labels
            axs[i].imshow(img, cmap=cmap)
        else:  # RGB oder RGBA
            axs[i].imshow(img)
        axs[i].set_title(titles[i])
        axs[i].axis("off")

    # Leere Felder verstecken
    for i in range(len(images), len(axs)):
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()
    
def load_manual_bitmask(manual_mask_path):
    if not os.path.exists(manual_mask_path):
        raise FileNotFoundError(f"Manual mask not found: {manual_mask_path}")

    manual_mask = io.imread(manual_mask_path)

    # Wenn Bild 4 Kanäle hat (RGBA), nur Helligkeit verwenden
    if manual_mask.ndim == 3:
        if manual_mask.shape[2] == 4:
            # Alpha-Kanal ignorieren oder Helligkeit aus RGB berechnen
            manual_mask = color.rgb2gray(manual_mask[:, :, :3])
        else:
            manual_mask = color.rgb2gray(manual_mask)

    binary_mask = manual_mask > 0.5
    bitmask = binary_mask.astype(np.uint8)
    #show_image(bitmask, "Manual Bitmask", cmap="gray")
    return bitmask

# Hauptprogramm
def main():
    # Pfad zum Eingabebild
    # good examples: data/Ex 3 day 02-1_image_BGR- Blue.tif, data/Ex 3 day 09-1_image_BGR- Blue.tif
    input_path = "output/test_masking/2_Blue.tif"
    output_path = "output/segmented_image.tif"
    manual_mask_path = "output/test_masking/2_manual_bitmask.png"

    # Speicherliste für Anzeige
    images = []
    titles = []

    # Workflow
    image = load_image(input_path)
    images.append(image)
    titles.append("Original Image in Grayscale")

    preprocessed_image = preprocess_image(image)

    compare_threshold_methods(preprocessed_image)

    binary_image = threshold_image(preprocessed_image)
    labels = region_growing(binary_image)
    filtered_labels = analyze_regions(labels)
    bitmask = create_bitmask (filtered_labels)
    manual_bitmask = load_manual_bitmask(manual_mask_path)

    masked_image = apply_bitmask(bitmask)
    images.append(masked_image)
    titles.append("Masked Image")

    show_results_grid(images, titles, cols=3, cmap="gray")

    save_results(filtered_labels, output_path)

main()
