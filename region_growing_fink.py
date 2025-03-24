# Region Growing Elaine Fink

# Anleitung zur Installation für Teammitglieder
# Stelle sicher, dass Python installiert ist (empfohlen: Python 3.8 oder neuer).
# Installiere die benötigten Bibliotheken mit folgendem Befehl:
# pip install opencv-python-headless scikit-image matplotlib numpy

# Import der notwendigen Bibliotheken
import cv2
import numpy as np
from skimage import io, filters, measure, morphology, color
import matplotlib.pyplot as plt

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
    show_image(image, "Original Image")
    return image

# 2. Vorverarbeitung
def preprocess_image(image):
    # Gaussian Blur zur Rauschreduzierung
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    show_image(blurred, "Blurred Image")
    return blurred

# 3. Schwellenwertsetzung
def threshold_image(image):
    # Otsu-Schwellenwert bestimmen
    threshold = filters.threshold_otsu(image)
    binary_image = image > threshold
    show_image(binary_image, "Thresholded Image")
    return binary_image

# 4. Region Growing (Flood Fill mit connectedComponents)
def region_growing(binary_image):
    # Connected Components finden
    num_labels, labels = cv2.connectedComponents(binary_image.astype("uint8"))
    show_image(labels, "Connected Components", cmap="nipy_spectral")
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
    
    show_image(filtered_labels, "Filtered Regions", cmap="nipy_spectral")
    return filtered_labels


# 6. Ergebnisse speichern
# Funktioniert noch nicht ganz
def save_results(labels, output_path):
    io.imsave(output_path, labels.astype("uint16"))
    print(f"Segmented image saved to {output_path}")

    # Geladenes Label-Bild anzeigen
    labels = io.imread(output_path)
    plt.imshow(labels, cmap="nipy_spectral")

    plt.title("Loaded Label Image")
    plt.colorbar()
    plt.show()

    # Pixelwerte im Bild prüfen
    print(f"Min pixel value: {labels.min()}, Max pixel value: {labels.max()}")

# Hauptprogramm
def main():
    # Pfad zum Eingabebild
    # good examples: data/Ex 3 day 02-1_image_BGR- Blue.tif, data/Ex 3 day 09-1_image_BGR- Blue.tif
    input_path = "data/Ex 3 day 09-1_image_BGR- Blue.tif"
    output_path = "output/segmented_image.tif"

    # Workflow
    image = load_image(input_path)
    preprocessed_image = preprocess_image(image)
    binary_image = threshold_image(preprocessed_image)
    labels = region_growing(binary_image)
    filtered_labels = analyze_regions(labels)
    save_results(filtered_labels, output_path)



main()
