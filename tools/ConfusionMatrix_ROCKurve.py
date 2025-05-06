
# Installiere die benötigten Bibliotheken mit folgendem Befehl:
# pip install scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from sklearn.metrics import confusion_matrix, roc_curve, auc
from skimage.measure import label, regionprops

# Auswertung der Segmentierung anhand einer Konfusionsmatrix
def evaluate_segmentation(pred_mask, true_mask):
    # Masken in 1D-Vektoren umwandeln (flatten), damit sklearn sie verarbeiten kann
    pred = pred_mask.flatten()
    true = true_mask.flatten()

    # Konfusionsmatrix berechnen: tn = True Negative, fp = False Positive, fn = False Negative, tp = True Positive
    tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()

    # Metriken berechnen:
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Trefferquote (Recall)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Spezifität (Wie gut wird Hintergrund erkannt?)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0    # Genauigkeit der Positiv-Erkennung

    # Ergebnisse ausgeben
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Sensitivity (Recall): {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    print(f"Precision: {precision:.2f}")

def plot_roc_curve(probability_image, true_mask):
    # Flatten ground truth and image to 1D
    y_true = true_mask.flatten()
    y_scores = probability_image.flatten()

    # Berechne FPR, TPR, Thresholds für alle möglichen Schwellenwerte
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # AUC berechnen
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--', label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()



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
        if np.sum(overlap) < 0.1 * region.area:  # Weniger als 10% Überlappung = FP
            false_positives.append(region.area)

    for region in manual_props:
        region_mask = (label(manual_mask) == region.label)
        overlap = region_mask & auto_mask
        if np.sum(overlap) < 0.1 * region.area:
            false_negatives.append(region.area)

    print("\n--- Fehleranalyse ---")
    print(f"Falsch Positive (n={len(false_positives)}): Ø Größe = {np.mean(false_positives) if false_positives else 0:.2f}")
    print(f"Falsch Negative (n={len(false_negatives)}): Ø Größe = {np.mean(false_negatives) if false_negatives else 0:.2f}")

