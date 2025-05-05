import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Funktion zum Laden der Ground-Truth-Maske (manuell erstellte Maske mit Zellen)
def load_ground_truth(path):
    gt = io.imread(path)  # Bild laden
    if len(gt.shape) == 3:  # Falls es ein RGB-Bild ist
        gt = color.rgb2gray(gt)  # In Graustufen umwandeln
    gt = gt > 0  # Alle Pixel > 0 gelten als Zellmaske (True)
    return gt.astype(np.uint8)  # In binäre Maske umwandeln (0 und 1)

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

# Funktion zur Erstellung einer ROC-Kurve
def plot_roc_curve(image, true_mask):
    # Ground Truth als 1D-Array
    flat_true = true_mask.flatten()

    # Listen für False-Positive-Rate (FPR) und True-Positive-Rate (TPR)
    fpr_list, tpr_list = [], []

    # Verschiedene Schwellenwerte (Thresholds) durchtesten von 0.1 bis 0.9
    thresholds = np.linspace(0.1, 0.9, 20)
    for thresh in thresholds:
        # Maske basierend auf Schwellenwert berechnen
        pred_mask = (image > thresh).astype(np.uint8)
        flat_pred = pred_mask.flatten()

        # ROC-Kurve berechnen: gibt mehrere Punkte zurück, aber wir nehmen den "positiven" Punkt bei Index 1
        fpr, tpr, _ = roc_curve(flat_true, flat_pred)
        fpr_list.append(fpr[1])
        tpr_list.append(tpr[1])

    # AUC (Area Under Curve) berechnen – Maß für die Qualität des Klassifikators
    roc_auc = auc(fpr_list, tpr_list)

    # ROC-Kurve zeichnen
    plt.figure()
    plt.plot(fpr_list, tpr_list, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonale als Referenzlinie (zufälliger Klassifikator)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
