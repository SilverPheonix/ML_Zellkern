import os
import shutil
from PIL import Image, ImageOps
import random

def augment_image(image_path, output_path, base_name):
    """
    Führt Bildaugmentierung durch: Rotation, Skalierung und Spiegeln.
    :param image_path: Pfad zur Originalbilddatei
    :param output_path: Pfad zum Ordner, in dem die augmentierten Bilder gespeichert werden
    :param base_name: Der Basisname für das Bild (ohne Erweiterung)
    """
    try:
        img = Image.open(image_path)

        # Augmentierungen durchführen und speichern
        augmentations = {
            'rotated_90': img.rotate(90, expand=True),
            'rotated_180': img.rotate(180, expand=True),
            'rotated_270': img.rotate(270, expand=True),
            'flipped_horizontally': ImageOps.mirror(img),
            'flipped_vertically': ImageOps.flip(img),
            'scaled_up': img.resize((int(img.width * 1.2), int(img.height * 1.2))),
            'scaled_down': img.resize((int(img.width * 0.8), int(img.height * 0.8))),
        }

        for aug_name, aug_img in augmentations.items():
            aug_file_name = f"{base_name}_{aug_name}.tif"  # Augmentiertes Bild im Zielordner mit .tif speichern
            aug_img.save(os.path.join(output_path, aug_file_name))
            print(f"Augmentiert und gespeichert: {aug_file_name}")
    except Exception as e:
        print(f"Fehler bei der Augmentierung von {image_path}: {str(e)}")

def process_tif_files_with_augmentation(source_folder, target_folder):
    """
    Durchsucht Ordner nach .tif-Dateien, kopiert und augmentiert sie.
    :param source_folder: Pfad zum Quellordner
    :param target_folder: Pfad zum Zielordner
    """
    # Überprüfen, ob das Quellverzeichnis existiert
    if not os.path.exists(source_folder):
        print(f"Fehler: Quellverzeichnis {source_folder} existiert nicht.")
        return
    
    if not os.path.exists(target_folder):
        print(f"Zielordner existiert nicht. Erstelle {target_folder}...")
        os.makedirs(target_folder)

    # Variable zum Überprüfen, ob .tif-Dateien gefunden wurden
    found_files = False

    # Durchlaufe alle Unterordner und Dateien im Quellordner
    for root, dirs, files in os.walk(source_folder):
        # Wenn keine Dateien im aktuellen Ordner vorhanden sind
        if not files:
            print(f"Warnung: Der Ordner {root} enthält keine Dateien oder .tif-Dateien.")
            continue  # Weiter zum nächsten Ordner
        
        # Überprüfe, ob es .tif-Dateien gibt
        tif_files_in_folder = [file for file in files if file.lower().endswith('.tif')]
        if not tif_files_in_folder:
            print(f"Warnung: Der Ordner {root} enthält keine .tif-Dateien.")
            continue  # Weiter zum nächsten Ordner

        # Extrahiere den aktuellen Unterordnernamen
        subfolder_name = os.path.basename(root)

        # Gehe alle .tif-Dateien im aktuellen Ordner durch
        for file in tif_files_in_folder:
            found_files = True
            # Originalbild kopieren
            source_file = os.path.join(root, file)
            
            # Zielpfad für das Originalbild (im Zielordner ohne Unterordnerstruktur)
            original_target_file = os.path.join(target_folder, f"{subfolder_name}-{file}")
            shutil.copy2(source_file, original_target_file)
            print(f"Kopiert: {source_file} -> {original_target_file}")

            # Augmentierungen durchführen und direkt im Zielordner speichern
            base_name = f"{subfolder_name}-{file}"
            augment_image(original_target_file, target_folder, base_name)

    # Falls keine .tif-Dateien gefunden wurden
    if not found_files:
        print(f"Keine .tif-Dateien im Quellordner {source_folder} oder seinen Unterordnern gefunden.")

# Beispielaufruf
source_folder = "yH2AX/"  # Quellordner (zu ersetzen mit dem tatsächlichen Pfad)
target_folder = "data/"   # Zielordner (zu ersetzen mit dem tatsächlichen Pfad)

process_tif_files_with_augmentation(source_folder, target_folder)
