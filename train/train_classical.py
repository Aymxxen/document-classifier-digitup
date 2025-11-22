import os
import cv2
import numpy as np
import joblib
import sys
import yaml
from tqdm import tqdm
from sklearn.metrics import classification_report

# Ajout du path pour importer models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classical import get_classical_pipeline


def load_images_from_folder(folder, img_size=(64, 64)):
    images = []
    labels = []
    classes = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])

    print(f"Chargement des données depuis {folder}...")
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(folder, class_name)
        files = os.listdir(class_path)
        for f in tqdm(files, desc=class_name):
            try:
                img_path = os.path.join(class_path, f)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Niveaux de gris pour le classique
                    images.append(img)
                    labels.append(idx)
            except Exception as e:
                pass
    return np.array(images), np.array(labels), classes


def train_classical():
    # Config
    DATA_DIR = "data/processed"
    MODEL_PATH = "outputs/classical_model.pkl"
    CLASSES_PATH = "outputs/classes.yaml"

    # 1. Chargement
    X_train, y_train, classes = load_images_from_folder(os.path.join(DATA_DIR, 'train'))
    X_val, y_val, _ = load_images_from_folder(os.path.join(DATA_DIR, 'val'))

    print(f"Données chargées: {len(X_train)} train, {len(X_val)} val")

    # 2. Pipeline & Entraînement
    print("Entraînement du SVM (cela peut prendre un moment)...")
    pipeline = get_classical_pipeline()
    pipeline.fit(X_train, y_train)

    # 3. Evaluation rapide
    print("Évaluation...")
    acc = pipeline.score(X_val, y_val)
    print(f"Précision globale (Accuracy): {acc:.2%}")

    # 4. Sauvegarde
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    # Sauvegarde des classes si le CNN ne l'a pas déjà fait
    if not os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "w") as f:
            yaml.dump(classes, f)

    print(f"Modèle classique sauvegardé dans {MODEL_PATH}")


if __name__ == "__main__":
    train_classical()