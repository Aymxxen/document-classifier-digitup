import torch
import os
import sys
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms
from tqdm import tqdm

# Imports locaux
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn import DocumentClassifierCNN


def evaluate_model():
    # Config
    DATA_DIR = "data/processed/val"
    MODEL_PATH = "outputs/best_model.pth"
    CLASSES_PATH = "outputs/classes.yaml"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        print("Erreur: Modèle non trouvé. Lancez l'entraînement d'abord.")
        return

    # 1. Charger les classes
    with open(CLASSES_PATH, 'r') as f:
        class_names = yaml.safe_load(f)

    # 2. Préparation des données
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 3. Charger le modèle
    model = DocumentClassifierCNN(num_classes=len(class_names), freeze_backbone=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 4. Inférence
    all_preds = []
    all_labels = []

    print("Calcul des prédictions...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 5. Métriques & Rapport
    print("\n--- RAPPORT DE CLASSIFICATION ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 6. Matrice de Confusion
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    plt.title('Matrice de Confusion (Modèle CNN)')

    output_plot = "outputs/confusion_matrix.png"
    plt.savefig(output_plot)
    print(f"Matrice de confusion sauvegardée : {output_plot}")
    plt.show()


if __name__ == "__main__":
    # Nécessite seaborn : pip install seaborn
    evaluate_model()