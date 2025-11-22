import torch
import os
import sys
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from torchvision import datasets, transforms

# Imports locaux
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn import DocumentClassifierCNN


def evaluate_model_complete():
    # Config
    DATA_DIR = "data/processed/val"
    MODEL_PATH = "outputs/best_model.pth"
    CLASSES_PATH = "outputs/classes.yaml"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Chargement
    if not os.path.exists(CLASSES_PATH): return
    with open(CLASSES_PATH, 'r') as f:
        classes = yaml.safe_load(f)
    n_classes = len(classes)

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = DocumentClassifierCNN(num_classes=n_classes, freeze_backbone=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Inférence (Récupérer les probabilités pour ROC)
    all_preds, all_labels, all_probs = [], [], []

    print("Calcul des métriques et courbes ROC...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)  # Important pour ROC
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 3. Rapport Texte & Matrice
    print(classification_report(all_labels, all_preds, target_names=classes))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Matrice de Confusion")
    plt.savefig("outputs/confusion_matrix.png")

    # 4. --- NEW: Courbes ROC & AUC (Exigence Client) ---
    all_probs = np.array(all_probs)
    y_test_bin = label_binarize(all_labels, classes=range(n_classes))

    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'{classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbes ROC multi-classes')
    plt.legend(loc="lower right")
    plt.savefig("outputs/roc_curves.png")
    print("✅ Courbes ROC sauvegardées dans outputs/roc_curves.png")


if __name__ == "__main__":
    evaluate_model_complete()