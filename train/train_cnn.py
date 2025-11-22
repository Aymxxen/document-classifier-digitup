import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import sys
import yaml
import time
from tqdm import tqdm  # Barre de progression (pip install tqdm)

# --- Gestion des chemins d'import ---
# Permet d'importer le module 'models' qui se trouve au dossier parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn import DocumentClassifierCNN


def train_model():
    # --- 1. CONFIGURATION ---
    DATA_DIR = "data/processed"
    OUTPUT_DIR = "outputs"
    BATCH_SIZE = 32
    EPOCHS = 10  # Augmenter à 15-20 pour de meilleurs résultats
    LEARNING_RATE = 0.0001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🚀 Démarrage de l'entraînement sur : {DEVICE}")

    # --- 2. PRÉPARATION DES DONNÉES ---
    # Normalisation standard pour ResNet (ImageNet stats)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),  # Data augmentation légère
            transforms.RandomRotation(10),  # Data augmentation légère
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Chargement des datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f"📊 Classes détectées ({len(class_names)}) : {class_names}")
    print(f"🖼️ Images : {dataset_sizes['train']} train, {dataset_sizes['val']} val")

    # --- 3. SAUVEGARDE DES CLASSES ---
    # Indispensable pour que l'interface Streamlit connaisse les noms
    with open(os.path.join(OUTPUT_DIR, "classes.yaml"), "w") as f:
        yaml.dump(class_names, f)
    print(f"✅ Liste des classes sauvegardée dans {OUTPUT_DIR}/classes.yaml")

    # --- 4. INITIALISATION DU MODÈLE ---
    # freeze_backbone=False permet d'affiner tout le réseau (meilleur pour les docs)
    model = DocumentClassifierCNN(num_classes=len(class_names), freeze_backbone=False)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5. BOUCLE D'ENTRAÎNEMENT ---
    since = time.time()
    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        # Chaque époque a une phase d'entraînement et de validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Mode entraînement
            else:
                model.eval()  # Mode évaluation

            running_loss = 0.0
            running_corrects = 0

            # Itération sur les données (avec barre de progression)
            # Nécessite tqdm : pip install tqdm
            loop = tqdm(dataloaders[phase], leave=True, desc=f"{phase.capitalize()}")

            for inputs, labels in loop:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + Optimize uniquement en train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistiques
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Mise à jour barre de progression
                loop.set_postfix(loss=loss.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep Copy du modèle si c'est le meilleur
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"🏆 Nouveau record ! Modèle sauvegardé dans {best_model_path}")

    time_elapsed = time.time() - since
    print(f'\n🏁 Entraînement terminé en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')


if __name__ == "__main__":
    train_model()