import torch
import torch.nn as nn
from torchvision import models


class DocumentClassifierCNN(nn.Module):
    def __init__(self, num_classes=16, freeze_backbone=True):
        super(DocumentClassifierCNN, self).__init__()
        # Chargement de ResNet50 avec poids par défaut (ImageNet)
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)

        # Gel des poids (Feature Extraction) si demandé
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Remplacement de la couche finale (Fully Connected)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_last_conv_layer(self):
        # Nécessaire pour Grad-CAM : dernière couche conv du backbone
        return self.backbone.layer4[2].conv3