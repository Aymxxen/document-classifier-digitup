import unittest
import torch
import sys
import os

# Ajout du path pour importer models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn import DocumentClassifierCNN


class TestModel(unittest.TestCase):
    def test_model_structure(self):
        """Vérifie que le modèle sort bien le bon nombre de classes"""
        num_classes = 10
        model = DocumentClassifierCNN(num_classes=num_classes, freeze_backbone=True)

        # Créer une fausse image (Batch=1, Channels=3, H=224, W=224)
        fake_input = torch.randn(1, 3, 224, 224)

        # Forward pass
        output = model(fake_input)

        # Vérifier la taille de la sortie (doit être [1, 10])
        self.assertEqual(output.shape, (1, num_classes))
        print("\n✅ Test structure modèle : SUCCÈS")


if __name__ == '__main__':
    unittest.main()