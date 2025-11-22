# 📄 Système de Classification Intelligente de Documents (Digitup)

**Auteur :** Mohammed Aymene Gheriani  
**Type :** Test Technique - Ingénieur IA  

##  Contexte & Objectif
Ce projet vise à automatiser le tri de documents administratifs (Factures, CV, Lettres, Rapports, etc.) . 
La solution propose une approche **Deep Learning (CNN)** robuste, couplée à une interface explicable (XAI) pour garantir la transparence des décisions.

##  Fonctionnalités Clés
1.  **Pipeline de Données** : Ingestion, nettoyage et split train/val/test reproductible.
2.  **Modélisation Hybride** :
    *   **Deep Learning** : ResNet50 (Transfer Learning) pour une haute performance.
    *   **Machine Learning Classique** : Pipeline SVM (supporté dans l'architecture).
3.  **Explicabilité (XAI)** : Intégration de **Grad-CAM** pour visualiser les zones de décision sur l'image.
4.  **Interface UI** : Application **Streamlit** interactive permettant l'upload, l'analyse et la **correction manuelle (Feedback loop)**.
5.  **Métriques** : Évaluation complète (F1-Score, Matrice de Confusion, Courbes ROC).

## 🛠️ Architecture Technique
Le projet est structuré de manière modulaire :
```bash
├── app/            # Interface utilisateur (Streamlit)
├── data/           # Scripts de gestion des données (Preprocessing)
├── models/         # Architectures (CNN ResNet50, SVM)
├── train/          # Scripts d'entraînement
├── eval/           # Scripts d'évaluation (ROC, Confusion Matrix)
├── explain/        # Moteur d'explicabilité (Grad-CAM)
└── feedback/       # Stockage des corrections utilisateurs
