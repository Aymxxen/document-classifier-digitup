from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import numpy as np

class FlattenTransformer(BaseEstimator, TransformerMixin):
    """Transforme une image (H, W, C) en vecteur 1D"""
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # X est une liste d'images ou un array numpy
        # On aplatit chaque image
        return np.array([x.flatten() for x in X])

def get_classical_pipeline():
    """
    Retourne un pipeline Scikit-learn :
    1. Aplatissement de l'image
    2. Normalisation (StandardScaler)
    3. Réduction de dimension (PCA) pour accélérer
    4. Classifieur (SVM)
    """
    return Pipeline([
        ('flatten', FlattenTransformer()),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)), # Garde 95% de la variance
        ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced'))
    ])