from sklearn.base import BaseEstimator
import numpy as np


class SVDLeastSquares(BaseEstimator):
    def __init__(self):
        self._w = None

    def fit(self, X_train, y_train):
        n, p = X_train.shape
        sigma_inv = np.zeros((p, n))

        u, sigma, vt = np.linalg.svd(X_train)
        sigma_inv[:min(n, p), :min(n, p)] = np.diag(1 / sigma)
        self._w = (vt.T @ sigma_inv @ u.T) @ y_train

        return self

    def predict(self, X_test):
        if self._w is None:
            print("Weights not learned. Fit the model on the training set first.")
        else:
            return X_test @ self._w