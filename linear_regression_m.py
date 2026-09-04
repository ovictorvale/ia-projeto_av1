import numpy as np

class MRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.beta = None
        self.N = X.shape[0]

    def fit(self):
        self.X = np.column_stack((np.ones(self.N), self.X))
        self.beta = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y

    def predict(self, X_new):
        N = X_new.shape[0]
        X_new = np.column_stack((np.ones(N), X_new))
        return X_new @ self.beta

    def r2_score(self, y_true, y_preditivo):
        numerador = np.sum((y_true - y_preditivo) ** 2)
        denominador = np.sum((y_true - np.mean(y_true)) ** 2)

        r2_score = 1 - (numerador / denominador)

        return r2_score

    def r2_score_ajus(self, r2, n, p):
        numerador = (1 - r2) * (n - 1)
        denominador = (n - p - 1)

        r2_score_ajus =  1 - (numerador / denominador)

        return r2_score_ajus