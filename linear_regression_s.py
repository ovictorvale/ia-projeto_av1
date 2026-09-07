import numpy as np

class LinearRegressionS:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.b0 = None
        self.b1 = None

    def fit(self):
        xbar = np.mean(self.x)
        ybar = np.mean(self.y)

        self.b1 = (np.sum((self.y - ybar) * (self.x - xbar))) / (np.sum((self.x - xbar) ** 2))
        self.b0 = ybar - self.b1 * xbar

        return self

    def predict(self, x_new):
        return self.b0 + self.b1 * np.array(x_new)

    def rss(self):
        previsao = self.predict(self.x)
        residuos = self.y - previsao

        return np.sum(residuos ** 2) 

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

#####
    def mse(self):
        previsao = self.predict(self.x)
        erro = self.y - previsao

        return np.mean(erro ** 2)

    def rmse(self):
        previsao = self.predict(self.x)
        erro = self.y - previsao

        return np.sqrt(np.mean(erro ** 2))

    def mae(self):
        previsao = self.predict(self.x)
        erro = self.y - previsao

        return np.mean(np.abs(erro))
#####
    
    def summary(self):
        print(f"Interceptor: {self.b0}")
        print(f"Coeficiente Angular: {self.b1}")
        print(f"RSS: {self.rss()}")
        print(f"R²: {self.r2()}")
