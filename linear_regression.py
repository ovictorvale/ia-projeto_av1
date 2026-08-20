import numpy as np

class LinearRegression:
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

    def r2(self):
        return 1 - ((np.sum((self.y - self.predict(self.x)) ** 2)) / (np.sum((self.y - np.mean(self.y)) ** 2)))

    def summary(self):
        print(f"Interceptor: {self.b0}")
        print(f"Coeficiente Angular: {self.b1}")
        print(f"RSS: {self.rss()}")
        print(f"R²: {self.r2()}")