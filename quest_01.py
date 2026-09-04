from linear_regression_m import MRegression
from linear_regression_s import LinearRegressionS
import numpy as np

dados = np.loadtxt(r"arsenio_dataset (1).csv", delimiter=",", skiprows=1)
# print(dados)

y = dados[:, 5] #Arsênio nas unhas
x1 = dados[:, 0]#Idade
x2 = dados[:, 2]#Uso para beber
x3 = dados[:, 3]#Uso para conzinhar
x4 = dados[:, 4]#Arsênio na água

#a)
X = np.column_stack((x1, x2, x3, x4))

modelo = MRegression(X, y)

modelo.fit()

y_preditivo = modelo.predict(X)

print("\na)")
print("Coeficientes do modelo: ")
print(modelo.beta)

print("\nValores ajustados: ")
print(y_preditivo)

#b)

item_b = np.array([[30, 5, 5, 0.135]])

y_preditivo_b = modelo.predict(item_b)

print("\nb)")
print("Previsão do asernio nas unhas: ")
print(y_preditivo_b)

#d)
print("\nd)")
print("Valor R² para o modelo: ")
r2 = modelo.r2_score(y, y_preditivo)
print(r2)

#e)
print("\ne)")
n = len(y)
p = 4
r2_ajus = modelo.r2_score_ajus(r2, n, p)
print("Valor de R² ajustado: ")
print(r2_ajus)
print("\nJustificativa: ")
print("O R² ajustado tende a ser mais indicado pois considera a quantidade de " \
"\npreditores do modelo e penaliza o aumento da complexidade quando esse aumento não \n" 
"apresenta um ganho significativo na explicação dos dados. No caso dessa desse item ao aplicar \n" \
"o R² ajustado pode-se obeservar que houve uma redução no valor se comparado ao R² sem ajuste, onde \n" \
"isso demonstra que o modelo está adontando variáveis que não tem uma considerável melhora no valor explicativo do modelo.\n")

#f1)
print("\nf1)")
modelo_2 = LinearRegressionS(x4, y)
modelo_2.fit()
y_preditivo_2 = modelo_2.predict(x4)

r2_2 = modelo_2.r2_score(y, y_preditivo_2)

n2 = len(y)
p2 = 1

r2_2_ajus = modelo_2.r2_score_ajus(r2_2, n2, p2)

print("\nR² modelo alternativo: ")
print(r2_2)
print("\nR² modelo ajustdo alternativo: ")
print(r2_2_ajus)