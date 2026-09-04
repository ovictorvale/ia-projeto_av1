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
print("\nJustificativa: ")
print("Conforme pode ser observado a partir das saídas o presente modelo se mostrou mais adequado.\n" \
"Mesmo o mdelo mais completo tendo apresentado um R² maior, o seu R² ajustado foi menor. Isso nos \n" \
"indica que as variáveis adicionais tiveram pouco impácto no poder explicativo do modelo.")


#f2)
print("\nf2)")
residuos = y - y_preditivo
print("Resíduos: ")
print(residuos)

neg = 0
pos = 0

print("\nObservação i  |  Valor observado yi  |  Valor ajustado ^y  |  Resíduo ei  ")
for i in range(len(y)):
    print(f"    {i + 1}      |  {y[i]}  |  {y_preditivo[i]}  |  {residuos[i]}")


for i in range(len(y)):
    if residuos[i] > 0:
        pos += 1
    else:
        neg += 1

print(f"\nQuantidade de observações positivas: {pos} - Representatividade: {(pos/len(y) * 100):.2f}%")
print(f"\nQuantidade de observações negativas: {neg} - Representatividade: {(neg/len(y) * 100):.2f}%")

print("\nAnálise dos resíduos: ")
print("Conforme pode ser obsevado após a análise dos resíduos, alguns apresentaram valores positivos enquanto outros apresentaram\n" \
"valores negativos. Isso mostra que as predições tanto podem apresentar valores acima do real quanto abaixo do real. Mas importante mecionar\n" \
"que, em sua maioria, os erros apresentados são próximos de zero.")

#g)
print("\ng)")
beta_zero = np.linalg.pinv(X.T @ X) @ X.T @ y

y_preditivo_0 = X @ beta_zero
print("Coeficinetes do modelo com interceptor forçado a zero: ")
print(beta_zero)

r2_0 = modelo.r2_score(y, y_preditivo_0)
print(f"\nR² do modelo com interceptor forçado a zero: {r2_0:.4f}")
print(f"R² do modelo com interceptor: {r2:.4f}")

rmse_interceptor = modelo.rmse(y, y_preditivo)
rmse_sem_inteceptor = modelo.rmse(y, y_preditivo_0)

print(f"\nRMSE modelo com interceptor: {rmse_interceptor:.4f}")
print(f"RMSE modelo sem interceptor: {rmse_sem_inteceptor:.4f}")

print("\nJustificativas: ")
print("Intepretação prática do interceptor sendo igual a zero: ")
print("Forçar a interceptor a zero indica que a previsão também será igual a zero, \n" \
"quando todas as variáveis independentes forem zero. Porém, essa situação não representa necessarimente um caso real\n" \
"do conjunto de daods, já que algumas vriáveis não assumem valor zero.")

print("\nComparação das métricas:")
print("Conforme pode ser observado a partir do retorno do terminal o modelo com o intercéptor forçado a zero\n" \
"possui o R²  menor que o do modelo que adota o intercéptor, já analisando o RMSE o modelo sem o intercéptor possui um\n" \
"valor maior que o que adota o interceptor. Com os dados das métricas podemos afirmar que o modelo que adota o interceptor\n" \
"tende a apresentar um melhor desempenho.")

#h)
print("\nh)")

