from linear_regression_m import MRegression
from linear_regression_s import LinearRegressionS
from linear_regression_m import MRegression, MRegressionZeroIntercept
import numpy as np

dados = np.loadtxt(r"dose_radiacao_expandido.csv", delimiter=",", skiprows=1)

y = dados[:, 1]      # Dose de Radiação
x1 = dados[:, 2]     # Corrente (mAmp)
x2 = dados[:, 3]     # Tempo de Exposição

#a)
X = np.column_stack((x1, x2)) # Matriz dos preditores
modelo = MRegression(X, y)
modelo.fit()
y_preditivo = modelo.predict(X)
ei = y - y_preditivo
rss = modelo.rss(y, y_preditivo)


print("\na) "
      "Regressão Linear Múltipla")
print("\nX:")
print(X)
print("\ny:")
print(y)
print("\nCoeficientes:")
print("b0 (Intercepto):", modelo.beta[0])
print("b1 (mAmp):", modelo.beta[1])
print("b2 (Tempo de Exposição):", modelo.beta[2])
print("\nValores ajustados:")
print(y_preditivo)
print("\nResíduos (ei):")
print(ei)
print("\nRSS:")
print(rss)

#b) Previsão para 15 mA e 5 minutos
X_novo = np.array([[15, 5]])
dose_prevista = modelo.predict(X_novo)

print("\nb)")
print("Dose de radiação prevista:", dose_prevista[0])

#c) R² Score
r2 = modelo.r2_score(y, y_preditivo)

print("\nc)")
print("R² Score:", r2)

#d) R² ajustado
n = len(y)
p = 2
r2_ajus = modelo.r2_score_ajus(r2, n, p)

print("\nd)")
print("O R² ajustado é utilizado porque leva em consideração a quantidade de variáveis independentes "
      "\nutilizadas no modelo e penaliza a inclusão de novos preditores quando eles não proporcionam "
      "\numa melhoria proporcional na capacidade de explicação/previsão do modelo \n")
print("R² Score:", r2)
print("R² ajustado:", r2_ajus)
print("\nSim, o R² ajustado é mais adequado para avaliar o modelo. Embora tenha sido ligeiramente menor que o R² comum,"
      "\na diferença foi muito pequena, o que quer dizer que as variáveis adicionadas contribuem para o poder de explicação do modelo" 
      "\ne não apenas aumentam artificialmente o R²")

# e) RLS
modelo_simples = LinearRegressionS(x1, y)
modelo_simples.fit()
y_preditivo_simples = modelo_simples.predict(x1)
r2_simples = modelo_simples.r2_score(y, y_preditivo_simples)
n = len(y)
p_simples = 1
r2_ajus_simples = modelo_simples.r2_score_ajus(r2_simples, n, p_simples)

print("\ne)")
print("Comparação entre os modelos")
print("\nRegressão Linear Simples")
print("Intercepto:", modelo_simples.b0)
print("Coeficiente da Corrente:", modelo_simples.b1)
print("R²:", r2_simples)
print("R² ajustado:", r2_ajus_simples)

print("\nModelo de Regressão Linear Múltipla")
print("R²:", r2)
print("R² ajustado:", r2_ajus)

print("\nO melhor modelo é a Regressão Linear Múltipla.")
print("Isso ocorre porque o modelo que utiliza Corrente e Tempo de Exposição")
print("possui um R² ajustado maior, explicando melhor a variação")
print("da Dose de Radiação.")

#f) RLM com Intercepto Forçado a Zero
modelo_zero = MRegressionZeroIntercept(X, y)
modelo_zero.fit()
y_preditivo_zero = modelo_zero.predict(X)
r2_zero = modelo_zero.r2_score(y, y_preditivo_zero)
rmse_zero = modelo_zero.rmse(y, y_preditivo_zero)
rmse_intercepto = modelo.rmse(y, y_preditivo)


print("\nf)")
#primeira pergunta do item
print("Impor a restrição de que o Intercepto seja 0 "
      "significa que o modelo é obrigado a passar pela origem,"
      "ou seja, quando a Corrente e o Tempo de Exposição forem iguais a zero,"
      "a Dose de Radiação prevista também será necessariamente igual a zero.")
print("\nModelo com Intercepto Forçado a Zero")
print("b1 (mAmp):", modelo_zero.beta[0])
print("b2 (Tempo de Exposição):", modelo_zero.beta[1])
print("R²:", r2_zero)
print("RMSE:", rmse_zero)
print("\nModelo com Intercepto")
print("R²:", r2)
print("RMSE:", rmse_intercepto)

print("\nEu escolheria o modelo com intercepto não é forçado a zero" \
      "os resultados mostram que impor essa condição ao modelo piorou\n"
      "seu ajuste aos dados observados. ")

#h) Métricas de erro
#Modelo completo - RLM
mse_completo = modelo.mse(y, y_preditivo)
rmse_completo = modelo.rmse(y, y_preditivo)
mae_completo = modelo.mae(y, y_preditivo)

# Modelo alternativo - RLS
mse_simples = modelo_simples.mse()
rmse_simples = modelo_simples.rmse()
mae_simples = modelo_simples.mae()

print("\nh)")
print("\nMSE:")
print("Modelo completo:", mse_completo)
print("Modelo alternativo:", mse_simples)
print("O modelo completo apresentou um MSE menor, indicando menos erro.")
print("Como o MSE penaliza mais os erros grandes, o modelo alternativo")
print("apresenta erros consideravelmente maiores.")

print("\nRMSE:")
print("Modelo completo:", rmse_completo)
print("Modelo alternativo:", rmse_simples)
print("O modelo completo apresentou um RMSE menor.")
print("Como o RMSE está na mesma unidade da Dose de Radiação,")
print("isso indica que o tamanho típico do erro é menor no modelo completo.")

print("\nMAE:")
print("Modelo completo:", mae_completo)
print("Modelo alternativo:", mae_simples)
print("O modelo completo apresentou um MAE menor.")
print("Isso indica que suas previsões ficam, em média, mais próximas")
print("dos valores observados.")

print("\nConclusão:")
print("O modelo completo apresentou menores erros nas três métricas.")
print("Portanto, ele é consistentemente melhor que o modelo alternativo.")
print("Isso indica que adicionar o Tempo de Exposição melhora")
print("significativamente a capacidade de previsão da Dose de Radiação.")
