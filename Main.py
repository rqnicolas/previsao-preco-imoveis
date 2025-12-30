import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/imoveis.csv")

X = df[["tamanho_m2", "quartos", "banheiros"]]
y = df["preco"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = LinearRegression()
modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)

erro = mean_absolute_error(y_teste, previsoes)
print("Erro médio:", erro)
