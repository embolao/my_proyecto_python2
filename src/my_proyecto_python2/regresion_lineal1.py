import matplotlib.pyplot as plt

# import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1️⃣ Cargar datos
data = load_diabetes()
X = data.data[:, 2].reshape(-1, 1)  # Usamos solo la columna de IMC
y = data.target  # Progresión de la enfermedad

# 2️⃣ Dividir en entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ Crear y entrenar modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 4️⃣ Hacer predicciones
y_pred = modelo.predict(X_test)

# 5️⃣ Evaluar el modelo
pendiente = modelo.coef_[0]
interseccion = modelo.intercept_
precision = r2_score(y_test, y_pred)

print(f"Ecuación: Progresión = {interseccion:.2f} + {pendiente:.2f} * IMC")
print(f"Precisión del modelo (R²): {precision:.4f}")

# 6️⃣ Graficar resultados
plt.scatter(X_test, y_test, color="blue", label="Datos reales")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Regresión Lineal")
plt.xlabel("Índice de Masa Corporal (IMC)")
plt.ylabel("Progresión de la Diabetes")
plt.title("Regresión Lineal: IMC vs. Progresión de la Diabetes")
plt.legend()
plt.show()
