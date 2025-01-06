import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar el dataset de California Housing
california = fetch_california_housing()

# Usar una característica (por ejemplo, el número de habitaciones)
X = california.data[:, 3][:, np.newaxis]  # Media de habitaciones
y = california.target  # Precio medio de la vivienda

# Dividir en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Estandarización de los datos
scaler = StandardScaler()
# Ajuste y transformación en el conjunto de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Solo transformación en el conjunto de prueba

# Crear y ajustar un modelo Random Forest
modelo_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
)
modelo_rf.fit(X_train_scaled, y_train)

# Hacer predicciones con Random Forest
y_pred_rf = modelo_rf.predict(X_test_scaled)

# Crear y ajustar un modelo Gradient Boosting
modelo_gb = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42
)
modelo_gb.fit(X_train_scaled, y_train)

# Hacer predicciones con Gradient Boosting
y_pred_gb = modelo_gb.predict(X_test_scaled)

# Evaluación del modelo Random Forest
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Evaluación del modelo Gradient Boosting
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)

# Imprimir los resultados
# print(f"\nResultados del modelo Random Forest:")
print(f"  MSE: {mse_rf:.2f}")
print(f"  RMSE: {rmse_rf:.2f}")
print(f"  R²: {r2_rf:.2f}")

# print(f"\nResultados del modelo Gradient Boosting:")
print(f"  MSE: {mse_gb:.2f}")
print(f"  RMSE: {rmse_gb:.2f}")
print(f"  R²: {r2_gb:.2f}")

# Graficar los datos reales y las predicciones de ambos modelos
plt.figure(figsize=(12, 6))

# Subgráfico para Random Forest
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color="blue", label="Datos reales", alpha=0.5)
plt.scatter(
    X_test, y_pred_rf, color="red", label="Predicciones de Random Forest", alpha=0.5
)
plt.xlabel("Número de habitaciones")
plt.ylabel("Valor medio de la vivienda ($1000s USD)")
plt.title("Random Forest Regressor")
plt.legend()

# Subgráfico para Gradient Boosting
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color="blue", label="Datos reales", alpha=0.5)
plt.scatter(
    X_test,
    y_pred_gb,
    color="green",
    label="Predicciones de Gradient Boosting",
    alpha=0.5,
)
plt.xlabel("Número de habitaciones")
plt.ylabel("Valor medio de la vivienda ($1000s USD)")
plt.title("Gradient Boosting Regressor")
plt.legend()

# Mostrar las gráficas
plt.tight_layout()
plt.show()
