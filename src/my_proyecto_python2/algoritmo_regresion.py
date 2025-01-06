import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv("/home/kent/my_proyecto_python2/dates/Advertising.csv")
data = data.iloc[:, 1:]


cols = ["TV", "Radio", "Newspaper"]

for col in cols:
    plt.plot(data[col], data["Sales"], "ro")
    plt.title("Ventas respecto a la publicidad en %s" % col)
    plt.show()

X = data["TV"].values.reshape(-1, 1)
y = data["Sales"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)
print("Predicciones: {}, Reales: {}".format(y_pred[:4], y_test[:4]))

rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)

print("R2:", r2_score(y_test, y_pred))

plt.plot(X_test, y_test, "ro")
plt.plot(X_test, y_pred)
plt.show()


def modelos_simple(x):

    X = data[x].values.reshape(-1, 1)
    y = data["Sales"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred = lin_reg.predict(X_test)
    print("Predicciones: {}, Reales: {}".format(y_pred[:4], y_test[:4]))
    rmse = root_mean_squared_error(y_test, y_pred)
    print("RMSE:", rmse)
    print("R2:", r2_score(y_test, y_pred))

    plt.plot(X_test, y_test, "ro")
    plt.plot(X_test, y_pred)
    plt.show()


modelos_simple("Radio")


"""Regresión lineal Múltiple"""


X = data.drop(["Radio", "Sales"], axis=1).values
y = data["Sales"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)
print("Predicciones: {}, Reales: {}".format(y_pred[:4], y_test[:4]))

rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)
print("R2:", r2_score(y_test, y_pred))

reg_sns = sns.regplot(x=y_test, y=y_pred)
