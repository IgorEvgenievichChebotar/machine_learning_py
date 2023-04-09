import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

df = pd.read_csv('../wines.csv')
print(df.info())

df.hist(bins=50, figsize=(20, 15))
plt.show()

sns.pairplot(data=df, kind='scatter', diag_kind='kde')
plt.show()

df_train, df_test = sk.model_selection.train_test_split(df, train_size=0.2)

Y = df_train["quality"]
Y_t = df_test["quality"]

numeric = ['fixed acidity', 'volatile acidity', 'citric acid',
           'residual sugar', 'chlorides', 'free sulfur dioxide',
           'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

categorical = ['type']

X = df_train[numeric]

model = sk.linear_model.LinearRegression().fit(X, Y)

X_t = df_test[numeric]
Y_t_pred = model.predict(X_t)

print(f"MSE={sk.metrics.mean_squared_error(Y_t, Y_t_pred)},\
MAE={sk.metrics.mean_absolute_error(Y_t, Y_t_pred)},\
MAE(%)={sk.metrics.mean_absolute_percentage_error(Y_t, Y_t_pred)}")

transformer = make_column_transformer(
    (sk.preprocessing.OneHotEncoder(), ['type']),
    remainder='passthrough'
)

X = transformer.fit_transform(df_train[numeric + categorical])

model = sk.linear_model.LinearRegression().fit(X, Y)

X_t = transformer.transform(df_test[numeric + categorical])
Y_t_pred = model.predict(X_t)

print(f"MSE={sk.metrics.mean_squared_error(Y_t, Y_t_pred)},\
MAE={sk.metrics.mean_absolute_error(Y_t, Y_t_pred)},\
MAE(%)={sk.metrics.mean_absolute_percentage_error(Y_t, Y_t_pred)}")
