import pandas as pd
import sklearn as sk
from sklearn import *
from pandas import *

# Загрузка данных о вине из CSV файла и использование первого столбца в качестве индекса
df = pd.read_csv('../wines.csv', index_col=0)

# Замена категориальной целевой переменной 'type' числовыми значениями с помощью LabelEncoder
le = sk.preprocessing.LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Разделение набора данных на обучающую и тестовую выборки, с 20% данных в обучающей выборке
df_train, df_test = sk.model_selection.train_test_split(df, train_size=0.2)

# Создание списка для числовых столбцов в наборе данных
numeric = ['fixed acidity', 'volatile acidity', 'citric acid',
           'residual sugar', 'chlorides', 'free sulfur dioxide',
           'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

# Извлечение меток (т.е. столбца "type") из обучающей и тестовой выборок
Y = df_train["type"]
Y_test = df_test["type"]

# Извлечение числовых функций из обучающей и тестовой выборок
X = df_train[numeric]
X_test = df_test[numeric]

# Обучение классификатора дерева решений на обучающей выборке
logistic_regression = sk.tree.DecisionTreeClassifier().fit(X, Y)

# Использование обученного классификатора для прогнозирования меток в тестовой выборке
Y_pred_linear = logistic_regression.predict(X_test)

# Обучение классификатора k-ближайших соседей на обучающей выборке
knn = sk.neighbors.KNeighborsClassifier().fit(X, Y)

# Использование обученного классификатора для прогнозирования меток в тестовой выборке
Y_pred_knn = knn.predict(X_test)

# Обучение классификатора опорных векторов на обучающей выборке
svm = sk.svm.SVC(kernel='linear', C=1).fit(X, Y)

# Использование обученного классификатора для прогнозирования меток в тестовой выборке
Y_pred_svm = svm.predict(X_test)

# Вычисление и печать точности каждого классификатора на тестовой выборке
print(sk.metrics.accuracy_score(Y_test, Y_pred_linear))
print(sk.metrics.accuracy_score(Y_test, Y_pred_knn))
print(sk.metrics.accuracy_score(Y_test, Y_pred_svm))
