import graphviz
import pandas as pd
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

df = pd.read_csv('../wines.csv')

df["ntype"] = df['type'].apply(lambda x: 0 if x == "White" else 1)

df_train, df_test = sk.model_selection.train_test_split(df, train_size=0.2)

numeric = ['fixed acidity', 'volatile acidity', 'citric acid',
           'residual sugar', 'chlorides', 'free sulfur dioxide',
           'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

Y = df_train["type"]
Y_test = df_test["type"]

X = df_train[numeric]
X_test = df_test[numeric]

model = DecisionTreeClassifier()
model = model.fit(X, Y)
Y_pred = model.predict(X_test)

accuracy = sk.metrics.accuracy_score(Y_test, Y_pred)
print(accuracy)

graphviz.backend.dot_command.DOT_BINARY = 'C:/Program Files/Graphviz/bin/dot.exe'
graph = graphviz.Source(export_graphviz(
    model,
    feature_names=numeric,
    class_names=['White', 'Red'],
    filled=True,
    rounded=True,
    special_characters=True,
    impurity=False)
)

graph.render("iris")
