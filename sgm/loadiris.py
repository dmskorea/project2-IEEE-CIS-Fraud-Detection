from scipy.special.tests.test_sici import test_sici_consistency

from sklearn.datasets import  load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors.tests.test_nca import iris_data
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

iris = load_iris()
iris.feature_names

iris_data = iris.data

print(iris_data)

iris_data[0]

iris_label = iris.target
print(iris_label)
print(iris.target)

iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['lable'] = iris.target
iris_df.head(3)

X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label, test_size=0.2, random_state=11)

dt_clf = DecisionTreeClassifier(random_state=11)


dt_clf.fit(X_train, y_train)

pred = dt_clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))