from sklearn.datasets import  load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
from sklearn.model_selection._validation import cross_val_predict,\
    cross_val_score

iris = load_iris()
dt_clf = DecisionTreeClassifier(random_state=156)

data = iris.data
label = iris.target

scores = cross_val_score(dt_clf, data, label, scoring='accuracy', cv=3)

print(np.round(scores,4))
print(np.round(np.mean(scores),4))