'''

train_transaction['TransactionAmt'].apply(np.log)


train.groupby('ProductCD')['ProductCD'].count()

'''

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection._validation import cross_val_predict, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd

path = 'C:/public/ieee-fraud-detection/'
#path = 'C:/Users/seuk/git/project2-IEEE-CIS-Fraud-Detection/input/view/'
#path = '../input/ieee-fraud-detection/'

train_tr = pd.read_csv(path + 'train_transaction.csv')
test_tr = pd.read_csv(path + 'test_transaction.csv')

train_id = pd.read_csv(path + 'train_identity.csv')
test_id = pd.read_csv(path + 'test_identity.csv')

train = pd.merge(train_tr, train_id, on='TransactionID', how='left')
test = pd.merge(test_tr, test_id, on='TransactionID', how='left')

train['ProductCD']

ss = pd.read_csv(path + 'sample_submission.csv')

test_n = train.values
train_n = train.valu

dt_clf = DecisionTreeClassifier(random_state=156)

dt_clf.fit(train, label)

print('train_tr shape is {}'.format(train_tr.shape))
print('train shape is {}'.format(train.shape))

print('test_tr shape is {}'.format(test_tr.shape))
print('test shape is {}'.format(test.shape))


X_train, X_test, y_train, y_test = train_test_split(train_n, label_n, test_size=0.2, random_state=11)

dt_clf.fit(X_train, y_train)
pred = dt_clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))