import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import os

plt.hist
print(os.listdir("../input"))

train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
print("1")

train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
print("1")

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
print("1")

del train_transaction, train_identity
print("1")

train.shape
print("1")

train.head().T
print("1")
