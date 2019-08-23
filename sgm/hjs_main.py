import os
import time
import random
import datetime

import numpy as np 
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import decomposition
from sklearn import model_selection

# load data
# datapath = '/kaggle/input'

datapath = 'C:/Users/seuk/git/project2-IEEE-CIS-Fraud-Detection/input/'
trainid_df = pd.read_csv(os.path.join(datapath, 'train_identity.csv'))
traintx_df = pd.read_csv(os.path.join(datapath, 'train_transaction.csv'))
testid_df = pd.read_csv(os.path.join(datapath, 'test_identity.csv'))
testtx_df = pd.read_csv(os.path.join(datapath, 'test_transaction.csv'))
submit_df = pd.read_csv(os.path.join(datapath, 'sample_submission.csv'))

# merge dfs
train_df = traintx_df.merge(trainid_df, how='left', left_index=True, right_index=True)
test_df = testtx_df.merge(testid_df, how='left', left_index=True, right_index=True)

# unload raw csvs from the memory
del trainid_df
del traintx_df
del testid_df
del testtx_df

# set seed
random.seed(42)

# remove columns with a high proportion of null values
nb_rows = train_df.shape[0]
cols_to_remove = []
for col in train_df.columns:
    if train_df[col].isna().sum() / nb_rows >= .7:
        cols_to_remove.append(col)

train_df.drop(cols_to_remove, axis=1, inplace=True)
test_df.drop(cols_to_remove, axis=1, inplace=True)

# encode categorized columns
cat_cols = [
    'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19',
    'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27',
    'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35',
    'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD',
    'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
    'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
    'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3',
    'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']
for col in cat_cols:
    if col in train_df.columns:
        le = preprocessing.LabelEncoder()
        le.fit(list(train_df[col].astype(str).values) + list(test_df[col].astype(str).values))
        train_df[col] = le.transform(list(train_df[col].astype(str).values))
        test_df[col] = le.transform(list(test_df[col].astype(str).values))

# split train dataset into X and y
X = train_df.drop(['isFraud'], axis=1)
y = train_df['isFraud']
del train_df

# simple LGBM (taken from https://www.kaggle.com/nroman/lgb-single-model-lb-0-9419)
params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth':-1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity":-1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47
         }
aucs = list()
folds = model_selection.TimeSeriesSplit(n_splits=5)
training_start_time = time.time()

for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):
    start_time = time.time()
    print('Training on fold {}'.format(fold + 1))
    trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])
    clf = lgb.train(params, trn_data, 10000, valid_sets=[trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)
    aucs.append(clf.best_score['valid_1']['auc'])
    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time.time() - start_time))))
    
print('-' * 30)
print('Training has finished.')
print('Total training time is {}'.format(str(datetime.timedelta(seconds=time.time() - training_start_time))))
print('Mean AUC:', np.mean(aucs))
print('-' * 30)

# predict
clf = lgb.LGBMClassifier(**params, num_boost_round=clf.best_iteration)
clf.fit(X, y)

# submit
submit_df['isFraud'] = clf.predict_proba(test_df)[:, 1]
submit_df.to_csv(datapath + 'ieee_cis_fraud_detection.csv', index=False)
