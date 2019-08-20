import math
import os, sys, gc, warnings, random

from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

path = 'C:/Users/seuk/git/project2-IEEE-CIS-Fraud-Detection/input/'


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


SEED = 42
seed_everything(SEED)
LOCAL_TEST = False
TARGET = 'isFraud'

print('Load Data')
train_df = pd.read_pickle(path + 'train_transaction.pkl')

if LOCAL_TEST:
    test_df = train_df.iloc[-100000:, ].reset_index(drop=True)
    train_df = train_df.iloc[:400000, ].reset_index(drop=True)
    train_identity = pd.read_pickle(path + 'train_identity.pkl')
    test_identity = train_identity[train_identity['TransactionID'].isin(test_df['TransactionID'])].reset_index(drop=True)
    train_identity = train_identity[train_identity['TransactionID'].isin(train_df['TransactionID'])].reset_index(drop=True)
else:
    test_df = pd.read_pickle(path + 'test_transaction.pkl')
    test_identity = pd.read_pickle(path + 'test_identity.pkl')

valid_card = train_df['card1'].value_counts()
valid_card = valid_card[valid_card > 10]
valid_card = list(valid_card.index)
    
train_df['card1'] = np.where(train_df['card1'].isin(valid_card), train_df['card1'], np.nan)
test_df['card1'] = np.where(test_df['card1'].isin(valid_card), test_df['card1'], np.nan)

i_cols = ['card1', 'card2', 'card3', 'card5',
          'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
          'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
          'addr1', 'addr2',
          'dist1', 'dist2',
          'P_emaildomain', 'R_emaildomain'
         ]

for col in i_cols:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    fq_encode = temp_df[col].value_counts().to_dict()   
    train_df[col + '_fq_enc'] = train_df[col].map(fq_encode)
    test_df[col + '_fq_enc'] = test_df[col].map(fq_encode)


for col in ['ProductCD', 'M4']:
    temp_dict = train_df.groupby([col])[TARGET].agg(['mean']).reset_index().rename(columns={'mean': col + '_target_mean'})
    temp_dict.index = temp_dict[col].values
    temp_dict = temp_dict[col + '_target_mean'].to_dict()
    train_df[col + '_target_mean'] = train_df[col].map(temp_dict)
    test_df[col + '_target_mean'] = test_df[col].map(temp_dict)

for col in list(train_df):
    if train_df[col].dtype == 'O':
        print(col)
        train_df[col] = train_df[col].fillna('unseen_before_label')
        test_df[col] = test_df[col].fillna('unseen_before_label')
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
        le = LabelEncoder()
        le.fit(list(train_df[col]) + list(test_df[col]))
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')

rm_cols = [
    'TransactionID', 'TransactionDT', TARGET,
]
features_columns = list(train_df)
for col in rm_cols:
    if col in features_columns:
        features_columns.remove(col)

lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    'num_leaves': 2 ** 8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':1,
                    'n_estimators':800,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                    'early_stopping_rounds':100,
                } 

def make_predictions(tr_df, tt_df, features_columns, target, lgb_params, NFOLDS=2):
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    X, y = tr_df[features_columns], tr_df[target]    
    P, P_y = tt_df[features_columns], tt_df[target]  
    tt_df = tt_df[['TransactionID', target]]    
    predictions = np.zeros(len(tt_df))
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print('Fold:', fold_)
        tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
        vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]
        print(len(tr_x), len(vl_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)
        if LOCAL_TEST:
            vl_data = lgb.Dataset(P, label=P_y) 
        else:
            vl_data = lgb.Dataset(vl_x, label=vl_y)  
        estimator = lgb.train(
            lgb_params,
            tr_data,
            valid_sets=[tr_data, vl_data],
            verbose_eval=200,
        )   
        pp_p = estimator.predict(P)
        predictions += pp_p / NFOLDS
        if LOCAL_TEST:
            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(), X.columns)), columns=['Value', 'Feature'])
            print(feature_imp)
        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
        gc.collect()
    tt_df['prediction'] = predictions
    return tt_df

if LOCAL_TEST:
    test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params)
    print(metrics.roc_auc_score(test_predictions[TARGET], test_predictions['prediction']))
else:
    lgb_params['learning_rate'] = 0.01
    lgb_params['n_estimators'] = 800
    lgb_params['early_stopping_rounds'] = 100    
    test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params, NFOLDS=2)

if not LOCAL_TEST:
    test_predictions['isFraud'] = test_predictions['prediction']
    test_predictions[['TransactionID', 'isFraud']].to_csv(path +'submission.csv', index=False)