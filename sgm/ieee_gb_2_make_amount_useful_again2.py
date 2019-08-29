# General imports
import numpy as np
import pandas as pd
import os, sys, gc, warnings, random, datetime
import lightgbm as lgb

from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

import math
warnings.filterwarnings('ignore')

LOCAL_TEST = False


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)



def init(path = 'C:/public/ieee-fraud-detection/'):
    
    SEED = 42
    seed_everything(SEED)
    TARGET = 'isFraud'
    START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
    
    print('Load Data')
    train_df = pd.read_pickle(path + 'train_transaction.pkl')
    
    if LOCAL_TEST:
        
        # Convert TransactionDT to "Month" time-period. 
        # We will also drop penultimate block 
        # to "simulate" test set values difference
        train_df['DT_M'] = train_df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        train_df['DT_M'] = (train_df['DT_M'].dt.year - 2017) * 12 + train_df['DT_M'].dt.month 
        test_df = train_df[train_df['DT_M'] == train_df['DT_M'].max()].reset_index(drop=True)
        train_df = train_df[train_df['DT_M'] < (train_df['DT_M'].max() - 1)].reset_index(drop=True)
        
        train_identity = pd.read_pickle(path + 'train_identity.pkl')
        test_identity = train_identity[train_identity['TransactionID'].isin(
                                        test_df['TransactionID'])].reset_index(drop=True)
        train_identity = train_identity[train_identity['TransactionID'].isin(
                                        train_df['TransactionID'])].reset_index(drop=True)
        del train_df['DT_M'], test_df['DT_M']
        
    else:
        test_df = pd.read_pickle(path + 'test_transaction.pkl')
        train_identity = pd.read_pickle(path + 'train_identity.pkl')
        test_identity = pd.read_pickle(path + 'test_identity.pkl')
        
    base_columns = list(train_df) + list(train_identity)
    print('Shape control:', train_df.shape, test_df.shape)
    
    ########################### All features columns
    #################################################################################
    # # Main Data
    # 'TransactionID',                     -> This is pure noise, we cannot use this column as feature
    # 'isFraud',                           -> Our Target
    # 'TransactionDT',                     -> Time from reference time point. VERY valuable column
    # 'TransactionAmt',                    -> Many unique values and has to be combined with other columns
    #                                         The best score boost should come from 
    #                                         TransactionDT->TransactionAmt combination
    # 'ProductCD',                         -> 100% categorical feature options to use:
    #                                         Frequency encoding/Target encoding/
    #                                         Combinations with other columns/Model categorical feature
    # 'card1' - 'card6',                   -> Categorical features with information about Client
    # 'addr1' - 'addr2',                   -> add2 - Country / addr1 - subzone
    # 'dist1' - 'dist2',                   -> dist2 - Country distance / dist1 - local distance from merchant
    # 'P_emaildomain' - 'R_emaildomain',   -> Categorical feature. It's possible to make 
    #                                         subgroup feature from it or general group
    # 'C1' - 'C14'                         -> Counts. Should be numerical features (all ints?)
    # 'D1' - 'D15'                         
    # 'M1' - 'M9'
    # 'V1' - 'V339'
    
    # # Identity Data
    # 'TransactionID'
    # 'id_01' - 'id_38'
    # 'DeviceType',
    # 'DeviceInfo'
    
    ########################### D9 and TransactionDT
    # Let's add temporary "time variables" for aggregations
    # and add normal "time variables"
    
    # Also, seems that D9 column is an hour
    # and it is the same as df['DT'].dt.hour
    for df in [train_df, test_df]:
        # Temporary
        df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        df['DT_M'] = (df['DT'].dt.year - 2017) * 12 + df['DT'].dt.month
        df['DT_W'] = (df['DT'].dt.year - 2017) * 52 + df['DT'].dt.weekofyear
        df['DT_D'] = (df['DT'].dt.year - 2017) * 365 + df['DT'].dt.dayofyear
        
        df['DT_hour'] = df['DT'].dt.hour
        df['DT_day_week'] = df['DT'].dt.dayofweek
        df['DT_day'] = df['DT'].dt.day
        
        # D9 column
        df['D9'] = np.where(df['D9'].isna(), 0, 1)
        
    ########################### Reset values for "noise" card1
    i_cols = ['card1']
    
    for col in i_cols: 
        valid_card = pd.concat([train_df[[col]], test_df[[col]]])
        valid_card = valid_card[col].value_counts()
        valid_card = valid_card[valid_card > 2]
        valid_card = list(valid_card.index)
    
        train_df[col] = np.where(train_df[col].isin(test_df[col]), train_df[col], np.nan)
        test_df[col] = np.where(test_df[col].isin(train_df[col]), test_df[col], np.nan)
    
        train_df[col] = np.where(train_df[col].isin(valid_card), train_df[col], np.nan)
        test_df[col] = np.where(test_df[col].isin(valid_card), test_df[col], np.nan)
    
    ########################### M columns (except M4)
    # All these columns are binary encoded 1/0
    # We can have some features from it
    i_cols = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']
    
    for df in [train_df, test_df]:
        df['M_sum'] = df[i_cols].sum(axis=1).astype(np.int8)
        df['M_na'] = df[i_cols].isna().sum(axis=1).astype(np.int8)
    
    ########################### ProductCD and M4 Target mean
    for col in ['ProductCD', 'M4']:
        temp_dict = train_df.groupby([col])[TARGET].agg(['mean']).reset_index().rename(
                                                            columns={'mean': col + '_target_mean'})
        temp_dict.index = temp_dict[col].values
        temp_dict = temp_dict[col + '_target_mean'].to_dict()
    
        train_df[col + '_target_mean'] = train_df[col].map(temp_dict)
        test_df[col + '_target_mean'] = test_df[col].map(temp_dict)
    
    ########################### TransactionAmt
    
    # Let's add some kind of client uID based on cardID ad addr columns
    # The value will be very specific for each client so we need to remove it
    # from final feature. But we can use it for aggregations.
    train_df['uid'] = train_df['card1'].astype(str) + '_' + train_df['card2'].astype(str)
    test_df['uid'] = test_df['card1'].astype(str) + '_' + test_df['card2'].astype(str)
    
    train_df['uid2'] = train_df['uid'].astype(str) + '_' + train_df['card3'].astype(str) + '_' + train_df['card5'].astype(str)
    test_df['uid2'] = test_df['uid'].astype(str) + '_' + test_df['card3'].astype(str) + '_' + test_df['card5'].astype(str)
    
    train_df['uid3'] = train_df['uid2'].astype(str) + '_' + train_df['addr1'].astype(str) + '_' + train_df['addr2'].astype(str)
    test_df['uid3'] = test_df['uid2'].astype(str) + '_' + test_df['addr1'].astype(str) + '_' + test_df['addr2'].astype(str)
    
    # Check if the Transaction Amount is common or not (we can use freq encoding here)
    # In our dialog with a model we are telling to trust or not to these values   
    train_df['TransactionAmt_check'] = np.where(train_df['TransactionAmt'].isin(test_df['TransactionAmt']), 1, 0)
    test_df['TransactionAmt_check'] = np.where(test_df['TransactionAmt'].isin(train_df['TransactionAmt']), 1, 0)
    
    # For our model current TransactionAmt is a noise
    # https://www.kaggle.com/kyakovlev/ieee-check-noise
    # (even if features importances are telling contrariwise)
    # There are many unique values and model doesn't generalize well
    # Lets do some aggregations
    i_cols = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2', 'uid3']
    
    for col in i_cols:
        for agg_type in ['mean', 'std']:
            new_col_name = col + '_TransactionAmt_' + agg_type
            temp_df = pd.concat([train_df[[col, 'TransactionAmt']], test_df[[col, 'TransactionAmt']]])
            # temp_df['TransactionAmt'] = temp_df['TransactionAmt'].astype(int)
            temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
                                                    columns={agg_type: new_col_name})
            
            temp_df.index = list(temp_df[col])
            temp_df = temp_df[new_col_name].to_dict()   
        
            train_df[new_col_name] = train_df[col].map(temp_df)
            test_df[new_col_name] = test_df[col].map(temp_df)
               
    # Small "hack" to transform distribution 
    # (doesn't affect auc much, but I like it more)
    # please see how distribution transformation can boost your score 
    # (not our case but related)
    # https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html
    train_df['TransactionAmt'] = np.log1p(train_df['TransactionAmt'])
    test_df['TransactionAmt'] = np.log1p(test_df['TransactionAmt'])        
    
    ########################### 'P_emaildomain' - 'R_emaildomain'
    p = 'P_emaildomain'
    r = 'R_emaildomain'
    uknown = 'email_not_provided'
    
    for df in [train_df, test_df]:
        df[p] = df[p].fillna(uknown)
        df[r] = df[r].fillna(uknown)
        
        # Check if P_emaildomain matches R_emaildomain
        df['email_check'] = np.where((df[p] == df[r]) & (df[p] != uknown), 1, 0)
    
        df[p + '_prefix'] = df[p].apply(lambda x: x.split('.')[0])
        df[r + '_prefix'] = df[r].apply(lambda x: x.split('.')[0])
    
    # # Local test doesn't show any boost here, 
    # # but I think it's a good option for model stability 
    
    # # Also, we will do frequency encoding later
    
    ########################### Device info
    for df in [train_identity, test_identity]:
        ########################### Device info
        df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
        df['DeviceInfo_device'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
        df['DeviceInfo_version'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))
        
        ########################### Device info 2
        df['id_30'] = df['id_30'].fillna('unknown_device').str.lower()
        df['id_30_device'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
        df['id_30_version'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))
        
        ########################### Browser
        df['id_31'] = df['id_31'].fillna('unknown_device').str.lower()
        df['id_31_device'] = df['id_31'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
    
    ########################### Merge Identity columns
    temp_df = train_df[['TransactionID']]
    temp_df = temp_df.merge(train_identity, on=['TransactionID'], how='left')
    del temp_df['TransactionID']
    train_df = pd.concat([train_df, temp_df], axis=1)
        
    temp_df = test_df[['TransactionID']]
    temp_df = temp_df.merge(test_identity, on=['TransactionID'], how='left')
    del temp_df['TransactionID']
    test_df = pd.concat([test_df, temp_df], axis=1)    
    
    ########################### Freq encoding
    i_cols = ['card1', 'card2', 'card3', 'card5',
              'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
              'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
              'addr1', 'addr2',
              'dist1', 'dist2',
              'P_emaildomain', 'R_emaildomain',
              'DeviceInfo', 'DeviceInfo_device', 'DeviceInfo_version',
              'id_30', 'id_30_device', 'id_30_version',
              'id_31_device',
              'id_33',
              'uid', 'uid2', 'uid3',
             ]
    
    for col in i_cols:
        temp_df = pd.concat([train_df[[col]], test_df[[col]]])
        fq_encode = temp_df[col].value_counts(dropna=False).to_dict()   
        train_df[col + '_fq_enc'] = train_df[col].map(fq_encode)
        test_df[col + '_fq_enc'] = test_df[col].map(fq_encode)
    
    for col in ['DT_M', 'DT_W', 'DT_D']:
        temp_df = pd.concat([train_df[[col]], test_df[[col]]])
        fq_encode = temp_df[col].value_counts().to_dict()
                
        train_df[col + '_total'] = train_df[col].map(fq_encode)
        test_df[col + '_total'] = test_df[col].map(fq_encode)
    
    periods = ['DT_M', 'DT_W', 'DT_D']
    i_cols = ['uid']
    for period in periods:
        for col in i_cols:
            new_column = col + '_' + period
                
            temp_df = pd.concat([train_df[[col, period]], test_df[[col, period]]])
            temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)
            fq_encode = temp_df[new_column].value_counts().to_dict()
                
            train_df[new_column] = (train_df[col].astype(str) + '_' + train_df[period].astype(str)).map(fq_encode)
            test_df[new_column] = (test_df[col].astype(str) + '_' + test_df[period].astype(str)).map(fq_encode)
            
            train_df[new_column] /= train_df[period + '_total']
            test_df[new_column] /= test_df[period + '_total']
    
    
    
    
    ########################### Encode Str columns
    # For all such columns (probably not)
    # we already did frequency encoding (numeric feature)
    # so we will use astype('category') here
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
    
    ########################### Model Features 
    # # We can use set().difference() but the order matters
    # # Matters only for deterministic results
    # # In case of remove() we will not change order
    # # even if variable will be renamed
    # # please see this link to see how set is ordered
    # # https://stackoverflow.com/questions/12165200/order-of-unordered-python-sets
    rm_cols = [
        'TransactionID', 'TransactionDT',  # These columns are pure noise right now
        TARGET,  # Not target in features))
        'uid', 'uid2', 'uid3',  # Our new client uID -> very noisy data
        'bank_type',  # Victims bank could differ by time
        'DT', 'DT_M', 'DT_W', 'DT_D',  # Temporary Variables
        'DT_hour', 'DT_day_week', 'DT_day',
        'DT_D_total', 'DT_W_total', 'DT_M_total',
        'id_30', 'id_31', 'id_33',
    ]        
    
    ########################### Features elimination 
    from scipy.stats import ks_2samp
    features_check = []
    columns_to_check = set(list(train_df)).difference(base_columns + rm_cols)
    for i in columns_to_check:
        features_check.append(ks_2samp(test_df[i], train_df[i])[1])
    
    features_check = pd.Series(features_check, index=columns_to_check).sort_values() 
    features_discard = list(features_check[features_check == 0].index)
    print(features_discard)
    
    # We will reset this list for now (use local test drop),
    # Good droping will be in other kernels
    # with better checking
    features_discard = [] 
    
    # Final features list
    features_columns = [col for col in list(train_df) if col not in rm_cols + features_discard]
    
    return train_df, test_df, features_columns, TARGET, SEED;
    
'''    

def predictions():    
    if LOCAL_TEST:
        lgb_params['learning_rate'] = 0.01
        lgb_params['n_estimators'] = 20000
        lgb_params['early_stopping_rounds'] = 100
        test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params)
        print(metrics.roc_auc_score(test_predictions[TARGET], test_predictions['prediction']))
    else:
        lgb_params['learning_rate'] = 0.005
        lgb_params['n_estimators'] = 1800
        lgb_params['early_stopping_rounds'] = 100    
        test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, lgb_params, NFOLDS=10)
    
    ########################### Export
    if not LOCAL_TEST:
        test_predictions['isFraud'] = test_predictions['prediction']
        test_predictions[['TransactionID', 'isFraud']].to_csv('submission.csv', index=False)
           

def make_predictions(tr_df, tt_df, features_columns, TARGET, lgb_params, NFOLDS=2):
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    
    X, y = tr_df[features_columns], tr_df[TARGET]    
    P, P_y = tt_df[features_columns], tt_df[TARGET]  

    tt_df = tt_df[['TransactionID', TARGET]]    
    predictions = np.zeros(len(tt_df))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print(fold_)
        tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
        vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]
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
'''        