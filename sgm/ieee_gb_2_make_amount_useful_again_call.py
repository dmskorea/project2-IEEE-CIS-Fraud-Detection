import numpy as np
from sklearn.model_selection import train_test_split, KFold
import ieee_gb_2_make_amount_useful_again2 as go
import lightgbm as lgb

tr_df, tt_df, features_columns, TARGET, lgb_params, SEED = go.init()

NFOLDS = 2
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
X, y = tr_df[features_columns], tr_df[TARGET]    
P, P_y = tt_df[features_columns], tt_df[TARGET]  

tt_df = tt_df[['TransactionID', TARGET]]    

predictions = np.zeros(len(tt_df))

tr_df['C_enc'] = tr_df['C1_fq_enc'].astype(str) \
    +'_' + tr_df['C2_fq_enc'].astype(str) \
    +'_' + tr_df['C3_fq_enc'].astype(str) \
    +'_' + tr_df['C4_fq_enc'].astype(str) \
    +'_' + tr_df['C5_fq_enc'].astype(str) \
    +'_' + tr_df['C6_fq_enc'].astype(str) \
    +'_' + tr_df['C7_fq_enc'].astype(str) \
    +'_' + tr_df['C8_fq_enc'].astype(str) \
    +'_' + tr_df['C9_fq_enc'].astype(str) \
    +'_' + tr_df['C10_fq_enc'].astype(str) \
    +'_' + tr_df['C11_fq_enc'].astype(str) \
    +'_' + tr_df['C12_fq_enc'].astype(str) \
    +'_' + tr_df['C13_fq_enc'].astype(str) \
    +'_' + tr_df['C14_fq_enc'].astype(str)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(tr_df['C_enc'].astype(str).values)
le.transform(tr_df['C_enc'].astype(str).values)



# CV make model

########################### Model params
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
                    'subsample':0.7,
                    'n_estimators':800,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                    'early_stopping_rounds':100,
}

for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    print(fold_)
    tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
    vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]
    tr_data = lgb.Dataset(tr_x, label=tr_y)
    vl_data = lgb.Dataset(vl_x, label=vl_y)
    estimator = lgb.train(
        lgb_params,
        tr_data,
        valid_sets=[tr_data, vl_data],
        verbose_eval=200,
    )
    pp_p = estimator.predict(P)
    predictions += pp_p / NFOLDS
    feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(), X.columns)), columns=['Value', 'Feature'])
    print(feature_imp)
    del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
    gc.collect()
    
