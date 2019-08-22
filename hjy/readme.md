TO-DO-LIST
- another FE : https://www.kaggle.com/nroman/eda-for-cis-fraud-detection
- train_df=nan2mean(train_df)
- jupyterlab : https://hub.gke.mybinder.org/user/jupyterlab-jupyterlab-demo-sudfcgxr/lab
- weight average : https://www.kaggle.com/paulorzp/gmean-of-light-gbm-models-lb-0-9476
- base_columns = list(train) + list(train_identity) 지우기 
- df['Float_a'] = pd.cut(x=df['Float_a'],bins=10, labels=[f'bin_{i}' for i in range(bins)])
- https://www.kaggle.com/yasagure/places-after-the-decimal-point-tell-us-a-lot
- LGB tune : https://www.kaggle.com/nicapotato/gpyopt-hyperparameter-optimisation-gpu-lgbm
- PCA + V or only PCA or only V 
- NOT WORKING!! -> averaging='rank', # rank,usual
- GPU!!!!
- GPU XGB with tune : https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt
- feature selection : https://www.kaggle.com/ogrellier/feature-selection-with-null-importances
- feature extraction
- CatBoost, XGBoost, NN
- train_model_classification : https://www.kaggle.com/artgor/eda-and-models
- GPU LGB : https://www.kaggle.com/kirankunapuli/ieee-fraud-lightgbm-with-gpu
- bayesian opt : https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt#2.-Bayesian-Optimisation
- FE : https://www.kaggle.com/robikscube/ieee-fraud-detection-first-look-and-eda
- FE : https://www.kaggle.com/jesucristo/fraud-complete-eda
- FE : https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt
- FE2 : https://www.kaggle.com/jacobgreen4477/extensive-eda-and-modeling-xgb-hyperopt/edit
- FE2 : https://www.kaggle.com/jolly2136/fe-xgb/output
- FE3 : histogram feature, https://www.kaggle.com/ogrellier/using-histogram-as-feature-1-43

***

benchmark 

- 0045 features CV 0.9172 LB 0.9465
- 0050 features CV 0.9258 LB 0.9480
- 0408 features CV 0.9450 LB 0.9460
- 0558 features CV 0.9265 LB 0.9472
- 0650 features CV 0.9389 LB 0.9502
- 1800 features CV 0.9400 LB 0.9492

my leaderboard

> FE : time of day(3), missing value(1),  lastest_browser(1), emaildomain(2), card(6), addr(2), aggregate(16), decimal(1)

| score(CV) | score(PL)| trial and error | ML | # of features | 
|-----------|----------|-----------------|----|--------------|
| 0.9522 | 0.9306 | TimeSeiresSplit CV=20 | LGB | 396 | 
| 0.9382 | 0.9292 | TimeSeiresSplit CV=10 | LGB | 396 | 
| 0.9044 | 0.9237 | TimeSeiresSplit CV=07 | LGB | 396 | 
| 0.9330 | 0.9270 | TimeSeiresSplit CV=05 | LGB | 396 | 
| 0.9260 | 0.9215 | TimeSeiresSplit CV=03 | LGB | 396 | 

***

- Q: how many features for feature aggregation?
- Q: in terms of data split, hold-out, time series split, or random split?
- Q: any interaction features?
- Q: the optimal k for cv? 
- Q: the optimal params for LGB?
- Q: label encoding or one-hot encoding? ... that tree based models work well with Label Encodings even if there is no ordinal relationship. (check later)

***

첫번째 방식이 더 좋음 

```
for col in list(train):
    if train[col].dtype=='O':
        print(col)
        train[col] = train[col].fillna('unseen_before_label')
        test[col]  = test[col].fillna('unseen_before_label')
        
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        
        le = LabelEncoder()
        le.fit(list(train[col])+list(test[col]))
        train[col] = le.transform(train[col])
        test[col]  = le.transform(test[col])
        
        train[col] = train[col].astype('category')
        test[col] = test[col].astype('category')

le = LabelEncoder()
for col in train.select_dtypes(include=['object', 'category']).columns:
    le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
    train[col] = le.transform(list(train[col].astype(str).values))
    test[col] = le.transform(list(test[col].astype(str).values))
```
***

histogram feature

```
%%time
def to_hist_func(row):
    return np.bincount(row, minlength=30)

features = [f for f in data.columns if f not in ['target', 'ID']]

hist_data = np.apply_along_axis(
    func1d=to_hist_func, 
    axis=1, 
    arr=(np.log1p(data[features])).astype(int)) 
    
%%time
hist_test = np.apply_along_axis(
    func1d=to_hist_func, 
    axis=1, 
    arr=(np.log1p(test[features])).astype(int))
```

***

label encoding

```
for col in train.columns:
    if train[col].dtype == 'object':
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))
```
***

model

```
def fit_predict(X, y, X_test, folds, model_params, training_params):
    in_fold, out_of_fold, test_preds = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X_test))
    for fold_nr, (trn_idx, val_idx) in enumerate(folds.split(X.values, y.values)):
        print("Fold {}".format(fold_nr))

        X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

        trn_data = lgb.Dataset(X_train, y_train)
        val_data = lgb.Dataset(X_valid, y_valid)
        
        # add live monitoring of lightgbm learning curves
        monitor = neptune_monitor(prefix='fold{}_'.format(fold_nr))
        clf = lgb.train(model_params, trn_data, 
                        training_params['num_boosting_rounds'], 
                        valid_sets = [trn_data, val_data], 
                        early_stopping_rounds = training_params['early_stopping_rounds'],
                        callbacks=[monitor])
        in_fold[trn_idx] = clf.predict(X.iloc[trn_idx], num_iteration=clf.best_iteration)
        out_of_fold[val_idx] = clf.predict(X.iloc[val_idx], num_iteration=clf.best_iteration)
        test_preds += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits
    return in_fold, out_of_fold, test_preds    
```

***

indicator feature: protonmail.com

```
train['P_isproton']=(train['P_emaildomain']=='protonmail.com')
train['R_isproton']=(train['R_emaildomain']=='protonmail.com')
test['P_isproton']=(test['P_emaildomain']=='protonmail.com')
test['R_isproton']=(test['R_emaildomain']=='protonmail.com')
```

***

count encoding

```
char_features <- tem[,colnames(tem) %in% 
                     c("ProductCD","card1","card2","card3","card4","card5","card6","addr1","addr2","P_emaildomain",
                   "R_emaildomain","M1","M2","M3","M4","M5","M6","M7","M8","M9","DeviceType","DeviceInfo","id_12",
                   "id_13","id_14","id_15","id_16","id_17","id_18","id_19","id_20","id_21","id_22","id_23","id_24",
                   "id_25","id_26","id_27","id_28","id_29","id_30","id_31","id_32","id_33","id_34","id_35","id_36",
                   "id_37","id_38")]

fe_part1 <- data.frame(0)
for(a in colnames(char_features) ){
  tem1 <- char_features %>% group_by(.dots = a) %>% mutate(count = length(card4)) %>% ungroup() %>% select(count)
  colnames(tem1) <- paste(a,"__count_encoding",sep="")
  fe_part1 <- data.frame(fe_part1,tem1)
}

fe_part1 <- fe_part1[,-1]
```

***

변수 목록

- isFraud : 타겟변수
- TransactionDT : 거래시간
- TransactionAmt : 거래금액
- ProductCD : 상품대분류(W,C,R,H,S)
- card1 ~ card6 : payment card information, such as card type, card category, issue bank, country, etc.
- addr1 : 도시
- addr2 : 나라 
- dist1 : distance
- dist2 : distance
- P_emaildomain : Purchaser 이메일 주소
- R_emaildomain : Recipient 이메일 주소 
- C1 ~ C14 : counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
- D1 ~ D15 : timedelta, such as days between previous transaction, etc.
- M1 ~ M9 : match, such as names on card and address, etc.
- V1 ~ V339 : Vesta engineered rich features, including ranking, counting, and other entity relations.
- id_23 : proxy info
- id_30 : device info ex.iOS 11.1.2, Android 7.0, Mac OS X 10_12_6
- id_31 : browser info 
- id_32 : 화면사이즈?
- id_34 : match_status?
- id_01 ~ id_38 : identity information – network connection information (IP, ISP, Proxy, etc) and digital signature (UA/browser/os/version, etc) associated with transactions. 
- DeviceType : mobile, desktop
- DeviceInfo : SAMSUNG,Windows,MacOS, ...

***

https://www.kaggle.com/c/ieee-fraud-detection/discussion/102883#latest-593864

AutoLGB, https://www.kaggle.com/jeongyoonlee/kaggler-s-autolgb

***

https://www.kaggle.com/c/ieee-fraud-detection/discussion/100333#latest-583866

faster read dataset!!!

```
import datatable as dt

folder_path = '../input/'
train_identity = dt.fread(f'{folder_path}train_identity.csv')
test_identity = dt.fread(f'{folder_path}test_identity.csv')
train_transaction = dt.fread(f'{folder_path}train_transaction.csv')
test_transaction = dt.fread(f'{folder_path}test_transaction.csv')

train_identity.key = 'TransactionID'
test_identity.key = 'TransactionID'
train = train_transaction[:, :, dt.join(train_identity)]
test = test_transaction[:, :, dt.join(test_identity)]

train.to_csv("train.csv")
test.to_csv("test.csv")
```

https://www.kaggle.com/c/ieee-fraud-detection/discussion/100400#latest-594758
- the hour of the day vs the fraction of fraudulent transactions 
- the high fraction of fraudulent transactions occur when there is a low number of transactions per hour
- The hour of the day is D9
- train['D9'] = (X_train['TransactionDT']%(3600*24)/3600//1)/24.0
- But the feature D9 has too many null values (nearly 87%) so I guess we should still extract hour of the day from TransactionDT
-  you were absolutely right about useless of year and month . But minute , second are really useful. I am now trying some time based features.
- Hour and TransactionPerHour features worked for me.
- HourTransactionVolume, DayTransactionVolume

```
START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
X_train['TransactionDT'] = X_train['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
#X_train['year'] = X_train['TransactionDT'].dt.year
#X_train['month'] = X_train['TransactionDT'].dt.month
X_train['dow'] = X_train['TransactionDT'].dt.dayofweek
X_train['hour'] = X_train['TransactionDT'].dt.hour
X_train['day'] = X_train['TransactionDT'].dt.day
```

***

https://www.kaggle.com/c/ieee-fraud-detection/discussion/100071#latest-589485

- the TransactionDT column is measured in seconds, starting at December 1.
- There are peaks in the number of transactions at around 25 days and around 390 days (exactly 365 days later), which could be caused by increased sales around the christmas period. The test data ends exactly at the 31st of December, when choosing December 1 as starting date.

***

https://www.kaggle.com/c/ieee-fraud-detection/discussion/100071#latest-589485

- Pemaildomain and Remaildomain 
- P is Purchaser and R is Recipient
- This might not be a good decision due to some of these binned values having drastically different rates of fraud. For example with the Microsoft bin, msn has a fraud rate around 2.2% while outlook is around 9.5%.

```
yahoo / ymail / frontier / rocketmail -> Yahoo
hotmail / outlook / live / msn -> Microsoft
icloud / mac / me -> Appe
prodigy / att / sbcglobal-> AT&T
centurylink / embarqmail / q -> Centurylink
aim / aol -> AOL
twc / charter -> Spectrum
```

***

https://www.kaggle.com/c/ieee-fraud-detection/discussion/101040#latest-590616

- In my first figure the first 20% of the test set are located to the left of the solid black vertical line. As you can see, this is before the observed change in missing values distribution. This could mean that our public LB score will not correlate well with the private LB. This may well be intentional by the competition hosts to see how well our models generalise.
- What does this mean? It may mean the winning solution will have to deal with the missing values in a novel way, but as far as I know there is no way of validating any way of dealing with the missing data.
- One conclusion from analysis - Null values increase with time in test set and decrease with time in train set.

***

https://www.kaggle.com/c/ieee-fraud-detection/discussion/102940#latest-597149

- TimeSeriesSplit is highly influenced by the training set size (I tested with 10 folds different baseline models (xgboost,lighgbm,catboost)and there seems to be a fairly linear relationship between validation AUC and train size. 

***

https://www.kaggle.com/c/ieee-fraud-detection/discussion/100778#latest-589021

- The Pemaildomain tells a lot about the isFraud. 95% transaction made by protonmail.com, isFraud=1 Also, 40% transaction made by protonmail.com in Remaildomain, isFraud=1

***

https://www.kaggle.com/c/ieee-fraud-detection/discussion/103439#latest-596759

- Use a good open source single mode：
- https://www.kaggle.com/timon88/lgbm-baseline-small-fe-no-blend
- https://www.kaggle.com/nroman/lgb-single-model-lb-0-9419
- https://www.kaggle.com/artgor/eda-and-models
- Integrate their useful information
- Increase feature（counts,aggregations）

***

https://datascience-enthusiast.com/R/pandas_datatable.html

***
NN

```
# model
def get_model(embedding_cols, numerical_cols):
embedding_inp = [Input(shape=(1,),dtype='int32') for x in range(len(embedding_cols))]
ex = [Embedding(output_dim=5, input_dim=1, embeddings_initializer='RandomUniform',
               input_length=1)(x) for x in embedding_inp]
ex = Concatenate(axis=1)(ex)
ex_1 = Permute((2,1))(ex)
max_pool = GlobalMaxPooling1D()(ex_1)
avg_pool = GlobalAveragePooling1D()(ex_1)
ex = Flatten()(ex)

# TimeSeries
# previous
pre_ts_inp = Input(shape=(8,37))
pre_ts_output = Bidirectional(CuDNNGRU(96,return_sequences=True))(pre_ts_inp)
pre_ts_output = CuDNNGRU(128)(pre_ts_output)

# bureau balance
bb_ts_inp = Input(shape=(36,2))
#bb_emb_inp = Lambda(lambda x: x[:, :,1])(bb_ts_inp)

# pos
pos_ts_inp = Input(shape=(36,7))
ins_ts_inp = Input(shape=(36,7))
bb_month_inp = Reshape((36,1))(Lambda(lambda x: x[:, :,0])(bb_ts_inp))
cr_ts_inp = Input(shape=(36,33))

x_ts_inp = concatenate([pos_ts_inp, ins_ts_inp, bb_month_inp, cr_ts_inp],axis=2)
x_ts_output = Bidirectional(CuDNNGRU(64,return_sequences=True))(x_ts_inp)
x_ts_output = CuDNNGRU(64)(x_ts_output)

# install
# bureau
bu_ts_inp = Input(shape=(12,31))
bu_ts_output = Bidirectional(CuDNNGRU(64,return_sequences=True))(bu_ts_inp)
bu_ts_output = CuDNNGRU(64)(bu_ts_output)

## Numerical inputs
numerical_inp = Input(shape=(len(numerical_cols),), dtype='float32')
x = BatchNormalization()(numerical_inp)

x = concatenate([ex, max_pool, avg_pool, x,
                 pre_ts_output,  bu_ts_output, x_ts_output])
x = Dropout(0.2)(x)

x = Dense(512)(x) #,kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001)
x = ELU()(x)
x = Dropout(0.2)(x)

x = Dense(128)(x) #,kernel_regularizer=l2(0.0001)
x = ELU()(x)
x = Dropout(0.2)(x)
x = Dense(32)(x)
x = ELU()(x)

out = Dense(1, activation='sigmoid')(x)
model = Model(embedding_inp+[numerical_inp, 
                             pre_ts_inp, pos_ts_inp, ins_ts_inp, bu_ts_inp, bb_ts_inp, cr_ts_inp], 
              out)

optimizer = optimizers.Adam(lr=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[rocauc]) #
return model
```
