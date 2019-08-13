| 알고리즘 | FE | 변수개수 | 전처리 | score(CV) | score(PL)|
|---------|----|---------|--------|-----------|----------|
| LGB | time of day, null, missing value | 350 | TimeSeiresSplit CV=3 | 0.91 | 0.92 |
| LGB | time of day, null, missing value | 350 | TimeSeiresSplit CV=5 | 0.91 | 0.92 |
| LGB | time of day, null, missing value | 350 | TimeSeiresSplit CV=7 | 0.91 | 0.92 |
| LGB | time of day, null, missing value | 350 | TimeSeiresSplit CV=10 | 0.91 | 0.92 |


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

- 0045 features CV 0.9172 LB 0.9465 
- 0050 features CV 0.9258 LB 0.9480
- 0408 features CV 0.9450 LB 0.9460
- 0558 features CV 0.9265 LB 0.9472
- 0650 features CV 0.9389 LB 0.9502
- 1800 features CV 0.9400 LB 0.9492
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
