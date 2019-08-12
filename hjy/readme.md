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

https://www.kaggle.com/c/ieee-fraud-detection/discussion/100071#latest-589485

- the TransactionDT column is measured in seconds, starting at December 1.
- There are peaks in the number of transactions at around 25 days and around 390 days (exactly 365 days later), which could be caused by increased sales around the christmas period. The test data ends exactly at the 31st of December, when choosing December 1 as starting date.
