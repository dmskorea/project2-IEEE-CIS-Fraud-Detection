<hr />
DMS.Korea LeaderBoard
<hr />

| date| name | 알고리즘 | 변수개수 | CV | LB | 비고 |
|-----|------|---------|---------|----|-----|-----|
| 2019-09-04 | 김윤영 | LGB | 419 | 0.9410 | 0.9509 | DT_group_kfold |
| 2019-09-02 | 김윤영 | LGB | 418 | 0.9399 | 0.9508 | + type_per_period |
| 2019-08-27 | 김윤영 | LGB | 410 | 0.9393 | 0.9504 | + (null sum + browser check) & drop some V |
| 2019-09-05 | 현종열 | LGB | - | 0.9408 | 0.9503 | - |
| 2019-08-24 | 현종열 | LGB | 506 | 0.9508 | 0.9489 | simple fe + lgb4 + (uid3+id_target_mean) v3 |
| 2019-09-03 | 석기명 | LGB | 492 | 0.9340 | 0.9461 | 10 folds + lgb |
| 2019-08-20 | 송현정 | LGB | 556 | 0.9372 | 0.9457 | simple fe + 5folds + lgb |
| 2019-08-22 | 하진성 | LGB | - | - | 0.9378 | no fe + 5 folds + lgb |
| 2019-08-24 | 이윤선 | LGB | 329 | - | 0.9352 |   |
| 2019-08-23 | 박민규 | LGB | 430 | - | 0.9245 |   |
| 2019-08-24 | 김민주 | XGB | 413 | 0.8669 | 0.8551 | no fe + 5folds + xgb |

<hr />
2019-08-24 공지사항
<hr />

안녕하세요^^ 2019-08-24 공지사항 공유드립니다. 

1. 김윤영님이 오늘부터 단머스 (부)운영진으로 활동하시게 되었습니다. 

2. 스터디 시간에 말씀드린데로 팀을 재구성했습니다. 
- A팀 팀장은 현종열, B팀 팀장은 김윤영입니다.  
- 팀장은 팀원들이 작업할 수 있는 공통 템플릿을 제공할 예정입니다. 공통 템플릿은 현종열님과 김윤영님이 작업한 내용을 종합해서 곧 공유드리겠습니다.
- 2019-08-24 기준 A팀 최고성적은 CV:0.9509 LB:0.9486 B팀 최고성적은 CV:0.9394 LB:0.9500 입니다. 

![TEAM](https://raw.githubusercontent.com/dmskorea/project2-IEEE-CIS-Fraud-Detection/master/hjy/png/DMS_TEAM_20190824.png)

3. 스터디룰을 다시 한번 공지드리면 아래의 경우에 해당되는 경우 팀에서 제외될 수 있습니다. 
- 스터디 참여가 저조한 경우
- 연속해서 스터디에 불참하는 경우 

4. 보다 효율적인 스터디를 위해서 스터디 운영 방식을 변경합니다.
- 새로운 스터디 방식 : 공통 템플릿이 제공되면, 그 템플릿을 기초로 아래의 영역 중 한가지 이상을 추가로 작업하여 CV,LB 점수가 향상되면 그 내용을 정리해서 스터디 시간에 발표 
- 아래 URL를 각 영역별 참고 자료입니다. 

1) 변수 생성 (feature aggregation, target mean encoding, weight of evidence, indicator features, interaction features ...)
- https://www.kaggle.com/robikscube/ieee-fraud-detection-first-look-and-eda
- https://www.kaggle.com/jesucristo/fraud-complete-eda
- https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt
- https://www.kaggle.com/nroman/eda-for-cis-fraud-detection
- https://www.kaggle.com/jolly2136/fe-xgb/output
- https://www.kaggle.com/ogrellier/using-histogram-as-feature-1-43
- https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering

2) 변수 선택  
- https://www.kaggle.com/ogrellier/feature-selection-with-null-importances
- https://www.kaggle.com/willkoehrsen/introduction-to-feature-selection#Feature-Selection-through-Feature-Importances

3) 모델 튜닝 (lgb, xgb, catboot, nn)  
- https://www.kaggle.com/nicapotato/gpyopt-hyperparameter-optimisation-gpu-lgbm
- https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt#2.-Bayesian-Optimisation
- https://www.kaggle.com/ogrellier/home-credit-hyperopt-optimization
- https://www.kaggle.com/willkoehrsen/automated-model-tuning

5. 공통템플릿을 1차 버전은 현재 notebook에서 LB점수가 가장 높은 notebook으로 사용합니다. 2차 버전은 추후에 공유드리겠습니다. 

- Python (LB: 0.9468)
https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again

- R (LB: 0.9452)
https://www.kaggle.com/duykhanh99/lightgbm-fe-with-r

<hr />
2019-08-23 공지사항
<hr />

- 조승우님 탈퇴 

<hr />
2019-08-12 단머스2기 이메일
<hr />

- 현종열 : jacobgreen4477@gmail.com
- 장주호 : jh.zhiang@gmail.com
- 최지원 : jionysos.c@gmail.com
- 김민주 : democracy7770@gmail.com
- 하진성 : jinsung.ha929@gmail.com
- 석기명 : zeroshift01@naver.com
- 송현정 : halohj12@gmail.com
- 김윤영 : kyy0810@naver.com
- 전승유 : b00307795@essec.edu
- 양현일 : hi9818@gmail.com
- 이윤선 : yoonseon555@gmail.com
- 박민규 : ruserive@gmail.com

<hr />
2019-08-11 커널 주요내용
<hr />

1. train, test 데이터셋은 not overlap!. train은 과거 데이터 test는 미래 데이터로 data split에서 random 으로 하지말고 시간(TransactionDT)기준으로 진행

2. train 데이터셋과 test 데이터셋은 샘플 사이즈는 동일.  (훈련시 샘플링 비추)

3. 몇몇 변수는 이미 normalized 되어 있습니다. 변수 전부를 normalized 해서 사용x

4. 몇몇 categorical 변수는 "found"라는 상태값 존재

5. client device 변수, some of info could be for old devices and may be absent from test data.
(device에 대한 참고 의견 : https://www.kaggle.com/c/ieee-fraud-detection/discussion/103565#latest-596633)

6. TransactionAmt를 log transform하면 skewed distribution이 조정 됨

7. card1 ~ card6는 categorical feature (card1,2,3,5의 변수값은 numeric이여도 이변수들은 category 입니다.)

8. kernel에 reduce_mem_usage 함수가 올라와 있습니다. 
https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt

9. TOP VI변수 :  TransactionID, card1, TransactionAmt, addr1, card2, dist1
(TransactionID에 대한 참고 의견 : https://www.kaggle.com/c/ieee-fraud-detection/discussion/103690#latest-596681)

10. LGB + Best Bayes params : 
{'bagging_fraction': 0.8999999999997461,
 'feature_fraction': 0.8999999999999121,
 'max_depth': 50.0,
 'min_child_weight': 0.0029805017044362268,
 'min_data_in_leaf': 20.0,
 'num_leaves': 381.85354295079446,
 'reg_alpha': 1.0,
 'reg_lambda': 2.0}

<hr />
2019-08-08 공지사항
<hr />

팀을 재구성했습니다. 팀별 최대 인원은 5명입니다. 

[A팀] (DMS.Korea.A)
- A-01. 현종열(hjy) : DB손해보험, 디지털혁신파트 
- A-02. 양현일(yhi) : 우리은행, 데이터분석
- A-03. 이윤선(lys) : 오렌지라이프, 데이터분석
- A-04. 박민규(pmk) : AIA생명, 데이터분석 

[B팀] (DMS.Korea.B)
- B-01. 김윤영(kyy) : 단머스1기 멤버
- B-02. 송현정(shj) : 단머스1기 멤버
- B-03. 장주호(jjh) : 연세대 의예과 본과 1학년, 주요언어 Python/R, 사용패키지 사이킷런, 방학기간이라 성실하게 follow up
- B-04. 석기명(sgm) : 주요언어Java, R/Python 사용가능, kmean/DBSCAN 사용 경험, CNN/RNN 배우고 싶음 

[C팀] (DMS.Korea.C)
- C-01. 전승유(jsy) : DB손해보험, 마키팅전략파트
- C-02. 하진성(hjs) : Python, Java, sklearn, 영국 유학생, https://github.com/jha929
- C-03. 최지원(cjw) : 은행 영업점에서 근무, 주요언어R, 은행들어오기 전 스타트업에서 데이터분석업무를 함 
- C-04. 김민주(kms) : 빅데이터 플랫폼 개발회사에서 데이터 분석 업무 담당, 주요언어 Python/R, 현재업무는 word2vec사용해서 키워드 분석


<hr />
2019-07-26 공지사항
<hr />

[대회 소개]
- Kaggle, IEEE-CIS Fraud Detection
- URL : https://www.kaggle.com/c/ieee-fraud-detection/overview

[대회 주요 일정]
- 최종 submit : 2019년 10월 1일 
- 외부 데이터 등록 : 2019년 9월 24일 
- 팀merge 데드라인 : 2019년 9월 24일 
- 일 최대 submit 횟수 : 5번 

[팀 정보]
- 팀별 최대 인원수 : 5명 
- 팀1(DMS.Korea1) : 김윤영, 송현정, 최지원, 김민수 
- 팀2(DMS.Korea2) : 현종열, 양현일, 이윤선, 박민규, 장주호
- 팀3(DMS.Korea3) : 전승유, 김민주, 석기명, 조승우

[스터디 일정] (*격주 토요일 오전 10시 시작)
- 1회차(2019-08-10) : 대회소개 및 팀 나누기
- 2회차(2019-08-24) : 데이터 탐색 공유 
- 3회차(2019-09-07) : 싱글 모델 공유  
- 4회차(2019-09-21) : 모델 앙상블 공유 
- 5회차(2019-10-05) : 팀별 마지막 공유 및 뒷풀이(기념사진촬영)
 
<hr />
2019-07-24 공지사항
<hr />

[단머스2기 지원자 현황]

01. 현종열(o) : 단머스 모임장, DB손해보험, 디지털혁신파트 
02. 조승우(o) : 직장인, 주요언어Java, Python학습 중, 주요업무SM, SM 8년차
03. 장주호(o) : 연세대 의예과 본과 1학년, 주요언어 Python/R, 사용패키지 사이킷런, 방학기간이라 성실하게 follow up
04. 최지원(o) : 은행 영업점에서 근무, 주요언어R, 은행들어오기 전 스타트업에서 데이터분석업무를 함 
05. 김민주(o) : 빅데이터 플랫폼 개발회사에서 데이터 분석 업무 담당, 주요언어 Python/R, 현재업무는 word2vec사용해서 키워드 분석
06. 전승유(o) : DB손해보험, 마키팅전략파트
07. 양현일(o) : 우리은행, 데이터분석
08. 이윤선(o) : 오렌지생명, 데이터분석
09. 박민규(o) : AIA생명, 데이터분석 
10. 하진성(o) : Python, Java, sklearn, 영국 유학생, https://github.com/jha929
11. 석기명(o) : 주요언어Java, R/Python 사용가능, kmean/DBSCAN 사용 경험, CNN/RNN 배우고 싶음 
12. 김민수(x) : 소프트웨어개발자, 주요언어c, Java/Python 등 다양하게 사용가능, Tensorflow기초, DNN/CNN 튜토리얼 학습함


<hr />
2019-07-15 공지사항
<hr />

[스터디원 모집공고]
- 홍보 일정 : 2019년 7월 19일 ~ 2019년 7월 31일
- 홍보 페이지 : OKKY, 아네모 
- 모집인원 : 0 명 

[스터디 장소]
- 선릉역 DB금융센터 지하 2층 (예정)
- 문의 : jacobgreen4477@gmail.com
