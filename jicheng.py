# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 17:46:25 2017

@author: hp
"""

import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, auc 
from sklearn.metrics import roc_auc_score
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image 
from sklearn.neighbors import NearestNeighbors
import math
from scipy import stats
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

data_train=pd.read_csv('C:/Users/hp/Desktop/在家学习/信用评分/cs-training.csv')
data_test=pd.read_csv('C:/Users/hp/Desktop/在家学习/信用评分/cs-test.csv')
data=pd.concat([data_train,data_test],axis=0)

data.reset_index(inplace=True)
data.drop(['index'],axis=1,inplace=True)
data=data.reindex_axis(data_train.columns,axis=1)
data=data.iloc[:,1:]
data[data.columns[data.isnull().any()].tolist()].isnull().sum()
#na
depNew = []
med = data.NumberOfDependents.median()
for val in data.NumberOfDependents:
    if val.is_integer() == False:
        depNew.append(med)
    else:
        depNew.append(val)
data.NumberOfDependents = depNew

train = data[data.MonthlyIncome.isnull() == False]
test = data[data.MonthlyIncome.isnull() == True]
train.shape, test.shape
X_train = train.drop(['MonthlyIncome', 'SeriousDlqin2yrs'], axis=1)
y_train = train.MonthlyIncome
X_test = test.drop(['MonthlyIncome', 'SeriousDlqin2yrs'], axis=1)
lmMod = LinearRegression(fit_intercept=True, normalize=True).fit(X_train, y_train)
pred = lmMod.predict(X_test)
predNoZero = []
for val in pred:
    if val >= 0:
        predNoZero.append(val)
    else:
        predNoZero.append(0.)
test['MonthlyIncome']=predNoZero
data=pd.concat([train,test])
data1=data[data['SeriousDlqin2yrs'].notnull()]
data2=data[data['SeriousDlqin2yrs'].isnull()]
data=pd.concat([data1,data2])
#Outlier detection
def mad_based_outlier(points, thresh=3.5):
    median = np.median(data.DebtRatio, axis=0)
    diff = (data.DebtRatio - median)**2#按行求和
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh#方差/方差中位数大于3.5倍

def percentile_based_outlier(data, threshold=95):
    diff = (100 - threshold) / 2.0
    (minval, maxval) = np.percentile(data, [diff, 100 - diff])
    return ((data < minval) | (data > maxval))

def std_div(data, threshold=3):
    std = data.std()
    return data/std > threshold

std_div(data.DebtRatio)
def outlierVote(data):
    x = percentile_based_outlier(data)
    y = mad_based_outlier(data)
    z = std_div(data)
    temp = pd.concat([x, y, z],axis=1)
    final = []
    for i in range(len(temp)):
        if list(temp.iloc[i]).count(False) >= 2:
            final.append(False)
        else:
            final.append(True)
    return final 
def outlierRatio(data):
    functions = [percentile_based_outlier, mad_based_outlier, std_div, outlierVote]
    outlierDict = {}
    for func in functions:
        funcResult = func(data)
        count = 0
        for val in funcResult:
            if val == True:
                count += 1 
        outlierDict[str(func)[10:].split()[0]] = [count, '{:.2f}%'.format((float(count)/len(data))*100)]
    
    return outlierDict

#data.RevolvingUtilizationOfUnsecuredLines
revNew=[]
for val in data.RevolvingUtilizationOfUnsecuredLines:
    if val <= 2:
        revNew.append(val)
    else:
        revNew.append(2.)    
data.RevolvingUtilizationOfUnsecuredLines = revNew
#Age var
ageNew = []
for val in data.age:
    if val < 22:
        ageNew.append(22)
    elif val>85:
        ageNew.append(85)
    else:
        ageNew.append(val)
                
data.age = ageNew
data.age.value_counts()
def can(data):
    
    
#NumberOfTime3059DaysPastDueNotWorse var¶
data['NumberOfTime30-59DaysPastDueNotWorse'].value_counts()
New = []
med =data['NumberOfTime30-59DaysPastDueNotWorse'].median()
for val in data['NumberOfTime30-59DaysPastDueNotWorse']:
    if (val>20):
        New.append(med)
    else:
        New.append(val)
data['NumberOfTime30-59DaysPastDueNotWorse'] = New

#DebtRatio var¶
#sns.distplot(data.DebtRatio)
minUpperBound = min([val for (val, out) in zip(data.DebtRatio, mad_based_outlier(data.DebtRatio)) if out == True])
newDebtRatio = []
for val in data.DebtRatio:
    if val > minUpperBound:
        newDebtRatio.append(minUpperBound)
    else:
        newDebtRatio.append(val)
data.DebtRatio = newDebtRatio
#Monthly income var¶
minincom=min([val for (val,out) in zip(data.MonthlyIncome,outlierVote(data.MonthlyIncome)) if out==True])
data.MonthlyIncome.describe()
incomNew = []
med=data.MonthlyIncome.median()
for val in data.MonthlyIncome:
    if val==minincom:
        incomNew.append(med)
    else:
        incomNew.append(val)
data.MonthlyIncome = incomNew
#NumberOfTimes90DaysLate var¶
data['NumberOfTimes90DaysLate'].value_counts()
new=[]
med=data['NumberOfTimes90DaysLate'].median()
for x in data['NumberOfTimes90DaysLate']:
    if ((x==98)|(x==96)):
        new.append(med)
    else:
        new.append(x)
data['NumberOfTimes90DaysLate']=new

#NumberRealEstateLoansOrLines var¶
data.NumberRealEstateLoansOrLines.value_counts()
realNew = []
for val in data.NumberRealEstateLoansOrLines:
    if val > 17:
        realNew.append(17)
    else:
        realNew.append(val)
data.NumberRealEstateLoansOrLines = realNew
#NumberOfTime6089DaysPastDueNotWorse var¶
data['NumberOfTime60-89DaysPastDueNotWorse'].value_counts()
new=[]
med=data['NumberOfTime60-89DaysPastDueNotWorse'].median()
for x in data['NumberOfTime60-89DaysPastDueNotWorse']:
    if ((x==98)|(x==96)):
        new.append(med)
    else:
        new.append(x)
        
data['NumberOfTime60-89DaysPastDueNotWorse']=new
#NumberOfDependents var¶
depNew = []
for var in data.NumberOfDependents:
    if var > 10:
        depNew.append(10)
    else:
        depNew.append(var)
data.NumberOfDependents = depNew
data['fuzhaijine']=data['DebtRatio']*data['MonthlyIncome']
data['shifouweiyue']=(data['NumberOfTime60-89DaysPastDueNotWorse']+data['NumberOfTimes90DaysLate']+data['NumberOfTime30-59DaysPastDueNotWorse'])/(data['NumberOfTime60-89DaysPastDueNotWorse']+data['NumberOfTimes90DaysLate']+data['NumberOfTime30-59DaysPastDueNotWorse'])
new=[]
for x in data['shifouweiyue']:
    if x==1:
        new.append(1)
    else:
        new.append(0)
data['shifouweiyue']=new
data.to_csv('C:/Users/hp/Desktop/在家学习/信用评分/jicheng.csv')
train=data.iloc[:len(data_train),:]
test=data.iloc[len(data_train):,:]
train.to_csv('C:/Users/hp/Desktop/在家学习/信用评分/jichengtrain.csv')
test.to_csv('C:/Users/hp/Desktop/在家学习/信用评分/jichengtest.csv')
test=pd.read_csv('C:/Users/hp/Desktop/在家学习/信用评分/jichengtest.csv')
train=pd.read_csv('C:/Users/hp/Desktop/在家学习/信用评分/jichengtrain.csv')

class Smote:
    def __init__(self,samples,N=10,k=3):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
       # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))
    def over_sampling(self):
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print ('neighbors',neighbors)
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print nnarray
            self._populate(N,i,nnarray)
        return self.synthetic
    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1
a=np.array(train.iloc[:len(train),:][train['SeriousDlqin2yrs']==1])
s=Smote(a,N=500)
data_train_sampling=s.over_sampling()
data_train_sampling=pd.DataFrame(data_train_sampling,columns=list(train.columns))

#负样本随机采样
data_train_samplingz=(train.iloc[:len(train),:][train['SeriousDlqin2yrs']==0]).sample(n=60000)
train_sampling=pd.concat([data_train_sampling,data_train_samplingz,train.iloc[:len(train),:][train['SeriousDlqin2yrs']==1]])
train_sampling[['SeriousDlqin2yrs','age','NumberOfTime30-59DaysPastDueNotWorse','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate',
                    'NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents','fuzhaijine','shifouweiyue']] = train_sampling[['SeriousDlqin2yrs','age','NumberOfTime30-59DaysPastDueNotWorse','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate',
                    'NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents','fuzhaijine','shifouweiyue']].round()
train_sampling['SeriousDlqin2yrs'].value_counts()
train_sampling.to_csv('C:/Users/hp/Desktop/在家学习/信用评分/trainsampling.csv')
train_sampling=pd.read_csv('C:/Users/hp/Desktop/在家学习/信用评分/trainsampling.csv')
train_sampling.describe()
cut_data=pd.DataFrame()

#特征工程
from sklearn.cross_validation import ShuffleSplit
clf=RandomForestClassifier()
clf.fit(train.iloc[:,2:],train.iloc[:,1])
names=train.columns[2:]
print (sorted(zip(map(lambda x:round(x,2),clf.feature_importances_),names),reverse=True))

'''
[(0.16800000000000001, 'RevolvingUtilizationOfUnsecuredLines'),
 (0.14480000000000001, 'MonthlyIncome'), 
 (0.13769999999999999, 'fuzhaijine'), 
 (0.1167, 'DebtRatio'), 
 (0.1087, 'age'), 
 (0.080199999999999994, 'NumberOfOpenCreditLinesAndLoans'), 
 (0.078, 'NumberOfTimes90DaysLate'), 
 (0.0465, 'NumberOfTime30-59DaysPastDueNotWorse'), 
 (0.038399999999999997, 'NumberOfDependents'),
 (0.0281, 'NumberRealEstateLoansOrLines'),
 (0.028000000000000001, 'NumberOfTime60-89DaysPastDueNotWorse'),
 (0.024799999999999999, 'shifouweiyue')]
'''
pca=PCA(n_components=2)
xPca = pca.fit_transform(train_sampling[['NumberOfTime60-89DaysPastDueNotWorse','NumberOfTimes90DaysLate','NumberOfTime30-59DaysPastDueNotWorse']])

test_pca=pca.fit_transform(test[['NumberOfTime60-89DaysPastDueNotWorse','NumberOfTimes90DaysLate','NumberOfTime30-59DaysPastDueNotWorse']])
xPcaDataframe = pd.DataFrame(xPca, columns=['PC1', 'PC2'])
tPcaDataframe = pd.DataFrame(test_pca, columns=['PC1', 'PC2'])
ax = sns.lmplot(data = xPcaDataframe, x='PC2', y='PC1', size=10, aspect=20, fit_reg=False,
               scatter_kws={'alpha': 0.3})
fig = plt.gcf()
fig.set_size_inches(15, 7)
scores=[]
pd.crosstab(train_sampling['SeriousDlqin2yrs'],train_sampling['NumberOfOpenCreditLinesAndLoans'])
for x in range(train.shape[1]):
    score=cross_val_score(clf,train.iloc[:,x:x+1],train.iloc[:,0],cv=ShuffleSplit(len(train),n_iter=3,test_size=0.3),scoring='roc_auc')
    scores.append((score.mean(),names[x]))
print(sorted(scores))
#[(0.49683699033926371, 'DebtRatio'), ###
#(0.51332758343988605, 'fuzhaijine'), 
#(0.53928172926297557, 'MonthlyIncome'),
#(0.54950529010594629, 'NumberOfDependents'),
#(0.56014057306505072, 'NumberOfOpenCreditLinesAndLoans'), 
#(0.56496782647904853, 'NumberRealEstateLoansOrLines'),
#(0.61659215307434556, 'NumberOfTime60-89DaysPastDueNotWorse'),
#(0.62883883182587164, 'age'), 
#(0.63029325433521544, 'RevolvingUtilizationOfUnsecuredLines'),
#(0.65000493022847439, 'NumberOfTimes90DaysLate'), 
#(0.67970315886520993, 'NumberOfTime30-59DaysPastDueNotWorse'), ###
#(0.74259013314575828, 'shifouweiyue'), 
from collections import defaultdict
scores = defaultdict(list)
for train_idx, test_idx in ShuffleSplit(len(train),100,.3):
  X_train, X_test = train.iloc[train_idx,2:], train.iloc[test_idx,2:]
  Y_train, Y_test = train.iloc[train_idx,1], train.iloc[test_idx,1]
  r = clf.fit(X_train, Y_train)
  acc = roc_auc_score(Y_test, clf.predict(X_test))
  for i in range(train.iloc[:,2:].shape[1]):
    X_t = X_test.copy()
    np.random.shuffle(X_t.iloc[:,i].values)
    shuff_acc = roc_auc_score(Y_test, clf.predict(X_t))
    scores[names[i]].append((acc-shuff_acc)/acc)
print ("Features sorted by their score:")
print (sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True))
train_sampling1=train_sampling.drop(['NumberOfTime60-89DaysPastDueNotWorse','NumberOfTime30-59DaysPastDueNotWorse','NumberOfTimes90DaysLate'],axis=1)
train_sampling2=pd.concat([train_sampling1,xPcaDataframe],axis=1)
'''
[(0.064399999999999999, 'shifouweiyue'), 
(0.062600000000000003, 'NumberOfTimes90DaysLate'), 
(0.048399999999999999, 'RevolvingUtilizationOfUnsecuredLines'), 
(0.024199999999999999, 'NumberOfTime60-89DaysPastDueNotWorse'), 
(0.023099999999999999, 'NumberOfTime30-59DaysPastDueNotWorse'), 
(0.0088000000000000005, 'age'), 
(0.0063, 'DebtRatio'), 
(0.0047000000000000002, 'MonthlyIncome'), 
(0.002, 'fuzhaijine'), 
(0.0018, 'NumberOfOpenCreditLinesAndLoans'),
 (0.00089999999999999998, 'NumberOfDependents'), 
 (-0.00069999999999999999, 'NumberRealEstateLoansOrLines')]
'''

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
x_train,x_test,y_train,y_test=train_test_split(train_sampling2.iloc[:,[3,4,11,12,13]],train_sampling2.iloc[:,2],test_size=0.25)
#需要采样
adaMod = AdaBoostClassifier(n_estimators=100, learning_rate=1.0)

gbMod = GradientBoostingClassifier(n_estimators=300)
rfMod = RandomForestClassifier(n_estimators=30,verbose=1)

CVdict={}
cvscore=cross_val_score(adaMod,x_train.values,y_train.values,cv=3,scoring='roc_auc')
CVdict[str(adaMod)[0]] = [cvscore.mean(), cvscore.std()] #采样[0.88106261363214233, 0.00079029682512620999]   
print(CVdict)#不采样[0.86039025845151096, 0.0010744195256330088]

cvscore=cross_val_score(rfMod,x_train.values,y_train.values,cv=3,scoring='roc_auc')
CVdict[str(rfMod)[0]] = [cvscore.mean(), cvscore.std()]#[0.94394231207122858, 0.0004589902499013064]}
print(CVdict)#[0.82198446881107678, 0.0040150328800815519]

cvscore=cross_val_score(knnMod,x_train.values,y_train.values,cv=3,scoring='roc_auc')
CVdict[str(knnMod)[0]] = [cvscore.mean(), cvscore.std()]  #[0.68758744471827027, 0.0024873125469339085]
print(CVdict)#[0.54853594784706239, 0.0016976128960593463]

cvscore=cross_val_score(gbMod,x_train.values,y_train.values,cv=3,scoring='roc_auc')
CVdict[str(gbMod)[0]] = [cvscore.mean(), cvscore.std()] #[0.89272268072809802, 0.00086011558938636388]
print(CVdict)#[0.86490019116814432, 0.0015809048468261046]

from scipy.stats import randint
#Ada model
adaHyperParams = {'n_estimators':np.arange(100,410,100),'learning_rate':np.arange(0.1,1,0.1)}

gridSearchAda = GridSearchCV(estimator=adaMod, param_grid=adaHyperParams, cv=5,
                                   scoring='roc_auc', fit_params=None,verbose=2).fit(x_train.iloc[:,[0,1,2,3,4,5,6,8,9,10]], y_train)
gridSearchAda.best_params_, gridSearchAda.best_score_
({'n_estimators': 400},0.8821398685399355)

#GB model
gbHyperParams = {'n_estimators': np.arange(240,300,20)}
gridSearchGB = GridSearchCV(estimator=GradientBoostingClassifier(loss='exponential',max_depth=5), param_grid=gbHyperParams, cv=3,
                                   scoring='roc_auc',fit_params=None,verbose=2).fit(x_train, y_train)
gridSearchGB.best_params_, gridSearchGB.best_score_
({'loss': 'exponential', 'max_depth': 5, 'n_estimators': 280},
 0.89831676973203)

admod=AdaBoostClassifier(n_estimators=400,learning_rate=0.8)
gbmod=GradientBoostingClassifier(loss='exponential',max_depth=5,n_estimators=280)
submission1=admod.fit(x_train, y_train)
submission2=gbmod.fit(x_train,y_train)
y_pred1=admod.predict(x_test.iloc)
answer1=admod.predict_proba(x_test)[:,1]
roc_auc_score(y_test,answer1)#0.88453990346130285#
y_pred2=gbmod.predict(x_test)
answer2=gbmod.predict_proba(x_test)[:,1]
roc_auc_score(y_test,answer2)#0.90130628872656349


test_data_pca=test.drop(['NumberOfTime60-89DaysPastDueNotWorse','NumberOfTime30-59DaysPastDueNotWorse','NumberOfTimes90DaysLate'],axis=1)
test2=pd.concat([test_data_pca,tPcaDataframe],axis=1)

w1=admod.predict_proba(test2.iloc[:,[2,3,10,11,12]])[:,1]
w2=gbmod.predict_proba(test2.iloc[:,[2,3,10,11,12]])[:,1]
submission=(w1+w2)/2

result = pd.DataFrame({"Id": test.index+1, "Probability":w2})  
result.to_csv('C:/Users/hp/Desktop/在家学习/信用评分/test_lm.csv', index=False)#0.847352
0.847518

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
#共线性和pca
x_train,x_test,y_train,y_test=train_test_split(train_sampling.iloc[:,3:],train_sampling.iloc[:,2],test_size=0.25)
x_train, x_train_lr, y_train, y_train_lr = train_test_split(x_train,
                                                            y_train,
                                                            test_size=0.5)
rf = RandomForestClassifier(max_depth=3, n_estimators=11)
rf_enc = OneHotEncoder()
rf_lm = LogisticRegression()
rf.fit(x_train, y_train)
rf_enc.fit(rf.apply(x_train))
rf_lm.fit(rf_enc.transform(rf.apply(x_train_lr)), y_train_lr)
y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(x_test)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)
mm=rf.apply(x_train)
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)

test_lm=rf_lm.predict_proba(rf_enc.transform(rf.apply(test.iloc[:,2:])))[:, 1]
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X_new = SelectKBest(chi2,k=9).fit_transform(train_sampling.iloc[:,[3,4,5,6,7,8,9,10,11,12,13,14]], train_sampling['SeriousDlqin2yrs'])
X_new.shape
from scipy.stats import pearsonr
pearsonr(train_sampling.iloc[:,5], train_sampling.iloc[:,9])#相关系数和p值
x=pd.crosstab(train_sampling.iloc[:,5], train_sampling.iloc[:,9])
corrmat=train_sampling.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14]].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)#

data.describe()






