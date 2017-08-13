# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 09:08:26 2017

@author: hp
"""
'''
SeriousDlqin2yrsY/N超过90天或更糟的逾期拖欠
RevolvingUtilizationOfUnsecuredLines
无担保放款的循环利用：除了不动产和像车贷那样除以信用额度总和的无分期付款债务的信用卡和个人信用额度总额
NumberOfTime30-59DaysPastDueNotWorse35-59天逾期但不糟糕次数
DebtRatio负债比率
NumberOfOpenCreditLinesAndLoans
开放式信贷和贷款数量，开放式贷款（分期付款如汽车贷款或抵押贷款）和信贷（如信用卡）的数量
NumberOfTimes90DaysLate
90天逾期次数：借款者有90天或更高逾期的次数
NumberRealEstateLoansOrLines
不动产贷款或额度数量：抵押贷款和不动产放款包括房屋净值信贷额度
NumberOfTime60-89DaysPastDueNotWorse
60-89天逾期但不糟糕次数：借款人在在过去两年内有60-89天逾期还款但不糟糕的次数
NumberOfDependents
家属数量：不包括本人在内的家属数量
'''
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegressionCV
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
from sklearn.cross_validation import train_test_split

data_train=pd.read_csv('...cs-training.csv')
data_test=pd.read_csv('...cs-test.csv')
data_train=data_train.ix[:,1:]
data_test=data_test.ix[:,1:]
data=pd.concat([data_train,data_test])
data.reset_index(inplace=True)
data.drop('index',axis=1,inplace=True)
data=data.reindex_axis(data_train.columns,axis=1)

#缺失值填充
data_test[data_test.columns[data_test.isnull().any()].tolist()].isnull().sum()
#monthlyincome

data_nul=data.drop(['SeriousDlqin2yrs','NumberOfDependents'],axis=1)
train=data_nul[(data_nul['MonthlyIncome'].notnull())]
test=data_nul[(data_nul['MonthlyIncome'].isnull())]
train_x=train.drop(['MonthlyIncome'],axis=1)
train_y=train['MonthlyIncome']
test_x=test.drop(['MonthlyIncome'],axis=1)
gbMod = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=30, subsample=1.0, min_samples_split=2,
                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None,
                                  random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                                  warm_start=False)
gbMod.fit(train_x,train_y)
m=gbMod.predict(test_x)
gbMod.feature_importances_
new=[]
for x in m :
    if x<=0:
        new.append(0)
    else:
        new.append(x)
data.loc[(data['MonthlyIncome'].isnull()),'MonthlyIncome']=new
data_nul=data.drop(['SeriousDlqin2yrs'],axis=1)
train=data_nul[(data_nul['NumberOfDependents'].notnull())]
test=data_nul[(data_nul['NumberOfDependents'].isnull())]
train_x=train.drop(['NumberOfDependents'],axis=1)
train_y=train['NumberOfDependents']
test_x=test.drop(['NumberOfDependents'],axis=1)
gbMod.fit(train_x,train_y)
m=gbMod.predict(test_x)
new=[]
for x in m :
    if x<=0:
        new.append(0)
    else:
        new.append(x)
data.loc[(data['NumberOfDependents'].isnull()),'NumberOfDependents']=new
data['fuzhaijine']=data['DebtRatio']*data['MonthlyIncome']
data['shifouweiyue']=(data['NumberOfTime60-89DaysPastDueNotWorse']+data['NumberOfTimes90DaysLate']+data['NumberOfTime30-59DaysPastDueNotWorse'])/(data['NumberOfTime60-89DaysPastDueNotWorse']+data['NumberOfTimes90DaysLate']+data['NumberOfTime30-59DaysPastDueNotWorse'])
new=[]
for x in data['shifouweiyue']:
    if x==1:
        new.append(1)
    else:
        new.append(0)
data['shifouweiyue']=new
data_test=data[data['SeriousDlqin2yrs'].isnull()]
#采样
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
a=np.array(data.iloc[:len(data_train),:][data['SeriousDlqin2yrs']==1])
s=Smote(a,N=500)
data_train_sampling=s.over_sampling()
data_train_sampling=pd.DataFrame(data_train_sampling,columns=list(data.columns))
#负样本随机采样
data_train_samplingz=(data.iloc[:len(data_train),:][data_train['SeriousDlqin2yrs']==0]).sample(n=60000)
train_data_sampling=pd.concat([data_train_sampling,data_train_samplingz,data.iloc[:len(data_train),:][data_train['SeriousDlqin2yrs']==1]])
train_data_sampling[['SeriousDlqin2yrs','age','NumberOfTime30-59DaysPastDueNotWorse','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate',
                    'NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']] = train_data_sampling[['SeriousDlqin2yrs','age','NumberOfTime30-59DaysPastDueNotWorse','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate',
                    'NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']].round()
train_data_sampling['SeriousDlqin2yrs'].value_counts()
train_data_sampling.to_csv('C:/Users/hp/Desktop/在家学习/信用评分/sampling.csv')
cut_data=pd.DataFrame()
#age
train_data_sampling=train_data_sampling.drop(data[data['age']==0].index)
train_data_sampling.reset_index(inplace=True)
train_data_sampling.drop('index',axis=1,inplace=True)

k_age=KMeans(n_clusters=9,random_state=4,init='random')
k_age.fit_transform(train_data_sampling[['age']])
k_age.cluster_centers_#28 35 40 47 54 61 68 77 86
cut_age=pd.cut(train_data_sampling.age,bins=[0,28,35,40,47,54,61,68,77,86,110])
pd.crosstab(train_data_sampling.loc[:,'age'],train_data_sampling.loc[:,'SeriousDlqin2yrs'])
#RevolvingUtilizationOfUnsecuredLines保留变量，因为相关性不高
'''
k_Rev=KMeans(n_clusters=4,random_state=4,init='random')
k_Rev.fit_transform(data[['RevolvingUtilizationOfUnsecuredLines']])
k_Rev.cluster_centers_
'''
cut_Rev=pd.qcut(train_data_sampling.RevolvingUtilizationOfUnsecuredLines,q=5)
cut_Rev.value_counts()
#NumberOfTime30-59DaysPastDueNotWorse
max=train_data_sampling.loc[(train_data_sampling['NumberOfTime30-59DaysPastDueNotWorse']!=98)&(train_data_sampling['NumberOfTime30-59DaysPastDueNotWorse']!=96),'NumberOfTime30-59DaysPastDueNotWorse'].max()
New=[]
for val in train_data_sampling['NumberOfTime30-59DaysPastDueNotWorse']:
    if ((val == 98) | (val == 96)):
        New.append(max)
    else:
        New.append(val)
train_data_sampling['NumberOfTime30-59DaysPastDueNotWorse']=New

cut_NumberOf3059Time=pd.cut(train_data_sampling['NumberOfTime30-59DaysPastDueNotWorse'],bins=[-np.inf,0,1,2,4,np.inf])
#cut_NumberOf3059Time=pd.qcut(train_data_sampling['NumberOfTime30-59DaysPastDueNotWorse'],q=5)
cut_NumberOf3059Time.value_counts()
#DebtRatio
cut_ratio=pd.qcut(train_data_sampling.DebtRatio,q=5)
cut_ratio.value_counts()
#MonthlyIncome
cut_income=pd.qcut(train_data_sampling.MonthlyIncome,q=10)
cut_income.value_counts()
#NumberOfOpenCreditLinesAndLoans
train_data_sampling['NumberOfOpenCreditLinesAndLoans'].value_counts()
cut_loans=pd.qcut(train_data_sampling['NumberOfOpenCreditLinesAndLoans'],q=10)
cut_loans.value_counts()
#NumberOfTimes90DaysLate
max=train_data_sampling.loc[(train_data_sampling['NumberOfTimes90DaysLate']!=98)&(train_data_sampling['NumberOfTimes90DaysLate']!=96),'NumberOfTimes90DaysLate'].max()
New=[]
for val in train_data_sampling['NumberOfTimes90DaysLate']:
    if ((val == 98) | (val == 96)):
        New.append(max)
    else:
        New.append(val)
train_data_sampling['NumberOfTimes90DaysLate']=New
cut_NumberOf90time=pd.cut(train_data_sampling['NumberOfTimes90DaysLate'],bins=[-np.inf,0,1,2,4,np.inf])
cut_NumberOf90time.value_counts()
#NumberRealEstateLoansOrLines
cut_EstateLoansOrLines=pd.cut(train_data_sampling['NumberRealEstateLoansOrLines'],bins=[-np.inf,0,1,2,4,np.inf])
cut_EstateLoansOrLines.value_counts()
#NumberOfTime60-89DaysPastDueNotWorse
cut_NumberOfTime6089Days=pd.cut(train_data_sampling['NumberOfTime60-89DaysPastDueNotWorse'],bins=[-np.inf,0,1,2,4,np.inf])
cut_NumberOfTime6089Days.value_counts()
#NumberOfDependents
cut_Dependents=pd.cut(train_data_sampling['NumberOfDependents'],bins=[-np.inf,0,1,2,np.inf])
cut_Dependents.value_counts()
#fuzhaijine
cut_fuzhaijine=pd.qcut(train_data_sampling['fuzhaijine'],q=5)
cut_fuzhaijine.value_counts()
#shifouweiyue
new=[]
for x in train_data_sampling.shifouweiyue:
    if x<0.5:
        new.append(0)
    else:
        new.append(1)
train_data_sampling.shifouweiyue=new
        
train_data_sampling_cut=train_data_sampling.copy()
train_data_sampling_cut['age']=cut_age
train_data_sampling_cut['RevolvingUtilizationOfUnsecuredLines']=cut_Rev
train_data_sampling_cut['NumberOfTime30-59DaysPastDueNotWorse']=cut_NumberOf3059Time
train_data_sampling_cut['DebtRatio']=cut_ratio
train_data_sampling_cut['MonthlyIncome']=cut_income
train_data_sampling_cut['NumberOfOpenCreditLinesAndLoans']=cut_loans
train_data_sampling_cut['NumberOfTimes90DaysLate']=cut_NumberOf90time
train_data_sampling_cut['NumberRealEstateLoansOrLines']=cut_EstateLoansOrLines
train_data_sampling_cut['NumberOfTime60-89DaysPastDueNotWorse']=cut_NumberOfTime6089Days
train_data_sampling_cut['NumberOfDependents']=cut_Dependents
train_data_sampling_cut['fuzhaijine']=cut_fuzhaijine
train_data_sampling_cut['shifouweiyue']=train_data_sampling['shifouweiyue']
train_data_sampling_cut['SeriousDlqin2yrs'].value_counts()
'''
tree1=tree.DecisionTreeClassifier(max_depth=6,min_samples_split=1000)
#data_tree=pd.concat([train_data_sampling['age'],train_data_sampling['SeriousDlqin2yrs']],axis=1)
tree2=tree1.fit(train_data_sampling[['MonthlyIncome']],train_data_sampling['SeriousDlqin2yrs'])
dot_data = tree.export_graphviz(tree2, out_file=None, 
                         feature_names='MonthlyIncome',  
                         class_names='SeriousDlqin2yrs',  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
#Image(graph.create_png()) 
graph.write_pdf('C:/Users/hp/Desktop/在家学习/信用评分/w.pdf')
'''
#计算woe
totalgood = len(train_data_sampling_cut[train_data_sampling_cut['SeriousDlqin2yrs']==0])
totalbad = len(train_data_sampling_cut[train_data_sampling_cut['SeriousDlqin2yrs']==1])

def getwoe(a,p,q):
    good=len(train_data_sampling[(a>p)&(a<=q)&(train_data_sampling['SeriousDlqin2yrs']==0)])
    bad=len(train_data_sampling[(a>p)&(a<=q)&(train_data_sampling['SeriousDlqin2yrs']==1)])
    WOE=np.log((bad/totalbad)/(good/totalgood))
    return WOE
def getgoodlen(a,p,q):
    good=len(train_data_sampling[(a>p)&(a<=q)&(train_data_sampling['SeriousDlqin2yrs']==0)])
    goodlen=good/totalgood
    return goodlen
def getbadlen(a,p,q):
    bad=len(train_data_sampling[(a>p)&(a<=q)&(train_data_sampling['SeriousDlqin2yrs']==1)])
    badlen=bad/totalgood
    return badlen
#data.loc[(data1[data1['MonthlyIncome']>9000]).index,'MonthlyIncome']
woe_train_data=train_data_sampling.copy()
getwoe(train_data_sampling['age'],0,28)
pd.crosstab(train_data_sampling_cut['SeriousDlqin2yrs'],train_data_sampling_cut['age'])
woe_age1=getwoe(train_data_sampling['age'],0,28)
woe_age2=getwoe(train_data_sampling['age'],28,35)
woe_age3=getwoe(train_data_sampling['age'],35,40)
woe_age4=getwoe(train_data_sampling['age'],40,47)
woe_age5=getwoe(train_data_sampling['age'],47,54)
woe_age6=getwoe(train_data_sampling['age'],54,61)
woe_age7=getwoe(train_data_sampling['age'],61,68)
woe_age8=getwoe(train_data_sampling['age'],68,77)
woe_age9=getwoe(train_data_sampling['age'],77,86)
woe_age10=getwoe(train_data_sampling['age'],86,110)
woe_age=[woe_age1,woe_age2,woe_age3,woe_age4,woe_age5,woe_age6,woe_age7,woe_age8,woe_age9,woe_age10]
woe_train_data.loc[train_data_sampling['age']<=28,'age']=woe_age1
woe_train_data.loc[(train_data_sampling['age']>28)&(train_data_sampling['age']<=35),'age']=woe_age2
woe_train_data.loc[(train_data_sampling['age']>35)&(train_data_sampling['age']<=40),'age']=woe_age3
woe_train_data.loc[(train_data_sampling['age']>40)&(train_data_sampling['age']<=47),'age']=woe_age4
woe_train_data.loc[(train_data_sampling['age']>47)&(train_data_sampling['age']<=54),'age']=woe_age5
woe_train_data.loc[(train_data_sampling['age']>54)&(train_data_sampling['age']<=61),'age']=woe_age6
woe_train_data.loc[(train_data_sampling['age']>61)&(train_data_sampling['age']<=68),'age']=woe_age7
woe_train_data.loc[(train_data_sampling['age']>68)&(train_data_sampling['age']<=77),'age']=woe_age8
woe_train_data.loc[(train_data_sampling['age']>77)&(train_data_sampling['age']<=86),'age']=woe_age9
woe_train_data.loc[(train_data_sampling['age']>86)&(train_data_sampling['age']<=111),'age']=woe_age10
woe_train_data.age.value_counts()
iv_age1=(getbadlen(train_data_sampling['age'],0,28)-getgoodlen(train_data_sampling['age'],0,28))*woe_age1
iv_age2=(getbadlen(train_data_sampling['age'],28,35)-getgoodlen(train_data_sampling['age'],28,35))*woe_age2
iv_age3=(getbadlen(train_data_sampling['age'],35,40)-getgoodlen(train_data_sampling['age'],35,40))*woe_age3
iv_age4=(getbadlen(train_data_sampling['age'],40,47)-getgoodlen(train_data_sampling['age'],40,47))*woe_age4
iv_age5=(getbadlen(train_data_sampling['age'],47,54)-getgoodlen(train_data_sampling['age'],47,54))*woe_age5
iv_age6=(getbadlen(train_data_sampling['age'],54,61)-getgoodlen(train_data_sampling['age'],54,61))*woe_age6
iv_age7=(getbadlen(train_data_sampling['age'],61,68)-getgoodlen(train_data_sampling['age'],61,68))*woe_age7
iv_age8=(getbadlen(train_data_sampling['age'],68,77)-getgoodlen(train_data_sampling['age'],68,77))*woe_age8
iv_age9=(getbadlen(train_data_sampling['age'],77,86)-getgoodlen(train_data_sampling['age'],77,86))*woe_age9
iv_age10=(getbadlen(train_data_sampling['age'],86,110)-getgoodlen(train_data_sampling['age'],86,110))*woe_age10
iv_age=iv_age1+iv_age2+iv_age3+iv_age4+iv_age5+iv_age6+iv_age7+iv_age8+iv_age9+iv_age10#0.25819490968759973
#RevolvingUtilizationOfUnsecuredLines
pd.crosstab(train_data_sampling_cut['SeriousDlqin2yrs'],train_data_sampling_cut['RevolvingUtilizationOfUnsecuredLines'])
woe_Revolving1=np.log((3198/totalbad)/(20834/totalgood))
woe_Revolving2=np.log((6745/totalbad)/(17285/totalgood))
woe_Revolving3=np.log((13531/totalbad)/(10500/totalgood))
woe_Revolving4=np.log((18043/totalbad)/(5989/totalgood))
woe_Revolving5=np.log((18639/totalbad)/(5391/totalgood))
woe_train_data['RevolvingUtilizationOfUnsecuredLines'].max()
woe_train_data.loc[(train_data_sampling['RevolvingUtilizationOfUnsecuredLines']<=0.0535)&(train_data_sampling['RevolvingUtilizationOfUnsecuredLines']>=0),'RevolvingUtilizationOfUnsecuredLines']=woe_Revolving1
woe_train_data.loc[(train_data_sampling['RevolvingUtilizationOfUnsecuredLines']>0.0535)&(train_data_sampling['RevolvingUtilizationOfUnsecuredLines']<=0.281),'RevolvingUtilizationOfUnsecuredLines']=woe_Revolving2
woe_train_data.loc[(train_data_sampling['RevolvingUtilizationOfUnsecuredLines']>0.281)&(train_data_sampling['RevolvingUtilizationOfUnsecuredLines']<=0.652),'RevolvingUtilizationOfUnsecuredLines']=woe_Revolving3
woe_train_data.loc[(train_data_sampling['RevolvingUtilizationOfUnsecuredLines']>0.652)&(train_data_sampling['RevolvingUtilizationOfUnsecuredLines']<=0.967),'RevolvingUtilizationOfUnsecuredLines']=woe_Revolving4
woe_train_data.loc[(train_data_sampling['RevolvingUtilizationOfUnsecuredLines']>0.967)&(train_data_sampling['RevolvingUtilizationOfUnsecuredLines']<=60000),'RevolvingUtilizationOfUnsecuredLines']=woe_Revolving5
woe_train_data['RevolvingUtilizationOfUnsecuredLines'].value_counts()
iv_Revolv1=(3198/totalbad-20834/totalgood)*woe_Revolving1
iv_Revolv2=(6745/totalbad-17285/totalgood)*woe_Revolving2
iv_Revolv3=(13531/totalbad-10500/totalgood)*woe_Revolving3
iv_Revolv4=(18043/totalbad-5989/totalgood)*woe_Revolving4
iv_Revolv5=(18639/totalbad-5391/totalgood)*woe_Revolving5

iv_Revolv=iv_Revolv1+iv_Revolv2+iv_Revolv3+iv_Revolv4+iv_Revolv5#1.2229730587073095

#NumberOfTime30-59DaysPastDueNotWorse
pd.crosstab(train_data_sampling_cut['SeriousDlqin2yrs'],train_data_sampling_cut['NumberOfTime30-59DaysPastDueNotWorse'])

woe_30591=np.log((28490/totalbad)/(51935/totalgood))
woe_30592=np.log((16626/totalbad)/(5743/totalgood))
woe_30593=np.log((7862/totalbad)/(1460/totalgood))
woe_30594=np.log((5133/totalbad)/(670/totalgood))
woe_30595=np.log((2045/totalbad)/(191/totalgood))
woe_train_data['NumberOfTime30-59DaysPastDueNotWorse'].max()
woe_train_data.loc[train_data_sampling['NumberOfTime30-59DaysPastDueNotWorse']==0,'NumberOfTime30-59DaysPastDueNotWorse']=woe_30591
woe_train_data.loc[(train_data_sampling['NumberOfTime30-59DaysPastDueNotWorse']==1),'NumberOfTime30-59DaysPastDueNotWorse']=woe_30592
woe_train_data.loc[(train_data_sampling['NumberOfTime30-59DaysPastDueNotWorse']>1)&(train_data_sampling['NumberOfTime30-59DaysPastDueNotWorse']<=2),'NumberOfTime30-59DaysPastDueNotWorse']=woe_30593
woe_train_data.loc[(train_data_sampling['NumberOfTime30-59DaysPastDueNotWorse']>2)&(train_data_sampling['NumberOfTime30-59DaysPastDueNotWorse']<=4),'NumberOfTime30-59DaysPastDueNotWorse']=woe_30594
woe_train_data.loc[(train_data_sampling['NumberOfTime30-59DaysPastDueNotWorse']>4)&(train_data_sampling['NumberOfTime30-59DaysPastDueNotWorse']<=97),'NumberOfTime30-59DaysPastDueNotWorse']=woe_30595
woe_train_data['NumberOfTime30-59DaysPastDueNotWorse'].value_counts()
iv_30591=(28490/totalbad-51935/totalgood)*woe_30591
iv_30592=(16626/totalbad-5743/totalgood)*woe_30592
iv_30593=(7862/totalbad-1460/totalgood)*woe_30593
iv_30594=(5133/totalbad-670/totalgood)*woe_30594
iv_30595=(2045/totalbad-191/totalgood)*woe_30595
iv_3059=iv_30591+iv_30592+iv_30593+iv_30594+iv_30595#0.83053544388188838


#DebtRatio
woe_train_data['DebtRatio'].max()
pd.crosstab(train_data_sampling_cut['SeriousDlqin2yrs'],train_data_sampling_cut['DebtRatio'])
woe_Ratio1=np.log((10577/totalbad)/(13454/totalgood))
woe_Ratio2=np.log((11320/totalbad)/(12711/totalgood))
woe_Ratio3=np.log((12385/totalbad)/(11646/totalgood))
woe_Ratio4=np.log((14783/totalbad)/(9251/totalgood))
woe_Ratio5=np.log((11091/totalbad)/(12937/totalgood))
woe_train_data.loc[train_data_sampling['DebtRatio']<=0.153,'DebtRatio']=-woe_Ratio1
woe_train_data.loc[(train_data_sampling['DebtRatio']>0.153)&(train_data_sampling['DebtRatio']<=0.311),'DebtRatio']=woe_Ratio2
woe_train_data.loc[(train_data_sampling['DebtRatio']>0.311)&(train_data_sampling['DebtRatio']<=0.5),'DebtRatio']=woe_Ratio3
woe_train_data.loc[(train_data_sampling['DebtRatio']>0.5)&(train_data_sampling['DebtRatio']<=1.49),'DebtRatio']=woe_Ratio4
woe_train_data.loc[(train_data_sampling['DebtRatio']>1.49)&(train_data_sampling['DebtRatio']<=400000),'DebtRatio']=woe_Ratio5

woe_train_data['DebtRatio'].value_counts()
iv_Ratio1=(10577/totalbad-13454/totalgood)*woe_Ratio1
iv_Ratio2=(11320/totalbad-12711/totalgood)*woe_Ratio2
iv_Ratio3=(12385/totalbad-11646/totalgood)*woe_Ratio3
iv_Ratio4=(14783/totalbad-9251/totalgood)*woe_Ratio4
iv_Ratio5=(11091/totalbad-12937/totalgood)*woe_Ratio5
iv_Ratio=iv_Ratio1+iv_Ratio2+iv_Ratio3+iv_Ratio4+iv_Ratio5#0.062844824089719628
                
#MonthlyIncome
pd.crosstab(train_data_sampling_cut['SeriousDlqin2yrs'],train_data_sampling_cut['MonthlyIncome'])
woe_incom1=np.log((6134/totalbad)/(5886/totalgood))
woe_incom2=np.log((5942/totalbad)/(6185/totalgood))
woe_incom3=np.log((7055/totalbad)/(5243/totalgood))
woe_incom4=np.log((7016/totalbad)/(5605/totalgood))
woe_incom5=np.log((6120/totalbad)/(4898/totalgood))
woe_incom6=np.log((6384/totalbad)/(5626/totalgood))
woe_incom7=np.log((6167/totalbad)/(5860/totalgood))
woe_incom8=np.log((5555/totalbad)/(6452/totalgood))
woe_incom9=np.log((5145/totalbad)/(6868/totalgood))
woe_incom10=np.log((4638/totalbad)/(7376/totalgood))
woe_train_data.loc[train_data_sampling['MonthlyIncome']<=1140.342,'MonthlyIncome']=woe_incom1
woe_train_data.loc[(train_data_sampling['MonthlyIncome']>1140.342)&(train_data_sampling['MonthlyIncome']<=1943.438),'MonthlyIncome']=woe_incom2
woe_train_data.loc[(train_data_sampling['MonthlyIncome']>1943.438)&(train_data_sampling['MonthlyIncome']<=2800.0),'MonthlyIncome']=woe_incom3
woe_train_data.loc[(train_data_sampling['MonthlyIncome']>2800.0)&(train_data_sampling['MonthlyIncome']<=3500.0),'MonthlyIncome']=woe_incom4
woe_train_data.loc[(train_data_sampling['MonthlyIncome']>3500.0)&(train_data_sampling['MonthlyIncome']<=4225.0),'MonthlyIncome']=woe_incom5
woe_train_data.loc[(train_data_sampling['MonthlyIncome']>4225.0)&(train_data_sampling['MonthlyIncome']<=5125.153),'MonthlyIncome']=woe_incom6
woe_train_data.loc[(train_data_sampling['MonthlyIncome']>5125.153)&(train_data_sampling['MonthlyIncome']<=6184.002),'MonthlyIncome']=woe_incom7
woe_train_data.loc[(train_data_sampling['MonthlyIncome']>6184.002)&(train_data_sampling['MonthlyIncome']<=7675.0),'MonthlyIncome']=woe_incom8
woe_train_data.loc[(train_data_sampling['MonthlyIncome']>7675.0)&(train_data_sampling['MonthlyIncome']<=10166.0),'MonthlyIncome']=woe_incom9
woe_train_data.loc[(train_data_sampling['MonthlyIncome']>10166.0),'MonthlyIncome']=woe_incom10
woe_train_data.MonthlyIncome.value_counts()
iv_incom1=(6134/totalbad-5886/totalgood)*woe_incom1
iv_incom2=(5942/totalbad-6185/totalgood)*woe_incom2
iv_incom3=(7055/totalbad-5243/totalgood)*woe_incom3
iv_incom4=(7016/totalbad-5605/totalgood)*woe_incom4
iv_incom5=(6120/totalbad-4898/totalgood)*woe_incom5
iv_incom6=(6384/totalbad-5626/totalgood)*woe_incom6
iv_incom7=(6167/totalbad-5860/totalgood)*woe_incom7
iv_incom8=(5555/totalbad-6452/totalgood)*woe_incom8
iv_incom9=(5145/totalbad-6868/totalgood)*woe_incom9
iv_incom10=(4638/totalbad-7376/totalgood)*woe_incom10
iv_incom=iv_incom1+iv_incom2+iv_incom3+iv_incom4+iv_incom5+iv_incom6+iv_incom7+iv_incom8+iv_incom9+iv_incom10#0.05260337229962106

#NumberOfOpenCreditLinesAndLoans
pd.crosstab(train_data_sampling_cut['SeriousDlqin2yrs'],train_data_sampling_cut['NumberOfOpenCreditLinesAndLoans'])
woe_Loans1=np.log((9379/totalbad)/(4883/totalgood))
woe_Loans2=np.log((8800/totalbad)/(8259/totalgood))
woe_Loans3=np.log((5067/totalbad)/(5146/totalgood))
woe_Loans4=np.log((4660/totalbad)/(5509/totalgood))
woe_Loans5=np.log((4522/totalbad)/(5302/totalgood))
woe_Loans6=np.log((8005/totalbad)/(9696/totalgood))
woe_Loans7=np.log((3590/totalbad)/(3916/totalgood))
woe_Loans8=np.log((5650/totalbad)/(6123/totalgood))
woe_Loans9=np.log((5409/totalbad)/(5627/totalgood))
woe_Loans10=np.log((5074/totalbad)/(5538/totalgood))
woe_train_data.loc[woe_train_data['NumberOfOpenCreditLinesAndLoans']<=2.0,'NumberOfOpenCreditLinesAndLoans']=woe_Loans1
woe_train_data.loc[(woe_train_data['NumberOfOpenCreditLinesAndLoans']>2.0)&(woe_train_data['NumberOfOpenCreditLinesAndLoans']<=4.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans2
woe_train_data.loc[(woe_train_data['NumberOfOpenCreditLinesAndLoans']>4.0)&(woe_train_data['NumberOfOpenCreditLinesAndLoans']<=5.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans3
woe_train_data.loc[(woe_train_data['NumberOfOpenCreditLinesAndLoans']>5.0)&(woe_train_data['NumberOfOpenCreditLinesAndLoans']<=6.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans4
woe_train_data.loc[(woe_train_data['NumberOfOpenCreditLinesAndLoans']>6.0)&(woe_train_data['NumberOfOpenCreditLinesAndLoans']<=7.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans5
woe_train_data.loc[(woe_train_data['NumberOfOpenCreditLinesAndLoans']>7.0)&(woe_train_data['NumberOfOpenCreditLinesAndLoans']<=9.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans6
woe_train_data.loc[(woe_train_data['NumberOfOpenCreditLinesAndLoans']>9.0)&(woe_train_data['NumberOfOpenCreditLinesAndLoans']<=10.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans7
woe_train_data.loc[(woe_train_data['NumberOfOpenCreditLinesAndLoans']>10.0)&(woe_train_data['NumberOfOpenCreditLinesAndLoans']<=12.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans8
woe_train_data.loc[(woe_train_data['NumberOfOpenCreditLinesAndLoans']>12.0)&(woe_train_data['NumberOfOpenCreditLinesAndLoans']<=15.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans9
woe_train_data.loc[(woe_train_data['NumberOfOpenCreditLinesAndLoans']>15.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans10
woe_train_data.NumberOfOpenCreditLinesAndLoans.value_counts()
iv_Loans1=(9379/totalbad-4883/totalgood)*woe_Loans1
iv_Loans2=(8800/totalbad-8259/totalgood)*woe_Loans2
iv_Loans3=(5067/totalbad-5146/totalgood)*woe_Loans3
iv_Loans4=(4660/totalbad-5509/totalgood)*woe_Loans4
iv_Loans5=(4522/totalbad-5302/totalgood)*woe_Loans5
iv_Loans6=(8005/totalbad-9696/totalgood)*woe_Loans6
iv_Loans7=(3590/totalbad-3916/totalgood)*woe_Loans7
iv_Loans8=(5650/totalbad-6123/totalgood)*woe_Loans8
iv_Loans9=(5409/totalbad-5627/totalgood)*woe_Loans9
iv_Loans10=(5074/totalbad-5538/totalgood)*woe_Loans10
iv_Loans=iv_Loans1+iv_Loans2+iv_Loans3+iv_Loans4+iv_Loans5+iv_Loans6+iv_Loans7+iv_Loans8+iv_Loans9+iv_Loans10#0.061174706202253015

#NumberOfTimes90DaysLate
woe_train_data['NumberOfTimes90DaysLate'].max()
pd.crosstab(train_data_sampling_cut['SeriousDlqin2yrs'],train_data_sampling_cut['NumberOfTimes90DaysLate'])
woe_901=np.log((38146/totalbad)/(38146/totalgood))
woe_902=np.log((12389/totalbad)/(1521/totalgood))
woe_903=np.log((4774/totalbad)/(344/totalgood))
woe_904=np.log((3085/totalbad)/(179/totalgood))
woe_905=np.log((1762/totalbad)/(95/totalgood))
woe_train_data.loc[train_data_sampling['NumberOfTimes90DaysLate']==0.0,'NumberOfTimes90DaysLate']=woe_901
woe_train_data.loc[(train_data_sampling['NumberOfTimes90DaysLate']==1.0),'NumberOfTimes90DaysLate']=woe_902
woe_train_data.loc[(train_data_sampling['NumberOfTimes90DaysLate']>1.0)&(train_data_sampling['NumberOfTimes90DaysLate']<=2.0),'NumberOfTimes90DaysLate']=woe_903
woe_train_data.loc[(train_data_sampling['NumberOfTimes90DaysLate']>2.0)&(train_data_sampling['NumberOfTimes90DaysLate']<=4.0),'NumberOfTimes90DaysLate']=woe_904
woe_train_data.loc[(train_data_sampling['NumberOfTimes90DaysLate']>4.0)&(train_data_sampling['NumberOfTimes90DaysLate']<=97),'NumberOfTimes90DaysLate']=woe_905
woe_train_data.NumberOfTimes90DaysLate.value_counts()
iv_901=(38146/totalbad-4883/totalgood)*woe_901
iv_902=(12389/totalbad-1521/totalgood)*woe_902
iv_903=(4774/totalbad-344/totalgood)*woe_903
iv_904=(3085/totalbad-179/totalgood)*woe_904
iv_905=(1762/totalbad-95/totalgood)*woe_905
iv_90=iv_901+iv_902+iv_903+iv_904+iv_905#0.55829418354740168
#NumberRealEstateLoansOrLines
pd.crosstab(train_data_sampling_cut['SeriousDlqin2yrs'],train_data_sampling_cut['NumberRealEstateLoansOrLines'])
woe_Lines1=np.log((26932/totalbad)/(22100/totalgood))
woe_Lines2=np.log((17936/totalbad)/(21270/totalgood))
woe_Lines3=np.log((10526/totalbad)/(12656/totalgood))
woe_Lines4=np.log((3621/totalbad)/(3429/totalgood))
woe_Lines5=np.log((1141/totalbad)/(544/totalgood))
woe_train_data.loc[train_data_sampling['NumberRealEstateLoansOrLines']<=0.0,'NumberRealEstateLoansOrLines']=woe_Lines1
woe_train_data.loc[(train_data_sampling['NumberRealEstateLoansOrLines']>0.0)&(train_data_sampling['NumberRealEstateLoansOrLines']<=1.0),'NumberRealEstateLoansOrLines']=woe_Lines2
woe_train_data.loc[(train_data_sampling['NumberRealEstateLoansOrLines']>1.0)&(train_data_sampling['NumberRealEstateLoansOrLines']<=2.0),'NumberRealEstateLoansOrLines']=woe_Lines3
woe_train_data.loc[(train_data_sampling['NumberRealEstateLoansOrLines']>2.0)&(train_data_sampling['NumberRealEstateLoansOrLines']<=4.0),'NumberRealEstateLoansOrLines']=woe_Lines4
woe_train_data.loc[(train_data_sampling['NumberRealEstateLoansOrLines']>4.0)&(train_data_sampling['NumberRealEstateLoansOrLines']<=54),'NumberRealEstateLoansOrLines']=woe_Lines5

woe_train_data.NumberRealEstateLoansOrLines.value_counts()
iv_Lines1=(26932/totalbad-22100/totalgood)*woe_Lines1
iv_Lines2=(17936/totalbad-21270/totalgood)*woe_Lines2
iv_Lines3=(10526/totalbad-12656/totalgood)*woe_Lines3
iv_Lines4=(3621/totalbad-3429/totalgood)*woe_Lines4
iv_Lines5=(1141/totalbad-544/totalgood)*woe_Lines5
iv_Lines=iv_Lines1+iv_Lines2+iv_Lines3+iv_Lines4+iv_Lines5#0.039425418770289836
woe_train_data['NumberRealEstateLoansOrLines'].max()
#NumberOfTime60-89DaysPastDueNotWorse
woe_train_data['NumberOfTime60-89DaysPastDueNotWorse'].min()
pd.crosstab(train_data_sampling_cut['SeriousDlqin2yrs'],train_data_sampling_cut['NumberOfTime60-89DaysPastDueNotWorse'])
woe_60891=np.log((42678/totalbad)/(57972/totalgood))
woe_60892=np.log((12210/totalbad)/(1653/totalgood))
woe_60893=np.log((3103/totalbad)/(248/totalgood))
woe_60894=np.log((1117/totalbad)/(77/totalgood))
woe_60895=np.log((1048/totalbad)/(49/totalgood))
woe_train_data.loc[(train_data_sampling['NumberOfTime60-89DaysPastDueNotWorse']<=0.0),'NumberOfTime60-89DaysPastDueNotWorse']=woe_60891
woe_train_data.loc[(train_data_sampling['NumberOfTime60-89DaysPastDueNotWorse']>0.0)&(train_data_sampling['NumberOfTime60-89DaysPastDueNotWorse']<=1.0),'NumberOfTime60-89DaysPastDueNotWorse']=woe_60892
woe_train_data.loc[(train_data_sampling['NumberOfTime60-89DaysPastDueNotWorse']>1.0)&(train_data_sampling['NumberOfTime60-89DaysPastDueNotWorse']<=2.0),'NumberOfTime60-89DaysPastDueNotWorse']=woe_60893
woe_train_data.loc[(train_data_sampling['NumberOfTime60-89DaysPastDueNotWorse']>2.0)&(train_data_sampling['NumberOfTime60-89DaysPastDueNotWorse']<=4.0),'NumberOfTime60-89DaysPastDueNotWorse']=woe_60894
woe_train_data.loc[(train_data_sampling['NumberOfTime60-89DaysPastDueNotWorse']>4.0)&(train_data_sampling['NumberOfTime60-89DaysPastDueNotWorse']<=98),'NumberOfTime60-89DaysPastDueNotWorse']=woe_60895
woe_train_data['NumberOfTime60-89DaysPastDueNotWorse'].value_counts()

iv_60891=(42678/totalbad-22100/totalgood)*woe_60891
iv_60892=(12210/totalbad-21270/totalgood)*woe_60892
iv_60893=(3103/totalbad-248/totalgood)*woe_60893
iv_60894=(1117/totalbad-77/totalgood)*woe_60894
iv_60895=(1048/totalbad-49/totalgood)*woe_60895
iv_6089=iv_60891+iv_60892+iv_60893+iv_60894+iv_60895#-0.19122287642712696

#NumberOfDependents
woe_train_data['NumberOfDependents'].max()
pd.crosstab(train_data_sampling_cut['SeriousDlqin2yrs'],train_data_sampling_cut['NumberOfDependents'])
woe_Dependents1=np.log((29464/totalbad)/(36205/totalgood))
woe_Dependents2=np.log((14313/totalbad)/(10825/totalgood))
woe_Dependents3=np.log((9926/totalbad)/(7763/totalgood))
woe_Dependents4=np.log((6453/totalbad)/(5206/totalgood))
woe_train_data.loc[(train_data_sampling['NumberOfDependents']==0.0),'NumberOfDependents']=woe_Dependents1
woe_train_data.loc[(train_data_sampling['NumberOfDependents']==1.0),'NumberOfDependents']=woe_Dependents2
woe_train_data.loc[(train_data_sampling['NumberOfDependents']==2.0),'NumberOfDependents']=woe_Dependents3
woe_train_data.loc[(train_data_sampling['NumberOfDependents']>2.0)&(train_data_sampling['NumberOfDependents']<=20),'NumberOfDependents']=woe_Dependents4
woe_train_data['NumberOfDependents'].value_counts()
iv_Dependents1=(29464/totalbad-36205/totalgood)*woe_Dependents1
iv_Dependents2=(14313/totalbad-10825/totalgood)*woe_Dependents2
iv_Dependents3=(9926/totalbad-7763/totalgood)*woe_Dependents3
iv_Dependents4=(6453/totalbad-5206/totalgood)*woe_Dependents4
iv_Dependents=iv_Dependents1+iv_Dependents2+iv_Dependents3+iv_Dependents4# 0.05263266442133803
#fuzhaijine
pd.crosstab(train_data_sampling_cut['SeriousDlqin2yrs'],train_data_sampling_cut['fuzhaijine'])
woe_fuzhaijine1=getwoe(train_data_sampling['fuzhaijine'],-0.001,538.43)
woe_fuzhaijine2=getwoe(train_data_sampling['fuzhaijine'],538.43,1495.849)
woe_fuzhaijine3=getwoe(train_data_sampling['fuzhaijine'],1495.849,2752.647)
woe_fuzhaijine4=getwoe(train_data_sampling['fuzhaijine'],2752.647,6402.004)
woe_fuzhaijine5=getwoe(train_data_sampling['fuzhaijine'],6402.004,1539561248.52)

woe_train_data.loc[(train_data_sampling['fuzhaijine']>-0.001)&(train_data_sampling['fuzhaijine']<=538.43),'fuzhaijine']=woe_fuzhaijine1
woe_train_data.loc[(train_data_sampling['fuzhaijine']>538.43)&(train_data_sampling['fuzhaijine']<=1495.849),'fuzhaijine']=woe_fuzhaijine2
woe_train_data.loc[(train_data_sampling['fuzhaijine']>1495.849)&(train_data_sampling['fuzhaijine']<=2752.647),'fuzhaijine']=woe_fuzhaijine3
woe_train_data.loc[(train_data_sampling['fuzhaijine']>2752.647)&(train_data_sampling['fuzhaijine']<=6402.004),'fuzhaijine']=woe_fuzhaijine4
woe_train_data.loc[(train_data_sampling['fuzhaijine']>6402.004)&(train_data_sampling['fuzhaijine']<=1539561248.52),'fuzhaijine']=woe_fuzhaijine5
woe_train_data['NumberOfDependents'].value_counts()
iv_fuzhaijine1=(getbadlen(train_data_sampling['fuzhaijine'],-0.001,538.43)-getgoodlen(train_data_sampling['fuzhaijine'],-0.001,538.43))*woe_fuzhaijine1
iv_fuzhaijine2=(getbadlen(train_data_sampling['fuzhaijine'],538.43,1495.849)-getgoodlen(train_data_sampling['fuzhaijine'],538.43,1495.849))*woe_fuzhaijine2
iv_fuzhaijine3=(getbadlen(train_data_sampling['fuzhaijine'],1495.849,2752.647)-getgoodlen(train_data_sampling['fuzhaijine'],1495.849,2752.647))*woe_fuzhaijine3
iv_fuzhaijine4=(getbadlen(train_data_sampling['fuzhaijine'],2752.647,56402.004)-getgoodlen(train_data_sampling['fuzhaijine'],2752.647,56402.004))*woe_fuzhaijine4
iv_fuzhaijine5=(getbadlen(train_data_sampling['fuzhaijine'],6402.004,2029810649.54)-getgoodlen(train_data_sampling['fuzhaijine'],6402.004,2029810649.54))*woe_fuzhaijine5

iv_fuzhaijine=iv_fuzhaijine1+iv_fuzhaijine2+iv_fuzhaijine3+iv_fuzhaijine4+iv_fuzhaijine5# 0.0086596257811806399
#shifouweiyue
pd.crosstab(train_data_sampling_cut['SeriousDlqin2yrs'],train_data_sampling_cut['shifouweiyue'])
woe_shifou1=getwoe(train_data_sampling['shifouweiyue'],-1,0)
woe_shifou2=getwoe(train_data_sampling['shifouweiyue'],0,1)
woe_train_data.loc[(train_data_sampling['shifouweiyue']==0.0),'shifouweiyue']=woe_shifou1
woe_train_data.loc[(train_data_sampling['shifouweiyue']==1.0),'shifouweiyue']=woe_shifou2
woe_train_data['shifouweiyue'].value_counts()

iv_shifou1=(getbadlen(train_data_sampling['shifouweiyue'],-1,0)-getgoodlen(train_data_sampling['shifouweiyue'],-1,0))*woe_fuzhaijine1
iv_shifou2=(getbadlen(train_data_sampling['shifouweiyue'],0,1)-getgoodlen(train_data_sampling['shifouweiyue'],0,1))*woe_fuzhaijine2
iv_shifou=iv_shifou1+iv_shifou2#0.050769156098225278

#建模
from sklearn.cross_validation import ShuffleSplit
clf_bl=RandomForestClassifier()
names=woe_train_data.columns
clf_bl.fit(woe_train_data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]],woe_train_data['SeriousDlqin2yrs'])
clf_bl.feature_importances_
print (sorted(zip(map(lambda x: round(x, 4), clf_bl.feature_importances_), names),reverse=True))
'''
基尼不纯度
这里特征得分实际上采用的是 Gini Importance 。使用基于不纯度的方法的时候，要记住：1、这种方法存在 偏向 ，对具有更多类别的变量会更有利；2、对于存在关联的多个特征，其中任意一个都可以作为指示器（优秀的特征），并且一旦某个特征被选择之后，其他特征的重要度就会急剧下降，因为不纯度已经被选中的那个特征降下来了，其他的特征就很难再降低那么多不纯度了，这样一来，只有先被选中的那个特征重要度很高，其他的关联特征重要度往往较低。在理解数据时，这就会造成误解，导致错误的认为先被选中的特征是很重要的，而其余的特征是不重要的，但实际上这些特征对响应变量的作用确实非常接近的（这跟Lasso是很像的）。
[(0.17549999999999999, 'SeriousDlqin2yrs'), 
(0.1221, 'MonthlyIncome'), 
(0.1076, 'RevolvingUtilizationOfUnsecuredLines'),
 (0.1055, 'DebtRatio'), 
 (0.1042, 'NumberOfOpenCreditLinesAndLoans'),
 (0.10059999999999999, 'fuzhaijine'), 
 (0.077899999999999997, 'age'), 
 (0.048500000000000001, 'NumberOfTime60-89DaysPastDueNotWorse'), 
 (0.045600000000000002, 'NumberOfTimes90DaysLate'), 
 (0.045100000000000001, 'NumberOfTime30-59DaysPastDueNotWorse'), 
 (0.037400000000000003, 'NumberOfDependents'), 
 (0.029999999999999999, 'NumberRealEstateLoansOrLines')]
'''
scores=cross_val_score(clf_bl,woe_train_data.iloc[:,[1,2,3,4,5,7,8,9,10,11,12]],woe_train_data['SeriousDlqin2yrs'])
scores.mean()# 0.80926299767756171
names=woe_train_data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]].columns
scores=[]
for i in range(woe_train_data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]].shape[1]):
    score=cross_val_score(clf_bl,woe_train_data.iloc[:,i:i+1],woe_train_data['SeriousDlqin2yrs'],
                          scoring='roc_auc',cv=ShuffleSplit(len(woe_train_data.iloc[:,[1,2,3,4,5,7,8,9,10,11,12]]),n_iter=3,test_size=0.3))
    scores.append((round(np.mean(score),3),names[i]))
print (sorted(scores))
'''   
[(0.54900000000000004, 'NumberRealEstateLoansOrLines'),
 (0.55600000000000005, 'MonthlyIncome'), 
 (0.56299999999999994, 'NumberOfDependents'),
 (0.56399999999999995, 'NumberOfTimes90DaysLate'), 
 (0.56499999999999995, 'shifouweiyue'),
 (0.63400000000000001, 'fuzhaijine'), 
 (0.63500000000000001, 'NumberOfTime30-59DaysPastDueNotWorse'),
 (0.67400000000000004, 'NumberOfTime60-89DaysPastDueNotWorse'),
 (0.70699999999999996, 'DebtRatio'), 
 (0.78300000000000003, 'age'),
 (1.0, 'RevolvingUtilizationOfUnsecuredLines')] 
'''    
param_test1={'C':np.arange(1,3,0.5)}
gsearch1=GridSearchCV(estimator=LogisticRegression(),param_grid=param_test1,cv=10)
gsearch1.fit(woe_train_data.iloc[:,[1,2,3,4,5,7,8,9,10,11,12]],woe_train_data['SeriousDlqin2yrs'])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_#650

clf=LogisticRegression(penalty='l2',C=2.5)
clf.fit(woe_train_data.iloc[:,[1,2,3,4,5,7,8,9,10,11,12]],woe_train_data['SeriousDlqin2yrs'])
clf.coef_#6的系数是负的可能多重共线性
train_x,test_x,train_y,test_y = train_test_split(woe_train_data.iloc[:,[1,2,3,4,5,7,8,9,10,11,12]],woe_train_data.iloc[:,0],test_size=0.4)
clf.fit(train_x,train_y)
y_pred=clf.predict(test_x)
answer=clf.predict_proba(test_x)[:,1]
y_pred_=[]
y_test_=[]
for x in answer:
    if x>0.5:
        y_pred_.append(1)
    else:
        y_pred_.append(0)
        
for x in test_y:
    if x>0.5:
        y_test_.append(1)
    else:
        y_test_.append(0)
#confusion_matrix=confusion_matrix(np.array(y_test_),np.array(y_pred_))
#pd.DataFrame(y_pred).describe()
roc_auc_score(test_y,y_pred)#0.78
#0.79367391796888187
#0.79728808571478227
#0.79570582284625913去掉6
data_test=data.iloc[len(data_train):,:]
data_test_woe=data_test.copy()

data_test_woe.loc[data_test['age']<=28,'age']=woe_age1
data_test_woe.loc[(data_test['age']>28)&(data_test['age']<=35),'age']=woe_age2
data_test_woe.loc[(data_test['age']>35)&(data_test['age']<=40),'age']=woe_age3
data_test_woe.loc[(data_test['age']>40)&(data_test['age']<=47),'age']=woe_age4
data_test_woe.loc[(data_test['age']>47)&(data_test['age']<=54),'age']=woe_age5
data_test_woe.loc[(data_test['age']>54)&(data_test['age']<=61),'age']=woe_age6
data_test_woe.loc[(data_test['age']>61)&(data_test['age']<=68),'age']=woe_age7
data_test_woe.loc[(data_test['age']>68)&(data_test['age']<=77),'age']=woe_age8
data_test_woe.loc[(data_test['age']>77)&(data_test['age']<=86),'age']=woe_age9
data_test_woe.loc[(data_test['age']>86)&(data_test['age']<=111),'age']=woe_age10
data_test_woe.age.value_counts()

#RevolvingUtilizationOfUnsecuredLines保留变量，因为相关性不高
data_test_woe.loc[(data_test['RevolvingUtilizationOfUnsecuredLines']<=0.0535)&(data_test['RevolvingUtilizationOfUnsecuredLines']>=0),'RevolvingUtilizationOfUnsecuredLines']=woe_Revolving1
data_test_woe.loc[(data_test['RevolvingUtilizationOfUnsecuredLines']>0.0535)&(data_test['RevolvingUtilizationOfUnsecuredLines']<=0.281),'RevolvingUtilizationOfUnsecuredLines']=woe_Revolving2
data_test_woe.loc[(data_test['RevolvingUtilizationOfUnsecuredLines']>0.281)&(data_test['RevolvingUtilizationOfUnsecuredLines']<=0.652),'RevolvingUtilizationOfUnsecuredLines']=woe_Revolving3
data_test_woe.loc[(data_test['RevolvingUtilizationOfUnsecuredLines']>0.652)&(data_test['RevolvingUtilizationOfUnsecuredLines']<=0.967),'RevolvingUtilizationOfUnsecuredLines']=woe_Revolving4
data_test_woe.loc[(data_test['RevolvingUtilizationOfUnsecuredLines']>0.967)&(data_test['RevolvingUtilizationOfUnsecuredLines']<=60000),'RevolvingUtilizationOfUnsecuredLines']=woe_Revolving5
data_test_woe['RevolvingUtilizationOfUnsecuredLines'].value_counts()

#NumberOfTime30-59DaysPastDueNotWorse
data_test_woe.loc[data_test['NumberOfTime30-59DaysPastDueNotWorse']==0,'NumberOfTime30-59DaysPastDueNotWorse']=woe_30591
data_test_woe.loc[(data_test['NumberOfTime30-59DaysPastDueNotWorse']==1),'NumberOfTime30-59DaysPastDueNotWorse']=woe_30592
data_test_woe.loc[(data_test['NumberOfTime30-59DaysPastDueNotWorse']>1)&(data_test['NumberOfTime30-59DaysPastDueNotWorse']<=2),'NumberOfTime30-59DaysPastDueNotWorse']=woe_30593
data_test_woe.loc[(data_test['NumberOfTime30-59DaysPastDueNotWorse']>2)&(data_test['NumberOfTime30-59DaysPastDueNotWorse']<=4),'NumberOfTime30-59DaysPastDueNotWorse']=woe_30594
data_test_woe.loc[(data_test['NumberOfTime30-59DaysPastDueNotWorse']>4)&(data_test['NumberOfTime30-59DaysPastDueNotWorse']<=98),'NumberOfTime30-59DaysPastDueNotWorse']=woe_30595
data_test_woe['NumberOfTime30-59DaysPastDueNotWorse'].value_counts()

#DebtRatio
data_test_woe.loc[data_test['DebtRatio']<=0.153,'DebtRatio']=-woe_Ratio1
data_test_woe.loc[(data_test['DebtRatio']>0.153)&(data_test['DebtRatio']<=0.311),'DebtRatio']=woe_Ratio2
data_test_woe.loc[(data_test['DebtRatio']>0.311)&(data_test['DebtRatio']<=0.5),'DebtRatio']=woe_Ratio3
data_test_woe.loc[(data_test['DebtRatio']>0.5)&(data_test['DebtRatio']<=1.49),'DebtRatio']=woe_Ratio4
data_test_woe.loc[(data_test['DebtRatio']>1.49)&(data_test['DebtRatio']<=400000),'DebtRatio']=woe_Ratio5

data_test_woe['DebtRatio'].value_counts()
                 
#MonthlyIncome
data_test_woe.loc[data_test['MonthlyIncome']<=1140.342,'MonthlyIncome']=woe_incom1
data_test_woe.loc[(data_test['MonthlyIncome']>1140.342)&(data_test['MonthlyIncome']<=1943.438),'MonthlyIncome']=woe_incom2
data_test_woe.loc[(data_test['MonthlyIncome']>1943.438)&(data_test['MonthlyIncome']<=2800.0),'MonthlyIncome']=woe_incom3
data_test_woe.loc[(data_test['MonthlyIncome']>2800.0)&(data_test['MonthlyIncome']<=3500.0),'MonthlyIncome']=woe_incom4
data_test_woe.loc[(data_test['MonthlyIncome']>3500.0)&(data_test['MonthlyIncome']<=4225.0),'MonthlyIncome']=woe_incom5
data_test_woe.loc[(data_test['MonthlyIncome']>4225.0)&(data_test['MonthlyIncome']<=5125.153),'MonthlyIncome']=woe_incom6
data_test_woe.loc[(data_test['MonthlyIncome']>5125.153)&(data_test['MonthlyIncome']<=6184.002),'MonthlyIncome']=woe_incom7
data_test_woe.loc[(data_test['MonthlyIncome']>6184.002)&(data_test['MonthlyIncome']<=7675.0),'MonthlyIncome']=woe_incom8
data_test_woe.loc[(data_test['MonthlyIncome']>7675.0)&(data_test['MonthlyIncome']<=10166.0),'MonthlyIncome']=woe_incom9
data_test_woe.loc[(data_test['MonthlyIncome']>10166.0),'MonthlyIncome']=woe_incom10
data_test_woe.MonthlyIncome.value_counts()
#NumberOfOpenCreditLinesAndLoans
data_test_woe.loc[data_test['NumberOfOpenCreditLinesAndLoans']<=2.0,'NumberOfOpenCreditLinesAndLoans']=woe_Loans1
data_test_woe.loc[(data_test['NumberOfOpenCreditLinesAndLoans']>2.0)&(data_test['NumberOfOpenCreditLinesAndLoans']<=4.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans2
data_test_woe.loc[(data_test['NumberOfOpenCreditLinesAndLoans']>4.0)&(data_test['NumberOfOpenCreditLinesAndLoans']<=5.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans3
data_test_woe.loc[(data_test['NumberOfOpenCreditLinesAndLoans']>5.0)&(data_test['NumberOfOpenCreditLinesAndLoans']<=6.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans4
data_test_woe.loc[(data_test['NumberOfOpenCreditLinesAndLoans']>6.0)&(data_test['NumberOfOpenCreditLinesAndLoans']<=7.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans5
data_test_woe.loc[(data_test['NumberOfOpenCreditLinesAndLoans']>7.0)&(data_test['NumberOfOpenCreditLinesAndLoans']<=9.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans6
data_test_woe.loc[(data_test['NumberOfOpenCreditLinesAndLoans']>9.0)&(data_test['NumberOfOpenCreditLinesAndLoans']<=10.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans7
data_test_woe.loc[(data_test['NumberOfOpenCreditLinesAndLoans']>10.0)&(data_test['NumberOfOpenCreditLinesAndLoans']<=12.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans8
data_test_woe.loc[(data_test['NumberOfOpenCreditLinesAndLoans']>12.0)&(data_test['NumberOfOpenCreditLinesAndLoans']<=15.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans9
data_test_woe.loc[(data_test['NumberOfOpenCreditLinesAndLoans']>15.0),'NumberOfOpenCreditLinesAndLoans']=woe_Loans10
data_test_woe.NumberOfOpenCreditLinesAndLoans.value_counts()

#NumberOfTimes90DaysLate
data_test_woe.loc[data_test['NumberOfTimes90DaysLate']==0.0,'NumberOfTimes90DaysLate']=woe_901
data_test_woe.loc[(data_test['NumberOfTimes90DaysLate']==1.0),'NumberOfTimes90DaysLate']=woe_902
data_test_woe.loc[(data_test['NumberOfTimes90DaysLate']>1.0)&(data_test['NumberOfTimes90DaysLate']<=2.0),'NumberOfTimes90DaysLate']=woe_903
data_test_woe.loc[(data_test['NumberOfTimes90DaysLate']>2.0)&(data_test['NumberOfTimes90DaysLate']<=4.0),'NumberOfTimes90DaysLate']=woe_904
data_test_woe.loc[(data_test['NumberOfTimes90DaysLate']>4.0)&(data_test['NumberOfTimes90DaysLate']<=98),'NumberOfTimes90DaysLate']=woe_905

data_test_woe.NumberOfTimes90DaysLate.value_counts()
#NumberRealEstateLoansOrLines
data_test_woe.loc[data_test['NumberRealEstateLoansOrLines']<=0.0,'NumberRealEstateLoansOrLines']=woe_Lines1
data_test_woe.loc[(data_test['NumberRealEstateLoansOrLines']>0.0)&(data_test['NumberRealEstateLoansOrLines']<=1.0),'NumberRealEstateLoansOrLines']=woe_Lines2
data_test_woe.loc[(data_test['NumberRealEstateLoansOrLines']>1.0)&(data_test['NumberRealEstateLoansOrLines']<=2.0),'NumberRealEstateLoansOrLines']=woe_Lines3
data_test_woe.loc[(data_test['NumberRealEstateLoansOrLines']>2.0)&(data_test['NumberRealEstateLoansOrLines']<=4.0),'NumberRealEstateLoansOrLines']=woe_Lines4
data_test_woe.loc[(data_test['NumberRealEstateLoansOrLines']>4.0)&(data_test['NumberRealEstateLoansOrLines']<=37),'NumberRealEstateLoansOrLines']=woe_Lines5
data_test_woe.NumberRealEstateLoansOrLines.value_counts()

#NumberOfTime60-89DaysPastDueNotWorse
data_test_woe.loc[(data_test['NumberOfTime60-89DaysPastDueNotWorse']<=0.0),'NumberOfTime60-89DaysPastDueNotWorse']=woe_60891
data_test_woe.loc[(data_test['NumberOfTime60-89DaysPastDueNotWorse']>0.0)&(data_test['NumberOfTime60-89DaysPastDueNotWorse']<=1.0),'NumberOfTime60-89DaysPastDueNotWorse']=woe_60892
data_test_woe.loc[(data_test['NumberOfTime60-89DaysPastDueNotWorse']>1.0)&(data_test['NumberOfTime60-89DaysPastDueNotWorse']<=2.0),'NumberOfTime60-89DaysPastDueNotWorse']=woe_60893
data_test_woe.loc[(data_test['NumberOfTime60-89DaysPastDueNotWorse']>2.0)&(data_test['NumberOfTime60-89DaysPastDueNotWorse']<=4.0),'NumberOfTime60-89DaysPastDueNotWorse']=woe_60894
data_test_woe.loc[(data_test['NumberOfTime60-89DaysPastDueNotWorse']>4.0)&(data_test['NumberOfTime60-89DaysPastDueNotWorse']<=98),'NumberOfTime60-89DaysPastDueNotWorse']=woe_60895
data_test_woe['NumberOfTime60-89DaysPastDueNotWorse'].value_counts()

#NumberOfDependents
data_test_woe.loc[data_test['NumberOfDependents']<=0.5,'NumberOfDependents']=0
data_test_woe.loc[(data_test['NumberOfDependents']<=1.5)&(data_test['NumberOfDependents']>=0.5),'NumberOfDependents']=1
data_test_woe.loc[(data_test['NumberOfDependents']<=2.5)&(data_test['NumberOfDependents']>=1.5),'NumberOfDependents']=2
data_test_woe.loc[(data_test['NumberOfDependents']>2.5),'NumberOfDependents']=4
data_test_woe.loc[(data_test['NumberOfDependents']==0.0),'NumberOfDependents']=woe_Dependents1
data_test_woe.loc[(data_test['NumberOfDependents']==1.0),'NumberOfDependents']=woe_Dependents2
data_test_woe.loc[(data_test['NumberOfDependents']==2.0),'NumberOfDependents']=woe_Dependents3
data_test_woe.loc[(data_test['NumberOfDependents']>2.0)&(data_test_woe['NumberOfDependents']<=20),'NumberOfDependents']=woe_Dependents4
data_test_woe['NumberOfDependents'].value_counts()
#fuzhaijine
pd.crosstab(train_data_sampling_cut['SeriousDlqin2yrs'],train_data_sampling_cut['fuzhaijine'])
data_test_woe.loc[(data_test['fuzhaijine']>-0.001)&(data_test['fuzhaijine']<=538.43),'fuzhaijine']=woe_fuzhaijine1
data_test_woe.loc[(data_test['fuzhaijine']>538.43)&(data_test['fuzhaijine']<=1495.849),'fuzhaijine']=woe_fuzhaijine2
data_test_woe.loc[(data_test['fuzhaijine']>1495.849)&(data_test['fuzhaijine']<=2752.647),'fuzhaijine']=woe_fuzhaijine3
data_test_woe.loc[(data_test['fuzhaijine']>2752.647)&(data_test['fuzhaijine']<=6402.004),'fuzhaijine']=woe_fuzhaijine4
data_test_woe.loc[(data_test['fuzhaijine']>6402.004)&(data_test['fuzhaijine']<=1539561248.52),'fuzhaijine']=woe_fuzhaijine5
data_test_woe['fuzhaijine'].value_counts()
#shifouweiyue
pd.crosstab(train_data_sampling_cut['SeriousDlqin2yrs'],train_data_sampling_cut['shifouweiyue'])
data_test_woe.loc[(data_test['shifouweiyue']==0.0),'shifouweiyue']=woe_shifou1
data_test_woe.loc[(data_test['shifouweiyue']==1.0),'shifouweiyue']=woe_shifou2
data_test_woe['shifouweiyue'].value_counts()


data_test_woe.reset_index(inplace=True)
data_test_woe.drop('index',axis=1,inplace=True)

test_y_pred=clf.predict(data_test_woe.iloc[:,[1,2,3,4,5,7,8,9,10,11,12]])
answer1=clf.predict_proba(data_test_woe.iloc[:,[1,2,3,4,5,7,8,9,10,11,12]])[:,1]
y_submission=answer1 
result = pd.DataFrame({"Id": data_test_woe.index+1, "Probability":y_submission})  
result.to_csv('C:/Users/hp/Desktop/在家学习/信用评分/stack_result.csv', index=False)#0.847352
#0.859255

corrmat=woe_train_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)#0.847518

#绘制roc曲线
from sklearn.cross_validation import StratifiedKFold  
from scipy import interp  

cv = StratifiedKFold(woe_train_data['SeriousDlqin2yrs'],n_folds=6)  
mean_tpr = 0.0  
mean_fpr = np.linspace(0, 1, 100)  
all_tpr = []  
train_x,test_x,train_y,test_y
for i, (train, test) in enumerate(cv):  
    probas_ = clf.fit(train_x, train_y).predict_proba(test_x)  
    fpr, tpr, thresholds = roc_curve(test_y, probas_[:, 1])  
    mean_tpr += interp(mean_fpr, fpr, tpr)          #对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数  
    mean_tpr[0] = 0.0                               #初始处为0  
    roc_auc = auc(fpr, tpr)  
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))  
  
#画对角线  
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  
  
mean_tpr /= len(cv)                     #在mean_fpr100个点，每个点处插值插值多次取平均  
mean_tpr[-1] = 1.0                      #坐标最后一个点为（1,1）  
mean_auc = auc(mean_fpr, mean_tpr)      #计算平均AUC值  
#画平均ROC曲线  
#print mean_fpr,len(mean_fpr)  
#print mean_tpr  
plt.plot(mean_fpr, mean_tpr, 'k--',  
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)  
  
plt.xlim([-0.05, 1.05])  
plt.ylim([-0.05, 1.05])  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title('Receiver operating characteristic example')  
plt.legend(loc="lower right")  
plt.show()  

ks=(tpr-fpr).max()
print(ks)#0.595559905785#cut的最佳切分点
    
























































