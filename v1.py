# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 01:15:37 2015

@author: bolaka
"""

from __future__ import division
from collections import defaultdict
from glob import glob
import sys

# suppress pandas warnings
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)

# imports
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
import pandas as pd
from ggplot import *
from numpy.random import seed
from scipy.special import cbrt
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold

# reproduce results
seed(786)

# load the training and test sets
data = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("Sample_Submission_i9bgj6r.csv")

alcohol = pd.read_csv("NewVariable_Alcohol.csv")
data = pd.merge(alcohol, data, how='inner')
test = pd.merge(alcohol, test, how='inner')

data.to_csv("train_joined.csv", index=False)
test.to_csv("test_joined.csv", index=False)

# sum up Missing
def sumMissing(s):
    return s.isnull().sum()

def dictMap(listOfMajors, non_major):
    mapped_dict = {}
    for i, major in enumerate(reversed(listOfMajors)):
        mapped_dict[major] = (i+1)
    mapped_dict[non_major] = 0
    return mapped_dict

def dictMap0(listOfMajors):
    mapped_dict = {}
    for i, major in enumerate(reversed(listOfMajors)):
        mapped_dict[major] = i
    mapped_dict[None] = -1
    return mapped_dict

## Pre-process data
#def dataPreprocessing(data, test):
# add the target columns to test data as 9999
test['Happy'] = "NA"

# combine the training and test datasets for data preprocessing
combined = pd.concat( [ data, test ] )    

#    # clean TV hours
#    combined.loc[ combined.TVhours > 19, 'TVhours' ] = 18   

# Alcohol_Consumption
#    alcohol_counts = data.Alcohol_Consumption.value_counts()
#    major_alcohol = list(alcohol_counts.index)
#    alcohol_map = dictMap0(major_alcohol)
alcohol_map = { 'Never' : 0,
               'Rarely' : 1,
               'Occassional' : 2,
               'Once a month' : 3,
               '2 - 3 times a month' : 4,
               'Once a week' : 5,
               'Multiple times in a week' : 6,
               None : -1
             }
combined['Alcohol_Consumption'] = combined['Alcohol_Consumption'].map( alcohol_map ).astype(int)    

# Divorce - yes = 1, no = 0, missing = -1
combined['Divorce'] = combined['Divorce'].map( {'yes': 1, 'no': 0, None: -1} ).astype(int)

# Widowed - no = 0, yes = 1
combined['Widowed'] = combined['Widowed'].map( {'yes': 1, 'no': 0, None: -1} ).astype(int)

# income
income_map = { 'lt $1000' : 0,
               '$1000 to 2999' : 1,
               '$3000 to 3999' : 2,
               '$4000 to 4999' : 3,
               '$5000 to 5999' : 4,
               '$6000 to 6999' : 5,
               '$7000 to 7999' : 6,
               '$8000 to 9999' : 7,
               '$10000 - 14999' : 8,
               '$15000 - 19999' : 9,
               '$20000 - 24999' : 10,
               '$25000 or more' : 11,
               None : -1
             }
combined['income'] = combined['income'].map( income_map ).astype(int)

# Engagement_Religion
religion_counts = data.Engagement_Religion.value_counts()
major_religion = list(religion_counts.index)
religion_map = dictMap0(major_religion)
#    religion_map = { 'never' : 0,
#                   'lt once a year' : 1,
#                   'once a year' : 2,
#                   'sevrl times a yr' : 3,
#                   'once a month' : 4,
#                   '2-3x a month' : 5,
#                   'nrly every week' : 6,
#                   'every week' : 7,
#                   'more thn once wk' : 8,
#                   None : -1
#                 }
combined['Engagement_Religion'] = combined['Engagement_Religion'].map( religion_map ).astype(int)

# Var1
var1_counts = data.Var1.value_counts()
major_var1 = list(var1_counts.index)
mapped_var1 = dictMap0(major_var1)
combined['Var1'] = combined['Var1'].map( mapped_var1 ).astype(int)

# WorkStatus
work_counts = data.WorkStatus.value_counts()
major_work = list(work_counts.index)
mapped_work = dictMap0(major_work)
combined['WorkStatus'] = combined['WorkStatus'].map( mapped_work ).astype(int)

# Residence_Region
resi_counts = data.Residence_Region.value_counts()
major_resi = list(resi_counts.index)
mapped_resi = dictMap0(major_resi)
combined['Residence_Region'] = combined['Residence_Region'].map( mapped_resi ).astype(int)

#    # Transformations to correct skewness...
#    combined.Monthly_Income = combined.Monthly_Income.apply(np.sqrt) 
#    
#    combined.Loan_Amount_Applied = combined.Loan_Amount_Applied.apply(np.sqrt)
#    
#    combined.Existing_EMI = [np.power(x, (float(1)/3)) for x in combined.Existing_EMI ] # combined.Existing_EMI.apply(np.sqrt)
#    
#    combined.Loan_Amount_Submitted = combined.Loan_Amount_Submitted.apply(np.sqrt)
#    
#    combined.DOB_yr = [np.log(x + 1) for x in combined.DOB_yr ] 
#    
#    combined.Processing_Fee = combined.Processing_Fee.apply(np.sqrt)

#    # removing outliers    
#    combined = removeOutliers(combined)

# sum up missing    
combined['missingness'] = combined.apply(sumMissing, 1)     

# fill missing
combined = fillMissingby99(combined)

# separate again into training and test sets
data = combined.loc[ combined.Happy != "NA" ]
test = combined.loc[ combined.Happy == "NA" ]

# remove the target columns from test data
test.drop(['Happy'], axis=1, inplace=True)    
    
#    return data, test

# tolerance 10 standard deviations - replace by NaN
def removeOutliers(data):
    # remove outliers (replacing by null for std > outlier_cutoff)
    outlier_cutoff = 10
    for feature in data.columns:
        if feature in ['Happy']:
            continue
        
        data[feature + '_std'] = np.abs( (data[feature] - data[feature].mean()) / data[feature].std() )
        if len( data.loc[ data[ feature + '_std' ] > outlier_cutoff, feature ] ) > 0:
            print('removing outliers in ', feature, ':\n', data.loc[ data[ feature + '_std' ] > outlier_cutoff, feature ])
            data.loc[ data[feature + '_std'] > outlier_cutoff, feature ] = float('nan')
        data.drop( [feature + '_std'], axis=1, inplace=True)
    return data

# fill missing values by -9999
def fillMissingby99(data):
    data.fillna(-99,inplace=True)
    return data

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    
    n = len(preds)
    diff = (labels - preds) * 5
    
    diff = [50 if x == 0 else x for x in diff]    
    diff = [100 if x == 5 else x for x in diff]        
    diff = [5 if x == 10 else x for x in diff]        
    diff = [10 if x == 100 else x for x in diff]        
    
    err = sum(diff) / (n * 50)
    return 'error', -err

def evalmetric(preds, labels):
    n = len(preds)
    diff = (labels - preds) * 5
    
    diff = [50 if x == 0 else x for x in diff]    
    diff = [100 if x == 5 else x for x in diff]        
    diff = [5 if x == 10 else x for x in diff]        
    diff = [10 if x == 100 else x for x in diff]        
    
    err = sum(diff) / (n * 50)
    return 'error', -err

## Cross validation & modeling
#def modeling(data, test):

# the feature set
#    features=list(data.columns[:len(data.columns)-1])
features = ['Alcohol_Consumption', 'Engagement_Religion'] # 
#    features = ['Alcohol_Consumption', 'Divorce', 'Education', 'Engagement_Religion',
#       'Gender', 'Residence_Region', 'Score', 'TVhours',
#       'Unemployed10', 'Var1', 'Var2', 'Widowed', 'WorkStatus', 'babies',
#       'income', 'preteen', 'teens']

# X and Y
le = LabelEncoder()
target = data['Happy']
y = le.fit_transform(target)

#    data = pd.get_dummies(data[list(features)])
#    f = data.columns
#    test = pd.get_dummies(test[list(features)])
data = data[list(features)]
test = test[list(features)]
f = features
x = data.values
x_test = test.values

k_folds = 10
skf = StratifiedKFold(y, n_folds=k_folds, shuffle=True)    
validation_index, train_index = list(skf)[0]  
x_train, x_validate = x[train_index], x[validation_index]
y_train, y_validate = y[train_index], y[validation_index]  

rf_model = RandomForestClassifier(n_estimators=400, criterion='entropy',
                         min_samples_leaf=10, bootstrap=True,
                         n_jobs=-1, random_state=1234)

rf_model.fit(x_train, y_train)    
val_preds = clf.predict(x_validate)
val_preds = le.inverse_transform(np.array(val_preds, dtype=int))

    

#    # error
#    dx = xgb.DMatrix(x, label=y)
#    dtrain = xgb.DMatrix(x_train, label=y_train)
#    dval = xgb.DMatrix(x_validate, label=y_validate) # 
#    dtest = xgb.DMatrix(x_test)
#    
#    # setup parameters for xgboost
#    param = {}
#    # use softmax multi-class classification
#    param['objective'] = 'multi:softmax'
#    # scale weight of positive examples
#    param['eta'] = 0.05
#    param['max_depth'] = 7
#    param['silent'] = 2
#    param['num_class'] = 3
##    param['subsample'] = 0.7
#    
#    watchlist = [ (dtrain,'train'), (dval, 'test') ]
#    num_round = 500
#    clf = xgb.train(param, dtrain, num_round, watchlist, early_stopping_rounds=30, feval=evalerror)
##    print(clf.best_iteration)
#    # fold feature importance
#    mapFeat = dict(zip(["f"+str(i) for i in range(len(f))],f))
#    ts = pd.Series(clf.get_fscore())
#    ts.index = ts.reset_index()['index'].map(mapFeat)
#    
#    # plot the feature importance accross folds
#    ts.dropna().order().plot(kind="barh", title=("features importance")) # [-15:]
       
#    whole = xgb.train(param, dx, num_round, watchlist, early_stopping_rounds=30, feval=evalerror)
test_preds = clf.predict(dtest)
test_preds = le.inverse_transform(np.array(test_preds, dtype=int))

#    return test_preds

## approach 1
## TODO - parameter tuning
#data1, test1 = dataPreprocessing(data.copy(deep=True), test.copy(deep=True))
#predictions = modeling(data1, test1) # 

test_ids = test['ID']
submission['ID'] = test_ids
submission['Happy'] = predictions
submission.to_csv("xgboost_benchmark.csv", index=False)