# imports libraries
from collections import defaultdict
import sys
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing
#from sklearn.metrics import roc_curve, auc
from numpy.random import seed
#from scipy.special import cbrt
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
#from scipy.stats import rankdata

# reproduce results
seed(584)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print train.shape, test.shape

#import the additional variable provided
alcohol = pd.read_csv('NewVariable_Alcohol.csv')
print alcohol.shape

train = train.merge(alcohol, on='ID')
test = test.merge(alcohol, on='ID')
print train.shape, test.shape

categorical_vars = ['Var1', 'WorkStatus', 'Divorce', 'Widowed', 'Residence_Region', 'income', 'Engagement_Religion', 
                    u'babies', u'preteen', u'teens', 'Var2', 'Gender', 'Unemployed10', 'Alcohol_Consumption']
                    
#encoding some of the categorical vars to capture information in their ordering
number = preprocessing.LabelEncoder()
for var in ['WorkStatus', 'Residence_Region', 'income', 'Engagement_Religion', 'Alcohol_Consumption']:
    train[var+'_encoded'] = number.fit_transform(train[var].astype('str'))
    test[var+'_encoded'] = number.fit_transform(test[var].astype('str'))
    
numeric_vars = [u'Education', u'TVhours', 'Score']

#removing outliers as per standard deviation
train.ix[train['babies'] >= 3, 'babies'] = 3
test.ix[test['babies'] >= 3, 'babies'] = 3

train.ix[train['preteen'] >= 4, 'preteen'] = 4
test.ix[test['babies'] >= 4, 'preteen'] = 4

train.ix[train['teens'] >= 3, 'teens'] = 3
test.ix[test['teens'] >= 3, 'teens'] = 3

#removing outliers
outlier_cutoff = 7
for feature in numeric_vars:
    train[feature + '_std'] = np.abs( (train[feature] - train[feature].mean()) / train[feature].std() )
    if len( train.ix[ train[ feature + '_std' ] > outlier_cutoff, feature ] ) > 0:
        print('removing outliers in ', feature, ':\n', train.loc[ train[ feature + '_std' ] > outlier_cutoff, feature ])
        #train.loc[ train[feature + '_std'] > outlier_cutoff, feature ] = np.nan
    train.drop( [feature + '_std'], axis=1, inplace=True)

#removing outliers in test set
outlier_cutoff = 7
for feature in numeric_vars:
    test[feature + '_std'] = np.abs( (test[feature] - test[feature].mean()) / test[feature].std() )
    if len( test.ix[ test[ feature + '_std' ] > outlier_cutoff, feature ] ) > 0:
        print('removing outliers in ', feature, ':\n', test.loc[ test[ feature + '_std' ] > outlier_cutoff, feature ])
        test.ix[ test[feature + '_std'] > outlier_cutoff, feature ] = np.nan
    test.drop( [feature + '_std'], axis=1, inplace=True)

train= train.fillna(-999)
test = test.fillna(-999)

data = train.copy()

label = data['Happy'].map({'Very Happy': 2, 'Pretty Happy': 1, 'Not Happy': 0})

dropCols = ['ID', 'Happy', 'Gender']
data.drop(dropCols, axis=1, inplace = True)

y = label
X = pd.get_dummies(data)

holdout_fold = StratifiedShuffleSplit(y, n_iter=1, test_size=0.25)
holdout_fold

for train_index, holdout_index in holdout_fold:
    X_train, X_test = X.ix[train_index], X.ix[holdout_index]
    y_train, y_test = y[train_index], y[holdout_index]
    print X_train.shape, X_test.shape, y_train.shape, y_test.shape
    
    #finding the best parameters for RF using GridSearchCV
    from sklearn import grid_search
    rf = RandomForestClassifier(class_weight={0:1, 1:.5, 2:.5}, criterion = 'gini', oob_score=True, bootstrap = True, n_jobs=-1, random_state=584) 
    parameters = {'n_estimators':[200, 400, 600], 'max_depth':[5, 7, 9], 'min_samples_leaf':[1,3,6]}
    clf_grid = grid_search.GridSearchCV(rf, parameters, cv=4, n_jobs=4)
    clf_grid.fit(X_train, y_train)
    
    print clf_grid.best_params_
    
    rf_best = clf_grid.best_estimator_
    pred_ytest = rf_best.predict_proba(X_test)
    
    importances = rf_best.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf_best.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(20):
        print("%d. feature %s (%f)" % (f + 1, X_train.columns[indices[f]], importances[indices[f]]))
    
    #predicting probabilities
    X_test['prediction'] = np.argmax(pred_ytest.reshape( y_test.shape[0], 3 ), axis=1)
    X_test['prediction'].value_counts()
    
    print confusion_matrix(X_test['prediction'],y_test)    
    
    #calculating the evaluation metric
    score = 0
    points = []
    for (predicted, true) in zip(X_test['prediction'].astype(str).map({'2':15, '1': 10, '0': 5}), 
                                 y_test.astype(str).map({'2':15, '1': 10, '0': 5})):
        points.append(true - predicted)
    X_test['Point'] = points
    X_test['Score'] = X_test['Point'].astype(str).map({'0':50, '5':10 ,'10':5, '-5':-5, '-10':-10}) 
    print X_test['Score'].sum()/float((len(y_test)*50))
    
    pred_class0 = [var[0] for var in pred_ytest.reshape( y_test.shape[0], 3 )]
    pred_class1 = [var[1] for var in pred_ytest.reshape( y_test.shape[0], 3 )]
    pred_class2 = [var[2] for var in pred_ytest.reshape( y_test.shape[0], 3 )]
    
    hold_sub = pd.DataFrame({ 'ID': train.ix[holdout_index,'ID'], 'class0':pred_class0, 'class1':pred_class1, 'class2':pred_class2})
    hold_sub.to_csv('rf_hold_sub.csv')

#For test
test2 = test.copy()
testdropcols = list(set(dropCols)-set(['Happy']))
test2 = test2.drop(testdropcols, axis=1)

for var in test2.columns:
    new = list(set(test2[var]) - set(train[var]))
    test2.ix[test2[var].isin(new), var] = np.nan

final_test = pd.get_dummies(test2)
missingCols = list(set(X.columns)-set(final_test.columns))
for col in missingCols:
    final_test[col] = 0
final_test = final_test[X.columns]
assert X.columns.equals(final_test.columns)
final_test = final_test.fillna(-999)

clf_full = RandomForestClassifier(n_estimators=400, max_depth=9, criterion = 'gini', min_samples_split=2, 
                             min_samples_leaf=3, class_weight={0:1, 1:.5, 2:.5}, oob_score=True, bootstrap = True, n_jobs=-1, random_state=584)
clf_full.fit(X, y)
pred_finaltest = clf_full.predict_proba(final_test)

final_class0 = [var[0] for var in pred_finaltest.reshape( final_test.shape[0], 3 )]
final_class1 = [var[1] for var in pred_finaltest.reshape( final_test.shape[0], 3 )]
final_class2 = [var[2] for var in pred_finaltest.reshape( final_test.shape[0], 3 )]

sub = pd.DataFrame({'ID': test['ID'], 'class0':final_class0, 'class1':final_class1, 'class2':final_class2})
sub.to_csv('rf_sub.csv')