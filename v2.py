import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("Sample_Submission_i9bgj6r.csv")

alcohol = pd.read_csv("NewVariable_Alcohol.csv")
train = pd.merge(train, alcohol, how='inner')
test = pd.merge(test, alcohol, how='inner')

le = LabelEncoder()
target = train['Happy']
target = le.fit_transform(target)

test_ids, train_ids = test['ID'], train['ID']
train.drop(['ID', 'Happy'], axis=1, inplace=True)
test.drop(['ID'], axis=1, inplace=True)

# Taking only 2 variables :D
train = train[['Alcohol_Consumption', 'Engagement_Religion']]
test = test[['Alcohol_Consumption', 'Engagement_Religion']]

train = pd.get_dummies(train)
test = pd.get_dummies(test)

clf = RandomForestClassifier(n_estimators=400, criterion='entropy',
                             min_samples_leaf=10, bootstrap=True,
                             n_jobs=-1, random_state=1234)

clf.fit(train, target)

test_preds = clf.predict(test)
test_preds = le.inverse_transform(np.array(test_preds, dtype=int))

submission['ID'] = test_ids
submission['Happy'] = test_preds
submission.to_csv("NewBenchmark_0.70.csv", index=False)