# -*- coding: utf-8 -*-
"""
Created on Wed Jun 06 11:01:57 2018

@author: deepe
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('affairs.csv')
features = dataset.iloc[:,:-1].values
labels = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

"""
labelencoder = LabelEncoder()
for i in range(6,8):
    features[:,i] = labelencoder.fit_transform(features[:,i])
"""
    
onehotencoder1 = OneHotEncoder(categorical_features = [6])
features = onehotencoder1.fit_transform(features).toarray()
features = features[:,1:]

onehotencoder2 = OneHotEncoder(categorical_features = [-1])
features = onehotencoder2.fit_transform(features).toarray()
features = features[:,1:]

from sklearn.model_selection import train_test_split
"""
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.25, random_state = 0)
"""

import statsmodels.formula.api as sm
features = np.append(arr = np.ones((6366,1)).astype(int),values = features,axis = 1)

features_opt = features[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
regressor_ols = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_ols.summary()

features_opt = features[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
regressor_ols = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_ols.summary()

features_opt = features[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15]]
regressor_ols = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_ols.summary()

features_opt = features[:, [0,1,2,3,4,5,7,8,9,10,11,12,13,15]]
regressor_ols = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_ols.summary()

features_opt = features[:, [0,1,2,4,5,7,8,9,10,11,12,13,15]]
regressor_ols = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_ols.summary()

features_opt = features[:, [0,1,2,4,7,8,9,10,11,12,13,15]]
regressor_ols = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_ols.summary()

features_opt = features[:, [0,1,2,7,8,9,10,11,12,13,15]]
regressor_ols = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_ols.summary()

features_opt = features[:, [0,2,7,8,9,10,11,12,13,15]]
regressor_ols = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_ols.summary()

features_opt = features[:, [0,2,7,9,10,11,12,13,15]]
regressor_ols = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_ols.summary()

features_opt = features[:, [0,7,9,10,11,12,13,15]]
regressor_ols = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_ols.summary()

features_opt = features[:, [0,7,9,11,12,13,15]]
regressor_ols = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_ols.summary()



features_train,features_test,labels_train,labels_test = train_test_split(features_opt,labels,test_size = 0.2, random_state = 0)




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test  =scaler.transform(features_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(features_train,labels_train)

labels_pred = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pred)

score = classifier.score(features_test,labels_test)

new_pred = classifier.predict(scaler.transform(np.array([1,1,0,25,3,1,16]).reshape(1,-1)))


"""
new_pred = classifier.predict(scaler.transform(np.array([1,0,0,0,0,0,0,1,0,0,3,25,3,1,4,16]).reshape(1,-1)))


new_pred = classifier.predict(scaler.transform(onehotencoder2.transform(onehotencoder1.transform((np.array([3,25,3,1,4,16,4,2]).reshape(1,-1))).toarray()[:,1:]).toarray()[:,1:]))
"""



"""
total = len(dataset['affair'])
count = len(dataset[dataset['affair']==1])

actual_affair = float(count)/float(total)

"""



"""
count = 0
for i in dataset['affair']:
    if i:
        count+=1
"""