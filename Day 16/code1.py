# -*- coding: utf-8 -*-
"""
Created on Fri Jun 01 11:16:06 2018

@author: deepe
"""
import numpy as np
import pandas as pd
dataset = pd.read_csv('iq_size.csv')

features = dataset.iloc[:,1:].values
labels = dataset.iloc[:,0].values

#Normal Multiple Linear Regression Model
from sklearn.model_selection import train_test_split

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor1 = LinearRegression()
regressor.fit(features_train,labels_train)

a = regressor.predict(features_test)

score = regressor.score(features_test,labels_test)

regressor.predict(np.array([90,70,150]).reshape(1,-1))



#Optimel Model Coversion

import statsmodels.formula.api as sm

features = np.append(arr = np.ones((38,1)).astype(int), values = features,axis = 1)

features_opt = features[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog = labels,exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[0,1,2]]
regressor_OLS = sm.OLS(endog = labels,exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[1,2]]
regressor_OLS = sm.OLS(endog = labels,exog = features_opt).fit()
regressor_OLS.summary()

features_opt = features[:,[2]]
regressor_OLS = sm.OLS(endog = labels,exog = features_opt).fit()
regressor_OLS.summary()

features_train1,features_test1,labels_train1,labels_test1 = train_test_split(features_opt,labels,test_size = 0.2,random_state = 0)

regressor1.fit(features_train1,labels_train1)

a1 = regressor1.predict(features_test1)

score = regressor1.score(features_test1,labels_test1)

regressor1.predict(90)

