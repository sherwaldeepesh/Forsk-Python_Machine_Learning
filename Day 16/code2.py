# -*- coding: utf-8 -*-
"""
Created on Fri Jun 01 12:45:21 2018

@author: deepe
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as sm


dataset = pd.read_csv('stats_females.csv')

#Where Father's and Mother's both ages feature depends to Student's age
features = dataset.iloc[:,1:].values
labels = dataset.iloc[:,0].values
features = np.append(arr = np.ones((214,1)).astype(int), values = features,axis = 1)
features_opt1 = features[:,[0,1,2]]
regressor_OLS = sm.OLS(endog = labels,exog = features_opt1).fit()
regressor_OLS.summary()

features_train1,features_test1,labels_train1,labels_test1 = train_test_split(features_opt1,labels,test_size = 0.2,random_state = 0)

regressor1 = LinearRegression()


regressor1.fit(features_train1,labels_train1)

pred1 = regressor1.predict(features_test1)

score1 = regressor1.score(features_test1,labels_test1)



#Where Father's constant and Mother's age feature depends to Student's age
features = dataset.iloc[:,1:].values
labels = dataset.iloc[:,0].values
features = np.append(arr = np.ones((214,1)).astype(int), values = features,axis = 1)
features[:, 1] =  1
features[:, 2] += 1
features_opt2 = features[:,[0,1,2]]
regressor_OLS = sm.OLS(endog = labels,exog = features_opt2).fit()
regressor_OLS.summary()

features_train2,features_test2,labels_train2,labels_test2 = train_test_split(features_opt2,labels,test_size = 0.2,random_state = 0)

regressor2 = LinearRegression()


regressor2.fit(features_train2,labels_train2)

pred2 = regressor2.predict(features_test2)

score2 = regressor2.score(features_test2,labels_test2)

#Where mother's constant and father's age feature depends to Student's age
features = dataset.iloc[:,1:].values
labels = dataset.iloc[:,0].values
features = np.append(arr = np.ones((214,1)).astype(int), values = features,axis = 1)
features[:, 2] =  1
features[:, 1] += 1
features_opt3 = features[:,[0,1,2]]
regressor_OLS = sm.OLS(endog = labels,exog = features_opt3).fit()
regressor_OLS.summary()

features_train3,features_test3,labels_train3,labels_test3 = train_test_split(features_opt3,labels,test_size = 0.2,random_state = 0)

regressor3 = LinearRegression()


regressor3.fit(features_train3,labels_train3)

pred3 = regressor3.predict(features_test3)

score3 = regressor3.score(features_test3,labels_test3)


normal_average = np.average(pred1)
average_on_father_const = np.average(pred2)
average_on_mother_const = np.average(pred3)

change_when_father_const = normal_average - average_on_father_const
change_when_mother_const = normal_average - average_on_mother_const