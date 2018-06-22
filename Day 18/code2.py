# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 11:47:52 2018

@author: deepe
"""
import numpy as np
import pandas as pd
dataset = pd.read_csv('Auto_mpg.txt', sep='\s+', header = None , names = ['mpg', 'cylinders', 'displacement','horsepower','weight','acceleration', 'model year', 'origin', 'car name'])

dataset['horsepower'] = dataset['horsepower'].replace(['?'],dataset['horsepower'].mode())
dataset['horsepower'] = dataset['horsepower'].astype("float64")

dataset[dataset['mpg']==max(dataset['mpg'])]['car name'].values[0]


features = dataset.iloc[:,1:-1].values
labels = dataset.iloc[:,0].values

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(features_train,labels_train)

pred1 = regressor.predict(features_test)
score1 = regressor.score(features_test,labels_test)

ran_for_regressor = RandomForestRegressor(n_estimators = 10,random_state = 0)
ran_for_regressor.fit(features_train,labels_train)

pred2 = ran_for_regressor.predict(features_test)
score2 = ran_for_regressor.score(features_test,labels_test)


pred3 = ran_for_regressor.predict(scaler.transform(np.array([6,215,100,2630,22.2,80,3]).reshape(1,-1)))


"""
import statsmodels.formula.api as sm
features = np.append(arr = np.ones((398,1)).astype(int),values = features,axis = 1)

features_opt = features[:, [0,1,2,3,4,5,6,7]]
regressor_ols = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_ols.summary()

features_opt = features[:, [0,1,2,4,5,6,7]]
regressor_ols = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_ols.summary()

features_opt = features[:, [0,2,4,5,6,7]]
regressor_ols = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_ols.summary()

features_opt = features[:, [0,2,4,6,7]]
regressor_ols = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_ols.summary()

features_opt = features[:, [0,4,6,7]]
regressor_ols = sm.OLS(endog = labels, exog = features_opt).fit()
regressor_ols.summary()

features_train,features_test,labels_train,labels_test = train_test_split(features_opt,labels,test_size = 0.2, random_state = 0)
"""