# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 11:10:09 2018

@author: deepe
"""

import numpy as np
import pandas as pd
dataset  = pd.read_csv('PastHires.csv')

features = dataset.iloc[:,0:-1].values
labels = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

features[:,1] = labelencoder.fit_transform(features[:,1])

for i in range(3,6):
    features[:,i] = labelencoder.fit_transform(features[:,i])
    
labels = labelencoder.fit_transform(labels)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(features,labels)

pred1 = regressor.predict(np.array([10,1,4,0,1,0]).reshape(1,-1))

pred2 = regressor.predict(np.array([10,1,4,1,0,1]).reshape(1,-1))



from sklearn.ensemble import RandomForestRegressor
ran_for_regressor = RandomForestRegressor(n_estimators = 10,random_state = 0)
ran_for_regressor.fit(features,labels)

pred3 = ran_for_regressor.predict(np.array([10,1,4,0,1,0]).reshape(1,-1))

pred4 = ran_for_regressor.predict(np.array([10,1,4,1,0,1]).reshape(1,-1))


