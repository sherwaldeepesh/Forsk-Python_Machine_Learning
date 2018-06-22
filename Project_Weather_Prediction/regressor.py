# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 07:38:40 2018

@author: deepe
"""
import numpy as np
import pandas as pd
dataset = pd.read_csv('train_set.csv')

features = dataset.iloc[:,2:-5].values

labels = dataset.iloc[:,-5:].values

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(features_train,labels_train)

regressor.predict(features_test)

regressor.score(features_test,labels_test)




"""
from sklearn.tree import DecisionTreeRegressor

regressor1 = DecisionTreeRegressor(random_state = 0)

regressor1.fit(features_train,labels_train)

regressor1.predict(features_test)

regressor1.score(features_test,labels_test)
"""

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = features_test, y = labels_test,cv = 10)
print ("mean accuracy is",accuracies.mean())

print (accuracies.std())

a = np.array([102,	63,	27,	29.5,	2.1,	102,63,27,	29.5,2.1,	102,63,27,	29.5,2.1]).reshape(1,-1)


def hj():
    a = 5
    print a

import threading

def printit():
  threading.Timer(5.0, printit).start()
  hj()

printit()

