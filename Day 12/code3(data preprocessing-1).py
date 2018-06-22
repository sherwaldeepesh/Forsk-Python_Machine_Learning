# -*- coding: utf-8 -*-
"""
Created on Mon May 28 14:14:44 2018

@author: deepe
"""

import pandas as pd
df2 = pd.read_csv('cars.xls')

price = df2.iloc[:,0].values
features = df2.iloc[:,1:].values

from sklearn.model_selection import train_test_split
features_train,features_test,price_train,price_test = train_test_split(features,price,test_size = 0.5,random_state = 0)
df3=pd.DataFrame(features_train)
df4=pd.DataFrame(features_test)
df5=pd.DataFrame(price_train)
df6=pd.DataFrame(price_test)

df3.to_csv('features_training.csv')

df4.to_csv('features_test.csv')

df5.to_csv('price_training.csv')

df6.to_csv('price_test.csv')