# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:43:45 2018

@author: deepe
"""

import pandas as pd

df = pd.read_csv('Loan.csv')
del df['Loan_ID']
labels = df.iloc[:,-1].values
features = df.iloc[:,:-1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features = [-1])

for i in range(0,5):
    features[:,i] = labelencoder.fit_transform(features[:,i])

features[:,-1] = labelencoder.fit_transform(features[:,-1])
features = onehotencoder.fit_transform(features).toarray()

labels = labelencoder.fit_transform(labels)

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2,random_state = 0)