# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:22:25 2018

@author: deepe
"""

import pandas as pd
df1 = pd.read_csv('Loan.csv')
del df1['Loan_ID']
labels = df1.iloc[:,-1].values
features = df1.iloc[:,:-1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()


for i in range(0,5):
    features[:,i] = labelencoder.fit_transform(features[:,i])

features[:,-1] = labelencoder.fit_transform(features[:,-1]) 

onehotencoder = OneHotEncoder(categorical_features=[-1])
features = onehotencoder.fit_transform(features).toarray()

labels = labelencoder.fit_transform(labels)

from sklearn.model_selection import train_test_split

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2,random_state = 0)

