# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:38:35 2018

@author: deepe
"""

import pandas as pd
df = pd.read_csv('Red_Wine.csv')

df.isnull().sum()

df = df.apply(lambda x: x.fillna(x.mode()[0]))

features = df.iloc[:,:-1].values
labels = df.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
labelencoder = LabelEncoder()

features[:,0] = labelencoder.fit_transform(features[:,0])

onehotencoder = OneHotEncoder(categorical_features = [0])
features = onehotencoder.fit_transform(features).toarray()

labels = labelencoder.fit_transform(labels)

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.3,random_state = 0)

sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

labels_train = sc.fit_transform(labels_train.reshape(-1,1))

