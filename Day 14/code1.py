# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:12:04 2018

@author: deepe
"""

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv",header = None,usecols = [0,1,2])

df = df.rename(columns={0: 'Class label', 1: 'Alcohol', 2: 'Malic acid'})
df.isnull().sum()

features = df.iloc[:,1:].values
labels = df.iloc[:,1].values

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size = 0.2,random_state=0)

#Only use one at a time Either StandardScaler or MinMaxScaler

sc = StandardScaler()
features_train = sc.fit_transform(features_train)
features_test = sc.transform(features_test)

labels_train = sc.fit_transform(labels_train.reshape(-1,1))


scaler = MinMaxScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

labels_train = scaler.fit_transform(labels_train)