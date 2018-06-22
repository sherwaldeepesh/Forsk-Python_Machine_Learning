# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:24:48 2018

@author: deepe
"""

import pandas as pd
df =  pd.read_csv('Automobile.csv')

df.dtypes

df2 = df.select_dtypes(include = ['object'])

df2.isnull().sum()

df2 = df2.apply(lambda x:x.fillna(x.mode()[0]))

val = 0
for i in ('convertible','hardtop','hatchback','sedan','wagon'):
    df2['body_style'][df2['body_style']==i] = val
    val += 1
    

ob2 = df2.values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder = LabelEncoder()

for i in range(10):
    ob2[:,i] = labelencoder.fit_transform(ob2[:,i])

onehotencoder = OneHotEncoder(categorical_features = [5])
ob2 = onehotencoder.fit_transform(ob2).toarray()

onehotencoder = OneHotEncoder(categorical_features = [7])
ob2 = onehotencoder.fit_transform(ob2).toarray()

djks = pd.DataFrame(ob2)