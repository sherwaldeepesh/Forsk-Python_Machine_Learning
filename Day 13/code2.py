# -*- coding: utf-8 -*-
"""
Created on Tue May 29 12:44:37 2018

@author: deepe
"""

import pandas as pd
df1 = pd.read_csv('Loan.csv')
del df1['Loan_ID']
labels = df1.iloc[:,-1]
features = df1.iloc[:,:-1]

for column in features.columns:
    if (features[column].dtype == 'object'):
        features[column] = features[column].astype('category')
        features[column] = features[column].cat.codes



pd.Series(labels,dtype = 'category')
labels = pd.get_dummies(labels)



if(labels.dtype == 'object'):
    labels = labels.astype('category')
    labels = pd.get_dummies(labels)












features[]
features = pd.Series(features,dtype = 'category')
for i in range(0,5):
    s = pd.Series(features[i],dtype = 'category')
s.cat.codes




new=pd.get_dummies(features[0])