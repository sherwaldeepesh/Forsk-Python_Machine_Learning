# -*- coding: utf-8 -*-
"""
Created on Tue May 29 19:48:29 2018

@author: deepe
"""

import pandas as pd
df =  pd.read_csv('Loan.csv')
del df['Loan_ID']
labels = df.iloc[:,-1]
features = df.iloc[:,:-1]


df.dtypes

for columns in features.columns:
    if features[columns].dtype == 'object':
        features[columns] = features[columns].astype('category')
        features[columns] = features[columns].cat.codes
    else:
        break
    
features['Property_Area'] = features['Property_Area'].astype('category')
features['Property_Area'] = features['Property_Area'].cat.codes    

df_with_dummies = pd.get_dummies(df['Property_Area'])

labels = labels.astype('category')
labels = labels.cat.codes