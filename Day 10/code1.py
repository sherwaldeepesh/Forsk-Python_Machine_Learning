# -*- coding: utf-8 -*-
"""
Created on Thu May 24 12:00:10 2018

@author: deepe
"""

import pandas as pd

df1 = pd.read_csv('training_titanic.xls')

print df1['Survived'].value_counts()

print df1["Survived"].value_counts(normalize = True)

df1_m = df1[df1['Sex'] == 'male']
print df1_m['Survived'].value_counts()
print df1_m['Survived'].value_counts(1)

df1_f = df1[df1['Sex'] == 'female']
print df1_f['Survived'].value_counts()
print df1_f['Survived'].value_counts(1)
