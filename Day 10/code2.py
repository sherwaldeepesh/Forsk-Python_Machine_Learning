# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:39:39 2018

@author: deepe
"""

import pandas as pd
df2 = pd.read_csv('training_titanic.xls')


df2 = pd.DataFrame.from_csv('training_titanic.xls')
df2["child"]=0

df2['child'][df2["Age"]<18] = 1
 


df2_c = df2[df2['child'] == 0]
print 'Childs having age >= 18:\n', df2_c['Survived'].value_counts(1)

df2_c = df2[df2['child'] == 1]
print 'Childs having age < 18:\n',df2_c['Survived'].value_counts(1)