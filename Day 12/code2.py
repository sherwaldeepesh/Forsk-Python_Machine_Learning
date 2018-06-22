# -*- coding: utf-8 -*-
"""
Created on Mon May 28 13:31:03 2018

@author: deepe
"""

import pandas as pd
df1 = pd.read_csv('Automobile.csv')

import matplotlib.pyplot as plt

from collections import Counter

sortedValues = Counter(df1['make'])
new = sortedValues.most_common()
x = [new[i][0] for i in range(len(new))]
y = [new[i][1] for i in range(len(new))]
explode = [i*0 for i in range(len(new))]
explode[0] = 0.1
plt.pie(y,labels = x,explode = explode)
plt.show()