# -*- coding: utf-8 -*-
"""
Created on Fri May 25 13:34:09 2018

@author: deepe
"""

import pandas as pd
import numpy as np
df = pd.read_csv('Automobile.csv')

df["price"] = df["price"].fillna(df["price"].mean())

valuesPrice = np.array(df["price"])

print valuesPrice.max()
print valuesPrice.min()
print np.average(valuesPrice)
print np.std(valuesPrice)