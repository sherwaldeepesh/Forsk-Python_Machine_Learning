# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:51:38 2018

@author: deepe
"""

import numpy as np
import matplotlib.pyplot as plt

incomes = np.random.normal(100.0,20.0,10000)
print incomes

plt.hist(incomes, 50)

print 'Mean =',np.mean(incomes)

print 'Median =',np.median(incomes)

incomes = np.append(incomes, [10000000])

print 'Mean =',np.mean(incomes)

print 'Median =',np.median(incomes)