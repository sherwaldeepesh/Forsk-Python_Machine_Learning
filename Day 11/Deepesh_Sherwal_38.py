# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:00:08 2018

@author: deepe
"""

import numpy as np
import matplotlib.pyplot as plt
randomData = np.random.normal(150,20,1000)
plt.hist(randomData, 100)
print 'Standard Deviation =', np.std(randomData)
print 'Variance =', np.var(randomData)