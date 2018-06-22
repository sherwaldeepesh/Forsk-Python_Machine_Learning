# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:03:25 2018

@author: deepe
"""

import numpy as np

randomArray =  np.random.random_integers(5,15,40)

countFrequent= np.bincount(randomArray)

print np.argmax(countFrequent),'is',np.max(countFrequent),'times'

from collections import Counter
cnt = Counter(randomArray)
print cnt.most_common()[0][0]