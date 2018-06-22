# -*- coding: utf-8 -*-
"""
Created on Fri May 25 13:24:16 2018

@author: deepe
"""
import numpy as np
Input = map(int, raw_input().split())
a = np.mat(Input)
print a.reshape(3,3)