# -*- coding: utf-8 -*-
"""
Created on Mon May 21 13:02:33 2018

@author: deepe
"""

positiveInteger=map(str,raw_input().split(' '))
all(int(i)>=0 for i in positiveInteger) and any(i==i[::-1] for i in positiveInteger)
