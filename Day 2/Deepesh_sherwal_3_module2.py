# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:53:30 2018

@author: deepe
"""

a=raw_input()
c=list(a)
b=set(a)
d=list(b)
dict1={}
for i in range(len(d)):
    dict1[d[i]]=c.count(d[i])
print dict1
