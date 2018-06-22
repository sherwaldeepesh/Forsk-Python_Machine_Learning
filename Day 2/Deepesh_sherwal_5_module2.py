# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:09:47 2018

@author: deepe
"""

a=input()
b=0
c=list(a)
c.sort()
c.pop()
c.pop(0)
print c
for i in c:
    b+=i
print b//len(c)