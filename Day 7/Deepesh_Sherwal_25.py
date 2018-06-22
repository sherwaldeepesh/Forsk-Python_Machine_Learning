# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:11:20 2018

@author: deepe
"""

list1=[]
while True:
    string=raw_input()
    if not string:
        break
    tup=string.split(',')
    list1.append((tup[0],int(tup[1]),int(tup[2])))
list1.sort()
print list1