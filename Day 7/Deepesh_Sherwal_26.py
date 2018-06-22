# -*- coding: utf-8 -*-
"""
Created on Mon May 21 12:03:00 2018

@author: deepe
"""
from collections import OrderedDict
dict1=OrderedDict()
while True:
    string=raw_input()
    if not string:
        break
    tup=string.split(' ')
    
    key = " ".join(tup[:-1]).upper()
    value = int(tup[-1])
    
    #list1.append((" ".join(tup[:-1]).upper(),int(tup[-1])))
    
    if key in dict1:
        dict1[key] = dict1[key] + value
    else:
        dict1[key] = value

for k,v in dict1.items():
    print k,v
