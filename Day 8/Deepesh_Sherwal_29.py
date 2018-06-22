# -*- coding: utf-8 -*-
"""
Created on Tue May 22 11:40:30 2018

@author: deepe
"""

import re
list1=[]
while True:
    String=raw_input()
    if not String:
        break
    list1.append(String)
    
rex = re.compile(r'^[+-]?[0-9]+\.\d+$' or r'^[+-]?\.\d+$')
for i in list1:
    response = rex.match(i)
    if response:
        print True
    else:
        print False