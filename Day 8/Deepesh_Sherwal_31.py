# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:56:37 2018

@author: deepe
"""

import re

list1=[]
while True:
    String=raw_input()
    if not String:
        break
    list1.append(String)

rex = re.compile(r'^[\w-]+@[0-9a-zA-Z]+\.[A-Za-z]{2,4}')
for i in list1:
    response = rex.match(i)
    if response:
        print True
    else:
        print False