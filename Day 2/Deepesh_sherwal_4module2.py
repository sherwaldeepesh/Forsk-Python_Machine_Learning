# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:51:19 2018

@author: deepe
"""

a=raw_input()

letter=0
digit=0
for i in a:
    if i.isalpha():
        letter+=1
    elif i.isdigit():
        digit+=1
print 'Letters',letter
print 'Digits',digit