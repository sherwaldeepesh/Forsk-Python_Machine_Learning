# -*- coding: utf-8 -*-
"""
Created on Thu May 17 10:12:43 2018

@author: deepe
"""

String=raw_input()

String.lower()
b=list(set(String))

def alphabet(c):
    count=0
    letters=map(chr,range(97,123))
    for i in c:
        if i in letters:
            count+=1
            continue
    if count==26:
        print 'PANGRAM'
    else:
        print 'NOT PANGRAM'

alphabet(b)