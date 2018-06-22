# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:28:51 2018

@author: deepe
"""
a=raw_input()
def translate(x):
    if x in ['a','e','i','o','u',' ']:
        return x
    else:
        return x+'o'+x

string=''  
for i in a:
    string+=translate(i) 
    
print string