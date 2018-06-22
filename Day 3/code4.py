# -*- coding: utf-8 -*-
"""
Created on Tue May 15 14:16:27 2018

@author: deepe
"""


sum1 = 0

dictionary = input()



def fix_teen(n):
    list1 = [13,14,17,18,19]
    if n in list1:
        return 0
    else:
        return n
    
    
    
for i in dictionary.values():
    
    sum1 = sum1+fix_teen(i)
    
    
print sum1


