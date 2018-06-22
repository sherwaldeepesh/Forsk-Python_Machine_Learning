# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:49:47 2018

@author: deepe
"""

in_num=input()
add=0
for i in range(len(in_num)):
    if(in_num[i]!=13):
        if(in_num[i-1]!=13):
            add=add+in_num[i]   
print add