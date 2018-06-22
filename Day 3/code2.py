# -*- coding: utf-8 -*-
"""
Created on Tue May 15 13:19:21 2018

@author: deepe
"""

small_brick,large_brick,target_size=input()
d=target_size//5
e=target_size%5
if large_brick>=d and small_brick>=e:
    print True
else:
    print False