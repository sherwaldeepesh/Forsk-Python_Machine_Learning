# -*- coding: utf-8 -*-
"""
Created on Mon May 21 13:14:52 2018

@author: deepe
"""
list1=[]
while True:
    string=raw_input().lower()
    if not string:
        break
    string.split('\n')
    a=string.split('@')

    b=a[1].split('.')
    if len(a)==2:
        if a[0].isalnum() or a[0].isalpha() or '-' in a[0] or '_' in a[0]:
            b=a[1].split('.')
            if len(b)==2:
                if b[0].isalnum() or b[0].isalpha():
                    if len(b[1])<=3:
                        list1.append(string) 
print list1