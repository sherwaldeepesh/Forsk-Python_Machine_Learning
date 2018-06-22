# -*- coding: utf-8 -*-
"""
Created on Tue May 15 11:22:08 2018

@author: deepe
"""

in_number=input()

def Add(x,y):
    return x+y

def Multiply(x,y):
    return x*y

def Largest(x,y):
    return max(x,y)

def Smallest(x,y):
    return min(x,y)

def Sorting(x):
    return x.sort()

def Remove_duplicates(x):
    return list(set(x))

def Print():
    Sorting(in_number)    
    print 'Sum =', reduce(Add,in_number)
    print 'Multiply =', reduce(Multiply,in_number)
    print 'Largest =', reduce(Largest,in_number)
    print 'Smallest =', reduce(Smallest,in_number)
    print 'Sorted= ',in_number
    print 'Without Duplicates =',Remove_duplicates(in_number)
Print()