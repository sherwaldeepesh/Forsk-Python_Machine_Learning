# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:11:54 2018

@author: deepe
"""
def f(x):
    if x%3==0:
        if x%5==0:
            print 'FizzBuzz'
        print 'Fizz'
    elif x%5==0:
        print 'Buzz'
    else:
        print x

filter(f,range(1,51))