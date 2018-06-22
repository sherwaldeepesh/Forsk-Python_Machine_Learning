# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:14:57 2018

@author: deepe
"""
inString = raw_input()
strippedString=inString.strip()
spaceLocation=strippedString.find(" ")
print strippedString[spaceLocation:].strip(),strippedString[:spaceLocation].strip()