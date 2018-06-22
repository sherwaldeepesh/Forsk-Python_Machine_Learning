# -*- coding: utf-8 -*-
"""
Created on Thu May 17 12:05:35 2018

@author: deepe
"""

import urllib2
import requests
def sending():
    url = "http://13.127.155.43/api_v0.1/sending"
    values ={"Phone_Number" : "9772375915", "Name" : "Deepesh Sherwal", "College_Name" : "Anand ICE", "Branch" : "Btech CSE"}
    req = requests.post(url = url,json = values)
    response = req.text
    print response   

def recieving():
    url = "http://13.127.155.43/api_v0.1/receiving?Phone_Number=9772375915"
    req=urllib2.urlopen(url)
    response = req.read()
    print response

sending()
recieving()
