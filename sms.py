# -*- coding: utf-8 -*-
"""
Created on Mon May 28 23:51:01 2018

@author: deepe
"""

import urllib2
import requests
import json

url = "http://sms.dataoxytech.com/index.php/smsapi/httpapi/?uname=sylvester007&password=forskmnit&sender=FORSKT&receiver=9772375915&route=TA&msgtype=1&sms=Hey there"










request = urllib2 
f = urllib2.urlopen(request)

fr = f.read()
return fr









req = urllib2.urlopen(url)
try:
    response = urllib2.urlopen(req)
    kittens = response.read()
    print kittens
except urllib2.URLError:
    print 'error'
    
    
    
from suds.client import Client
url="http://sms.dataoxytech.com/index.php/smsapi/httpapi/?uname=sylvester007&password=forskmnit&sender=FORSKT&receiver=9772375915&route=TA&msgtype=1&sms=Hey there"
client = Client(url)
print client

    
response = json.loads(urllib2.urlopen(req).read())
if response['response'] != "done\n":
    print "Error"
else:
    print "Message sent successfully"


import urllib2
import json
phone = raw_input("Enter receiver's number: ")
msg = raw_input("Enter the message to send: ")
headers = { "X-Mashape-Authorization": "<Your API key at Mashape>" }
url = "https://160by2.p.mashape.com/index.php?msg="+msg+"&phone="+phone+"&pwd=your < password>&uid=<your user id>"
req = urllib2.Request(url, '', headers)
response = json.loads(urllib2.urlopen(req).read())
if response['response'] != "done\n":
    print "Error"
else:
    print "Message sent successfully"