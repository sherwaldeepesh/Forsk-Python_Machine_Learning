# -*- coding: utf-8 -*-
"""
Created on Fri May 18 11:27:58 2018

@author: deepe
"""

import oauth2
import time
import urllib2
import json

url1 = "https://api.twitter.com/1.1/search/tweets.json"
params = {
        "oauth_version": "1.0",
        "oauth_nonce": oauth2.generate_nonce(),
        "oauth_timestamp": int(time.time())
    }
consumer = oauth2.Consumer(key="uVIqbpCMHnz4sMqSqCnZ1QJYx", secret="DC4rPixgLvOL17OKyg7OaIpaVUGz2nBpOmulc6PJdf6piNEuc4")
token = oauth2.Token(key="4716688098-RDEWhR53iZKYd0rNXexwdGeW7futq7ASjUfZ6tt", secret="vIkCzh4LNsaqTRMvgsfLEVP5S6iMW2SXM7Yodtg1BVvJd")
req = params["oauth_consumer_key"] = consumer.key

params['q'] = "Jaipur"
req = oauth2.Request(method = "GET", url = url1, parameters=params)
signature_method = oauth2.SignatureMethod_HMAC_SHA1() 
req.sign_request(signature_method, consumer, token)
url = req.to_url()
response = urllib2.Request(url)
data = json.load(urllib2.urlopen(response))

filename = params["q"]      
f = open(filename + "_File.txt", "w")
json.dump(data["statuses"], f)
f.close()