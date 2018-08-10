# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:56:05 2018

@author: THAKUR
"""
import urllib2
A = []
b=["https://botwiki.org/bot/?networks=twitter-bots","https://botwiki.org/bot/page/2/?networks=twitter-bots"]
for i in range(0,len(b)):    
    tweet=b[i]
    page=urllib2.urlopen(tweet)
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(page)
    user_name_list = soup.find_all(class_ = 'col-sm-12 col-md-8')

    for i in range(1,len(user_name_list)):
        for user_name in user_name_list[i]('a'):
            A.append(user_name.text)
    
b=["https://botwiki.org/bot/?networks=twitter-bots","https://botwiki.org/bot/page/2/?networks=twitter-bots"]
user_name_list[1]
user_name_list