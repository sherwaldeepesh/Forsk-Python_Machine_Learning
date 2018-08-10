# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:32:42 2018

@author: THAKUR
"""
import pandas as pd 
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException

driver = webdriver.Chrome("C:/Users/THAKUR/Downloads/chromedriver_win32/chromedriver.exe")
url = 'https://botwiki.org/bot/?networks=twitter-bots'
driver.get(url)
D=[]
while True:
    try:
        link = driver.find_element_by_class_name('next')
    except NoSuchElementException:
        break
    link.click()
    D.append(driver.current_url)
    
import urllib2
A = []
for i in range(0,len(D)):    
    tweet=D[i]
    page=urllib2.urlopen(tweet)
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(page)
    user_name_list = soup.find_all(class_ = 'col-sm-12 col-md-8')

    for i in range(1,len(user_name_list)):
        for user_name in user_name_list[i]('a'):
            A.append(user_name.text)
user_name_is=pd.DataFrame()
user_name_is['screen_name'] = A    
user_name_is.to_csv('bots.csv', sep=',')