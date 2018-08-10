# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 23:07:34 2018

@author: deepe
"""

from  selenium import webdriver
import urllib2
from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException
#driver = webdriver.chrome("/Users/rohitmishra/Downloads/chromedriver")

browser = webdriver.Chrome("C:/Users/deepe/Downloads/chromedriver.exe")
url = "https://www.tutorialspoint.com/cprogramming/"
browser.get(url)

A = []
while True:
    if len(A)<35:
        try:
            link = browser.find_element_by_class_name("nxt-btn")
        except NoSuchElementException:
            break
        link.click()
        A.append(browser.current_url)



B = []
for i in range(0,len(A)):    
    data=B[i]
    page=urllib2.urlopen(data)
    soup=BeautifulSoup(page)
    user_name_list = soup.find_all(class_ = 'col-md-7 middle-col')


data=A[1]
page=urllib2.urlopen(data)
soup=BeautifulSoup(page)
user_name_list = soup.find_all(class_ = 'col-md-7 middle-col')
fn = user_name_list[0].text.strip().encode()


from HTMLParser import HTMLParser

# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print "Encountered a start tag:", tag

    def handle_endtag(self, tag):
        print "Encountered an end tag :", tag

    def handle_data(self, data):
        print "Encountered some data  :", data

# instantiate the parser and fed it some HTML
parser = MyHTMLParser()
parser.feed('<html><head><title>Test</title></head>'
            '<body><h1>Parse me!</h1></body></html>')