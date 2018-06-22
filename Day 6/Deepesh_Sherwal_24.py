# -*- coding: utf-8 -*-
"""
Created on Fri May 18 13:48:27 2018

@author: deepe
"""
import tweepy

consumer_key ="q2sNpVVg3O4IS4igisZKD2hK0"
consumer_secret ="ILp61HCAWOYiZPcAp7X2mtv3ltFOSG7XyTtdYCnIQD8ovnCnWT"
key ="4716688098-RDEWhR53iZKYd0rNXexwdGeW7futq7ASjUfZ6tt"
secret ="vIkCzh4LNsaqTRMvgsfLEVP5S6iMW2SXM7Yodtg1BVvJd"
 
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(key, secret)
api = tweepy.API(auth)
# update the status
api.update_status('Welcome to Python Api')
