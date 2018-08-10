# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:45:01 2018

@author: deepe
"""
import pandas as pd
from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
from datetime import datetime, time, timedelta
import tweepy
import time as tm


consumer_key = 'abbgqUAkQAfRtpy2OhYmkRK1b'
consumer_secret = 'Jo0lpXUXUdp0tSHJzSGqDACmn8pkyQ9CxJ4ywlqwnoWcwA89y0'
access_token = '4716688098-IpRizG3hIZbjkrEoj1tJLmHzW7Vxc6sy807h92X'
access_token_secret = 'Edap1yfqb2X8wbPsdViXpIbpe3TD7EM1fSlJKFU2b3Mu5'

auth = OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
auth_api = API(auth)


data = pd.read_csv('            ',header = None)  #put csv file which have list of users

account_list = [i for i in data[1]]  #Put account details into it

list_count = []
verified_account = []
user_name = []
url_count = []
hashtags_diversity = []
user_tag_diversity = []
likes_no = []
retweet_no = []
friends_following_ratio = []
tweets_frequency = []
lexical_diversity = []

age_of_account = []
    
if len(account_list) > 0:
    for target in account_list:
        try:
            print('getting data for '+ target)
            item = auth_api.get_user(target)
            
            verified = item.verified
            screen_name = item.screen_name
            followers_count = item.followers_count
            following_count = item.friends_count
            listed_count = item.listed_count
            if followers_count + following_count>0:
                ff_ratio = (float(followers_count)/float(followers_count + following_count))
            else:
                ff_ratio = 0
            total_tweets = item.statuses_count
            account_created_date = item.created_at
            delta = datetime.utcnow() - account_created_date
            account_age_days = delta.days
            print ('account_age '+str(account_age_days))
            if account_age_days > 0:
                 tweet_frequency = '%.2f'%(float(total_tweets)/float(account_age_days))
                
                
            hashtags = []
            mentions = []
            tweet = []
            retweet_count = 0
            favorite_count = 0
            urls = 0
            tweet_count = 0
            end_date = datetime.utcnow() - timedelta(days = 30)
            for status in Cursor(auth_api.user_timeline,id = target).items():
                tweet_count += 1
                tweet.extend(str(status.text.encode('utf8')).split())
                          
                retweet_count += status.retweet_count
                favorite_count += status.favorite_count
                
                
                if hasattr(status, "entities"):
                    entities = status.entities
                    if "hashtags" in entities:
                        for ent in entities["hashtags"]:
                            if ent is not None:
                                if "text" in ent:
                                    hashtag = ent["text"]
                                    if hashtag is not None:
                                        hashtags.extend(str(hashtag.encode('utf8')).split())
                    if "user_mentions" in entities:
                        for ent in entities["user_mentions"]:
                            if ent is not None:
                                if "screen_name" in ent:
                                    name = ent["screen_name"]
                                    if name is not None:
                                        mentions.extend(str(name.encode('utf8')).split())
                                        
                    if "urls" in entities:
                        for url in entities["urls"]:
                            if url is not None:
                                if 'url' in url:
                                    url_ = url['url']
                                    if url_ is not None:
                                        urls+=1
                                        
                                        
                if status.created_at < end_date:
                    break
            if len(hashtags)>0:
                hashtag_diversity = float(len(set(hashtags)))/float(len(hashtags))
            else:
                hashtag_diversity = 0
            
            if len(mentions)>0:
                user_mention_diversity = float(len(set(mentions)))/float(len(mentions))
            else:
                user_mention_diversity = 0
                
            if len(tweet)>0:    
                lexicon_diversity = float(len(set(tweet)))/float(len(tweet))
            else:
                lexicon_diversity = 0
                
            list_count.append(listed_count)
            verified_account.append(verified)
            user_name.append(screen_name)
            url_count.append(urls)
            hashtags_diversity.append(hashtag_diversity)
            user_tag_diversity.append(user_mention_diversity)
            likes_no.append(favorite_count)
            retweet_no.append(retweet_count)
            friends_following_ratio.append(ff_ratio)
            tweets_frequency.append(tweet_frequency)
            lexical_diversity.append(lexicon_diversity)
            age_of_account.append(account_age_days)    
            
        except tweepy.TweepError:
            pass
        except tweepy.RateLimitError:
            print('sleep 15 minutes')
            tm.sleep(900)
            continue

dataset = pd.DataFrame()
dataset['user_name'] = user_name
dataset['list_count'] = list_count
dataset['verified_account'] = verified_account
dataset['url_count'] = url_count
dataset['hashtag_diversity'] = hashtags_diversity
dataset['user_tag_diversity'] = user_tag_diversity
dataset['likes_no'] = likes_no
dataset['retweet_no'] = retweet_no
dataset['friends_following_ratio'] = friends_following_ratio
dataset['tweet_frequency'] = tweets_frequency
dataset['lexical_diversity'] = lexical_diversity
dataset['age_of_account'] = age_of_account      
        
dataset.to_csv('             ',sep=',')        #name file which wants to save data in csv
