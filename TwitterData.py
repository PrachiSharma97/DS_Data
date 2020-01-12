import os
import tweepy as tw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from nltk.corpus import stopwords

consumer_key= ''
consumer_secret= ''
access_token= '-'
access_token_secret= ''

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Post a tweet from Python
#api.update_status("Look, I'm tweeting from #Python in my #earthanalytics class! @EarthLabCU")
# Your tweet has been posted!

search_words = "#modi"
date_since = "2018-8-01"

tweets = tw.Cursor(api.search,
              q=search_words,
              lang="en",
              since=date_since).items(1000)
tweets

#for tweet in tweets:
#    print(tweet.text)

tweets_list= [tweet.text for tweet in tweets]

new_search = search_words + " -filter:retweets"

tweets = tw.Cursor(api.search,
                       q=new_search,
                       lang="en",
                       since=date_since).items(1000)

newTweet=[tweet.text for tweet in tweets]

clean_tweets=[]

for i in newTweet:
    tweet = re.sub('@[\w]*', ' ', i)
    tweet = re.sub('[^a-zA-Z#]', ' ', tweet)
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words())]
    tweet = ' '.join(tweet)
    clean_tweets.append(tweet)             

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3000)
X= cv.fit_transform(clean_tweets)
X=X.toarray()
y= []
for i in range(1000):
    y.append(np.random.randint(0,2))

print(cv.get_feature_names())

from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()
n_b.fit(X,y)
n_b.score(X,y)



