import tweepy
import json
from pymongo import MongoClient
from requests_oauthlib import OAuth1

# Authentication details. To  obtain these visit dev.twitter.com
import cnfg
config = cnfg.load(".twitter_config")

oauth = OAuth1(config["consumer_key"],
               config["consumer_secret"],
               config["access_token"],
               config["access_token_secret"])

# This is the listener, resposible for receiving data
class StdOutListener(tweepy.StreamListener):
    def on_data(self, data):

        # Also, we convert UTF-8 to ASCII ignoring all bad characters sent by users
        client = MongoClient('localhost', 27017)
        db = client['twitter_db']
        collection = db['twitter_collection2']
        tweet = json.loads(data)
        # We only want to store tweets in English
        if "lang" in tweet and tweet["lang"] == "en":
            # Store tweet info into the cooltweets collection if not retweeted.
            if "retweeted" in tweet and not tweet["retweeted"] and 'RT @' not in tweet['text']::
                collection.insert({text: tweet["text"], date: tweet[""],
                favorites_count: tweet["favourites_count"],
                retweet_count: tweet["retweet_count"]})

        return True

    def on_error(self, status):
        print(status)

if __name__ == '__main__':
    l = StdOutListener()
    auth = tweepy.OAuthHandler(config["consumer_key"],
                               config["consumer_secret"])
    auth.set_access_token(config["access_token"],
                          config["access_token_secret"])

    stream = tweepy.Stream(auth, l)
    keys = ["obama","Obama", "barack"]
    stream.filter(track=keys)
