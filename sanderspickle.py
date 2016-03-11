from pymongo import MongoClient
import pickle
import pandas as pd

def tweetList():
    dates = []
    text = []
    fails = []

    client = MongoClient('localhost', 27017)
    db = client.twitter_db
    col = db.twitter_collection
    cursor = col.find()

    for i in range(cursor.count()):
        x = cursor.next()
        try:
            dates.append(x["date"])
            text.append(x["text"])
        except:
            fails.append(i)

    df = pd.DataFrame([dates,text]).T
    df.columns = ["Date","Tweet"]
    df.Date = pd.to_datetime(df.Date)
    return df


if __name__ == '__main__':
    df = tweetList()
    df.to_csv('political_tweets.csv')
