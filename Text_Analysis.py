import pickle
import pandas as pd


x = open("/Users/ingrid/Desktop/yellp/review_data_10_percent.csv", 'r')

df = pd.DataFrame.from_csv(x)
text = df.text
subset_text = df.text.sample(frac=.1)



## Build arrays out of timestamp and tweet content
date = []
text = []
shit = []
for line in x:
    row = line.split(',')
    try:
        text.append(row[2])
        date.append(row[1])
    except:
        shit.append(row)


# In[ ]:

## build Pandas DataFrame
df = pd.DataFrame([date,text]).T


# In[ ]:

## Subselect from df
ndf = df.sample(frac=0.03)


# ## Clean Data

# In[14]:

import string
from nltk.stem import WordNetLemmatizer
from functools import lru_cache

wnl = WordNetLemmatizer()
lemmatize = lru_cache(maxsize=50000)(wnl.lemmatize)

def lemmas(tweet):
    return ' '.join([lemmatize(word) for word in tweet.split(" ")])


def collect(tweet,char):
    """Input: tweet and Output: hashtags"""
    collection = []
    for word in tweet.split(" "):
        if len(word) > 0 and word[0:len(char)] == char:
            collection.append(word[len(char):len(word)].rstrip('\n'))
    return collection

collect = lru_cache(maxsize=50000)(collect)


def remove_char(tweet,char):
    """Input: tweet and Output: tweet without specified char"""
    words = [word for word in tweet.split(" ") if len(word) > 0]
    return ' '.join([word for word in words if word[0:len(char)] != char])

remove_char = lru_cache(maxsize=50000)(remove_char)

def count_chars(tweet,char):
    """Input: tweet and Output: char count"""
    return len([letter for letter in tweet if letter == char])


def remove_punct(tweet):
    """Input: tweet and Output: tweet without specified char"""

    exclude = string.punctuation #set([',','.','!',';',":","?",""])
    return ''.join(ch for ch in tweet if ch not in exclude)


def remove_chars(tweet,char_list):
    """Input: tweet and Output: tweet without specified char"""
    exclude = char_list
    return ''.join(ch for ch in tweet if ch not in exclude)


def lowercase(tweet):
    """Input: tweet and Output: tweet without specified char"""
    return tweet.lower()

def remove_numbers(tweet):
    return ''.join(ch for ch in tweet if not ch.isdigit())

remove_numbers = lru_cache(maxsize=50000)(remove_numbers)


def clean_tweet(tweet):
    new_tweet = remove_punct(tweet)
    new_tweet = remove_chars(new_tweet,['#','@','\n','&',"\""])
    new_tweet = remove_char(new_tweet,'https')
    new_tweet = lowercase(new_tweet)
    new_tweet = remove_numbers(new_tweet)
    new_tweet = lemmas(new_tweet)
    return new_tweet


# In[ ]:

## Run the cleaner
ndf['clean'] = ndf[1].apply(lambda tweet: clean_tweet(tweet))


# In[ ]:

ndf.head()


# ## tokenize

# In[16]:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import string
import re


# In[17]:

# A custom stoplist
STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "“", "”", "'ve"]


# In[18]:

from spacy.en import English
parser = English()


# In[19]:

# Every step in a pipeline needs to be a "transformer".
# Define a custom transformer to clean text using spaCy
class CleanTextTransformer(TransformerMixin):
    """
    Convert text to cleaned text
    """

    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# A custom function to clean the text before sending it into the vectorizer
def cleanText(text):
    # get rid of newlines
    text = text.strip().replace("\n", " ").replace("\r", " ")

    # lowercase
    text = text.lower()

    return text

# A custom function to tokenize the text using spaCy
# and convert to lemmas
def tokenizeText(sample):

    # get the tokens using spaCy
    tokens = parser(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")

    return tokens

# tokenizeText = lru_cache(maxsize=50000)(tokenizeText)


def printNMostInformative(vectorizer, clf, N):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefs_with_fns[:N]
    topClass2 = coefs_with_fns[:-(N + 1):-1]
    print("Class 1 best: ")
    for feat in topClass1:
        print(feat)
    print("Class 2 best: ")
    for feat in topClass2:
        print(feat)



# In[ ]:




# ### CountVectorizer

# In[21]:

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(encoding='utf-8', decode_error='strict',
                             strip_accents=None, lowercase=True, preprocessor=None,
                             tokenizer=tokenizeText, stop_words=STOPLIST, token_pattern='(?u)\b\w\w+\b',
                             ngram_range=(1, 1), analyzer='word', max_df=1000, min_df=30,
                             max_features=None)


# In[22]:

X = vectorizer.fit_transform(subset_text)   ##list(ndf.clean))


# In[ ]:

X.shape


# In[ ]:

vectorizer.get_feature_names()


# In[ ]:




# In[ ]:

## spectral Clustering


# In[ ]:

# from sklearn.clustering import SpectralClustering
from sklearn import cluster, datasets
spectral = cluster.SpectralClustering(n_clusters=2)


# In[ ]:

spectral.fit_predict(X)


# In[ ]:




# ### TFIDF

# In[ ]:

from sklearn.feature_extraction.text import TfidfVectorizer
transformer = TfidfVectorizer(stop_words = 'english', tokenizer=tokenizeText, max_df=1000, min_df=3)


# In[ ]:

t = transformer.fit_transform(list(ndf.clean))   #direclty from corpus


# In[ ]:

from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
# sim_matrix = pairwise_distances(t,metric='euclidean')


# In[ ]:

dist_matrix = cosine_similarity(t)


# In[ ]:

# visualize network
# topic analysis: LDA -- with the sparse matrix


# In[ ]:

ndf.clean.iloc[0:10]


# In[ ]:

ndf.clean.iloc[2]


# In[ ]:

pd.DataFrame(dist_matrix).stacked()#.unstacked() #.sort_values(ascending=False)


# # Clustering

# In[ ]:

from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation()


# In[ ]:

from sklearn.cluster import KMeans


# In[ ]:

## lda gives the probability distribution for terms in each topic
## sort the terms by probability : print the top ones to see which are the closest

## lda .components = topic word distribution { all of your terms and the probability each term is part of topic}


# In[ ]:

model = lda.fit(t)


# In[ ]:

model


# In[ ]:

feature_names = transformer.get_feature_names
n_top_words = 20


# In[ ]:

model.components_


# In[ ]:

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


# In[ ]:

print_top_words(model,feature_names, n_top_words)


# In[ ]:




# In[ ]:

from sklearn.cluster import SpectralClustering
mat_clust = SpectralClustering(n_clusters=50,affinity='precomputed').fit_predict(t)


# In[ ]:

# sklearn.manifold.MDS (to flatten so you can graph stuff)
from sklearn.manifold import MDA
