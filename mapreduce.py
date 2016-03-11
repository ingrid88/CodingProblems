
import sys
import collections
# in_data = sys.stdin


def map(str):
    word_list = str.split()
    tuple_words = [(word,1) for word in word_list]
    return tuple_words

def reduce(t_list):
    d = collections.defaultdict(int)
    for k,v in t_list:
        d[k] += v
    return dict(d)
