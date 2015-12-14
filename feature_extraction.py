import nltk
from positive_tweets import *
from negative_tweets import *

def word_filter(positive_tweets, negative_tweets):
    tweets = []
    for (words, sentiment) in positive_tweets + negative_tweets:
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        tweets.append( (words_filtered, sentiment) )
    return tweets

def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

def select_relevant_words(word_features):
    file = open('lexico_v3.0.txt','r')
    lexicon = [row.strip().split(',') for row in file]
    non_neutral_words = {entry[0] for entry in lexicon if entry[2] != '0'}
    relevant_words = [word for word in word_features if word in non_neutral_words]
    return relevant_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = [i for (i,j) in sorted(wordlist.items(), key=lambda t: t[1], reverse=True)]
    return select_relevant_words(word_features)

word_features = get_word_features(get_words_in_tweets(word_filter(positive_tweets,negative_tweets)))

def NB_extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features

def LR_extract_features(document):
    document_words = set(document)
    features = []
    for word in word_features:
        features.append(1 if word in document_words else 0)
    return features
