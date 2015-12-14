import nltk
import pprint
from positive_tweets_collection import positive_tweets
from negative_tweets_collection import negative_tweets
from tweets_pre_processing import *

## Constants ##

## Preprocess the tweets ##
filtered_positive_tweets = filter_tweets(positive_tweets)
filtered_negative_tweets = filter_tweets(negative_tweets)

## Split the data into training and testing ##

## Train a classifier ##

## Classify unseen tweets ##

## Assess the results ##
