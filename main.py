import nltk
import pprint
import random
import numpy as np
# from positive_tweets_collection import positive_tweets
# from negative_tweets_collection import negative_tweets
from tweets_pre_processing import *
from feature_extraction import *
from logistic_regression import *

## Constants ##
TRAINING_EXAMPLES_RATIO = 0.75
ALPHA = 0.1
ITERATIONS = 3000
THRESHOLD = 0.5

## Preprocess the tweets ##
filtered_positive_tweets = filter_tweets(positive_tweets)
filtered_negative_tweets = filter_tweets(negative_tweets)

## Split the data into training and testing ##
print("splitting")
number_training_positive_examples = int(TRAINING_EXAMPLES_RATIO * len(positive_tweets))
number_training_negative_examples = int(TRAINING_EXAMPLES_RATIO * len(negative_tweets))

training_tweets = word_filter(positive_tweets[:number_training_positive_examples], negative_tweets[:number_training_negative_examples])
random.shuffle(training_tweets)

test_tweets = word_filter(positive_tweets[number_training_positive_examples:], negative_tweets[number_training_negative_examples:])
random.shuffle(test_tweets)

## Train the logistic regression classifier ##
print("preparing logistic regression training")
training_featureset = nltk.classify.apply_features(lr_extract_features, training_tweets)

training_matrix = np.array( [x for (x,y) in training_featureset] )
classes_vector = np.array( [[1] if y=='positive' else [0] for (x,y) in training_featureset] )

n = training_matrix.shape[1] # number of features
m = len(training_matrix) # number of training documents

training_matrix = np.concatenate((np.ones((m,1)), training_matrix), axis = 1)

lr_classifier = np.zeros((n+1,1))

print("training...")
print("initial cost function: %.2f"%cost_function(training_matrix, classes_vector, lr_classifier))

lr_classifier = gradient_descent(training_matrix, classes_vector, lr_classifier, ALPHA, ITERATIONS)

print("done")
print("final cost function: %.2f"%cost_function(training_matrix, classes_vector, lr_classifier))

## Train the naive bayes classifier ##
print('preparing naive bayes training')
training_featureset = nltk.classify.apply_features(nb_extract_features, training_tweets)

print('training...')

nb_classifier = nltk.NaiveBayesClassifier.train(training_featureset)

print('done')


## Classify unseen tweets ##

## Assess the results ##
