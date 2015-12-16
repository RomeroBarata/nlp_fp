import nltk
import pprint
import random
import numpy as np
from feature_extraction import *
from logistic_regression import *
from assessment_metrics import *

## Constants ##
TRAINING_EXAMPLES_RATIO = 0.75
ALPHA = 0.1
ITERATIONS = 5000
THRESHOLD = 0.5
RLAMBDA = 1

## Split the data into training and testing ##
print("Splitting")
number_training_positive_examples = int(TRAINING_EXAMPLES_RATIO * len(positive_tweets))
number_training_negative_examples = int(TRAINING_EXAMPLES_RATIO * len(negative_tweets))

random.seed(1234) # Make the results reproducible
random.shuffle(positive_tweets)
random.shuffle(negative_tweets)
training_tweets = word_filter(positive_tweets[:number_training_positive_examples], negative_tweets[:number_training_negative_examples])
test_tweets = word_filter(positive_tweets[number_training_positive_examples:], negative_tweets[number_training_negative_examples:])

## Train the Logistic Regression classifier ##
print("Preparing the Logistic Regression training...")
training_featureset = nltk.classify.apply_features(lr_extract_features, training_tweets)

training_matrix = np.array( [x for (x,y) in training_featureset] )
classes_vector = np.array( [[1] if y=='positive' else [0] for (x,y) in training_featureset] )

n = training_matrix.shape[1] # number of features
m = len(training_matrix) # number of training tweets

training_matrix = np.concatenate((np.ones((m,1)), training_matrix), axis = 1)

lr_classifier = np.zeros((n+1,1))

print("Training...")
print("Initial cost: %.2f"%cost_function(training_matrix, classes_vector, lr_classifier))

lr_classifier = gradient_descent(training_matrix, classes_vector, lr_classifier, ALPHA, ITERATIONS)

print("Done")
print("Final cost: %.2f"%cost_function(training_matrix, classes_vector, lr_classifier))

## Train the Naive Bayes classifier ##
print('Preparing the Naive Bayes training...')
training_featureset = nltk.classify.apply_features(nb_extract_features, training_tweets)

print('Training...')
nb_classifier = nltk.NaiveBayesClassifier.train(training_featureset)
print('Done')

## Classify unseen tweets ##
test_classes = [y for x, y in test_tweets]

# Logistic Regression #
test_featureset = nltk.classify.apply_features(lr_extract_features, test_tweets)

test_matrix = np.array( [x for (x,y) in test_featureset] )
m = len(test_matrix) # number of test tweets
test_matrix = np.concatenate((np.ones((m,1)), test_matrix), axis = 1)

lr_hypothesis = lr_classify(test_matrix, lr_classifier, THRESHOLD)

# Naive Bayes #
test_featureset = nltk.classify.apply_features(nb_extract_features, test_tweets)
nb_hypothesis = [nb_classifier.classify(tweet) for tweet, c in test_featureset]

## Assess the results ##
# Logistic Regression results
print("Logistic Regression results:")
lr_results = compute_metrics(test_classes, lr_hypothesis, 'positive')
pprint.pprint(lr_results)

# Naive Bayes results
print("Naive Bayes results:")
nb_results = compute_metrics(test_classes, nb_hypothesis, 'positive')
pprint.pprint(nb_results)
