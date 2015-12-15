import nltk
from positive_tweets_collection import positive_tweets
from negative_tweets_collection import negative_tweets

# word_filter will discard words with len <= 2 and put them to lowercase
# furthermore it will transform the phrases into arrays of words
def word_filter(positive_tweets, negative_tweets):
    tweets = []
    for (words, sentiment) in positive_tweets + negative_tweets:
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        tweets.append( (words_filtered, sentiment) )
    return tweets

# return all the words in the tweets set
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

# filter the words and return only those which have a positive or negative influence
def select_relevant_words(word_features):
    file = open('lexico_v3.0.txt','r')
    lexicon = [row.strip().split(',') for row in file]
    non_neutral_words = {entry[0] for entry in lexicon if entry[2] != '0'}
    relevant_words = [word for word in word_features if word in non_neutral_words]
    return relevant_words

# return the features to be used in the classifier, that is all the relevant words 
# sorted by order of frequence
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = [i for (i,j) in sorted(wordlist.items(), key=lambda t: t[1], reverse=True)]
    return select_relevant_words(word_features)

# the features to be used in the classifier
word_features = get_word_features(get_words_in_tweets(word_filter(positive_tweets,negative_tweets)))

# a feature extractor for the naive bayes classifier
# it will return a dictionary that says which words are contained in the document
def nb_extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features

# a feature extractor for the logistic regression classifier
# returns a list of tuples ([list of 0s and 1s], class)
def lr_extract_features(document):
    document_words = set(document)
    features = []
    for word in word_features:
        features.append(1 if word in document_words else 0)
    return features
