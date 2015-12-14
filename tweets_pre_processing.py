def filter_tweets(original_tweets):
    tweets = []
    for (words, sentiment) in original_tweets:
        filter_1 = [w.rstrip(',.!?…-:"').lstrip('"') for w in words.split() if not (w.startswith('RT') or
                                                                     w.startswith('@') or
                                                                     w.startswith('http') or
                                                                     w.startswith('#'))]
        filter_2 = [w.lower() for w in filter_1 if len(w) >= 3 and w.isalpha()]
        tweets.append((filter_2, sentiment))

    return tweets
