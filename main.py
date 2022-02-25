from nltk.corpus import twitter_samples
from nltk.tag import pos_tag_sents

# Renders each tweet as a string
tweets = twitter_samples.strings("positive_tweets.json")

# Renders each string into a list of parseable tokens
tweets_tokens = twitter_samples.tokenized("positive_tweets.json")
print(tweets)